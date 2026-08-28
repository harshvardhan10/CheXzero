#!/usr/bin/env python3
"""Extract pathology-conditioned CheXzero ViT attention Grad-CAM maps.

For each image x and pathology c, it computes CheXzero's positive/negative
zero-shot similarities

    s_pos = cos(image(x), text(positive prompt for c))
    s_neg = cos(image(x), text(negative prompt for c))

and backpropagates the margin

    target = s_pos - s_neg.

The target is differentiated with respect to the *actual visual self-attention
probabilities used in the forward pass* of one CheXzero visual Transformer
block.  For the selected layer, the final map is

    CAM = mean_heads( ReLU(A_cls->patch * d(target)/dA_cls->patch) ).

This deliberately mirrors an ITC-margin attention-Grad-CAM style extraction:
- pathology conditioned via positive/negative text prompts
- visual self-attention, not text-to-image cross-attention
- gradient weighted
- one map per image/pathology

Output
------
One .pt file per image.  Each requested label stores:
- positive/negative prompt
- positive/negative cosine similarity
- margin
- positive probability = softmax([s_neg, s_pos])[positive]
- signed attention-gradient map
- positive/ReLU map
- normalized low-resolution patch map
- normalized upsampled map

The script also writes:
- chexzero_attention_gradcam_index.csv
- chexzero_attention_gradcam_manifest.json

Notes
-----
The released CheXzero ViT uses torch.nn.MultiheadAttention with
``need_weights=False``.  Therefore a normal forward hook cannot recover the
attention probabilities.  This script instruments exactly one *visual* block
by replacing its ``attention`` method with a mathematically equivalent
self-attention implementation that explicitly retains the softmax attention
probabilities and their gradients.
"""

from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm
import clip
from model import CLIP


# -----------------------------------------------------------------------------
# Released CheXzero architecture / preprocessing defaults
# -----------------------------------------------------------------------------

CHEXZERO_MODEL_PARAMS = {
    "embed_dim": 768,
    "image_resolution": 320,
    "vision_layers": 12,
    "vision_width": 768,
    "vision_patch_size": 16,
    "context_length": 77,
    "vocab_size": 49408,
    "transformer_width": 512,
    "transformer_heads": 8,
    "transformer_layers": 12,
}

# The released zero_shot.py normalizes raw 0..255 CXR values with these stats.
CHEXZERO_CXR_MEAN = (101.48761, 101.48761, 101.48761)
CHEXZERO_CXR_STD = (83.43944, 83.43944, 83.43944)


# -----------------------------------------------------------------------------
# CLI / parsing helpers
# -----------------------------------------------------------------------------

def parse_labels(values: Sequence[str]) -> List[str]:
    """Accept both space-separated and comma-separated labels."""
    labels: List[str] = []
    for value in values:
        labels.extend(part.strip() for part in str(value).split(",") if part.strip())
    if not labels:
        raise ValueError("At least one target label is required")
    return list(dict.fromkeys(labels))


def safe_label_name(label: str) -> str:
    return (
        str(label)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def build_prompt_pair(
    label: str,
    positive_template: str,
    negative_template: str,
) -> Tuple[str, str]:
    """Build one exact positive/negative prompt pair for a pathology."""
    clean = str(label).replace("_", " ").strip()
    if not clean:
        raise ValueError("Encountered an empty label")

    try:
        positive = positive_template.format(clean)
        negative = negative_template.format(clean)
    except Exception as exc:
        raise ValueError(
            "Prompt templates must be Python format strings containing `{}`, "
            "for example --positive_template '{}' --negative_template 'no {}'."
        ) from exc

    return positive.strip(), negative.strip()


def load_prompt_pairs(
    labels: Sequence[str],
    positive_template: str,
    negative_template: str,
    prompts_json: Optional[str],
) -> Dict[str, Dict[str, str]]:
    """Load per-label prompts or construct them from templates.

    --prompts_json can be either:
      1) a path to a JSON file, or
      2) an inline JSON object.

    Expected structure:
      {
        "Cardiomegaly": {
          "positive": "cardiomegaly",
          "negative": "no cardiomegaly"
        }
      }
    """
    supplied: Dict[str, Any] = {}
    if prompts_json:
        candidate = Path(prompts_json)
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as handle:
                supplied = json.load(handle)
        else:
            supplied = json.loads(prompts_json)
        if not isinstance(supplied, dict):
            raise ValueError("--prompts_json must decode to a JSON object")

    result: Dict[str, Dict[str, str]] = {}
    for label in labels:
        if label in supplied:
            entry = supplied[label]
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Prompt entry for {label!r} must be an object with "
                    "'positive' and 'negative'"
                )
            positive = str(entry.get("positive", "")).strip()
            negative = str(entry.get("negative", "")).strip()
            if not positive or not negative:
                raise ValueError(
                    f"Prompt entry for {label!r} must contain non-empty "
                    "'positive' and 'negative' values"
                )
        else:
            positive, negative = build_prompt_pair(
                label, positive_template, negative_template
            )
        result[label] = {"positive": positive, "negative": negative}
    return result


# -----------------------------------------------------------------------------
# Checkpoint / model loading
# -----------------------------------------------------------------------------

def _unwrap_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("CheXzero checkpoint must be a state-dict-like dictionary")

    for key in ("state_dict", "model", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            checkpoint = value
            break

    if not checkpoint:
        raise ValueError("Checkpoint state dict is empty")

    # Common DDP wrapper cleanup.
    keys = list(checkpoint.keys())
    if keys and all(str(key).startswith("module.") for key in keys):
        checkpoint = {str(key)[7:]: value for key, value in checkpoint.items()}

    return checkpoint


def build_chexzero_model(
    checkpoint_path: Path,
    device: torch.device,
    context_length: int,
    pretrained: bool,
) -> torch.nn.Module:
    """Build the same CheXzero model family used by zero_shot_vindr.py."""
    if pretrained:
        # Matches the old evaluator's --pretrained branch exactly.
        model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
    else:
        params = dict(CHEXZERO_MODEL_PARAMS)
        params["context_length"] = int(context_length)
        model = CLIP(**params)
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _unwrap_state_dict(raw_checkpoint)

    try:
        load_result = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint does not match the released CheXzero ViT architecture "
            "(320x320, ViT-B/16-like visual encoder, 12 visual blocks). "
            f"Original load error:\n{exc}"
        ) from exc

    print(f"[Model] Loaded {checkpoint_path}")
    print(f"[Model] state_dict: {load_result}")

    model = model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# CheXzero preprocessing
# -----------------------------------------------------------------------------

def resolve_image_path(images_root: Path, image_id: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = images_root / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No image found for {image_id} under {images_root}")


def load_chexzero_image(
    image_path: Path,
    image_resolution: int,
    pretrained: bool,
    resize_nonpretrained: bool,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load a VinDr PNG with preprocessing matching zero_shot_vindr.py.

    That evaluator converts the RGB PNG to float32 on the original 0..255
    scale, applies CheXzero's Normalize(mean=101.48761, std=83.43944), and only
    resizes when --pretrained is used.  We reproduce that order here.
    """
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
        original_size_wh = tuple(int(value) for value in image.size)
        array = np.asarray(image, dtype=np.float32)

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Image contains NaN/Inf: {image_path}")

    # [H,W,3] -> [3,H,W], preserving the 0..255 intensity scale.
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().float()

    mean = torch.tensor(CHEXZERO_CXR_MEAN, dtype=tensor.dtype)[:, None, None]
    std = torch.tensor(CHEXZERO_CXR_STD, dtype=tensor.dtype)[:, None, None]
    tensor = (tensor - mean) / std

    h, w = int(tensor.shape[-2]), int(tensor.shape[-1])
    if pretrained:
        # Matches old make_transform(pretrained=True): Normalize -> Resize.
        tensor = TF.resize(
            tensor,
            [image_resolution, image_resolution],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    elif (h, w) != (image_resolution, image_resolution):
        if not resize_nonpretrained:
            raise ValueError(
                f"{image_path} is {w}x{h}, but the non-pretrained CheXzero ViT "
                f"expects {image_resolution}x{image_resolution}. Your old "
                "zero_shot_vindr.py did not resize in this branch. Preprocess "
                "the PNGs identically, or explicitly pass --resize_nonpretrained "
                "if you intentionally want resizing."
            )
        tensor = TF.resize(
            tensor,
            [image_resolution, image_resolution],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    return tensor.float(), original_size_wh


# -----------------------------------------------------------------------------
# Text encoding
# -----------------------------------------------------------------------------

@torch.no_grad()
def encode_prompt_pairs(
    model: torch.nn.Module,
    prompts: Dict[str, Dict[str, str]],
    labels: Sequence[str],
    device: torch.device,
    context_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode one positive and one negative text embedding per label."""
    positive_prompts = [prompts[label]["positive"] for label in labels]
    negative_prompts = [prompts[label]["negative"] for label in labels]
    all_prompts = positive_prompts + negative_prompts

    tokens = clip.tokenize(all_prompts, context_length=context_length).to(device)
    features = model.encode_text(tokens)
    features = F.normalize(features.float(), dim=-1)

    count = len(labels)
    positive = features[:count]
    negative = features[count:]
    return positive, negative


# -----------------------------------------------------------------------------
# Instrument visual self-attention
# -----------------------------------------------------------------------------

class VisualAttentionInstrument:
    """Expose the actual attention probabilities used by one visual block.

    CheXzero's released ResidualAttentionBlock calls nn.MultiheadAttention with
    need_weights=False, so there is no attention tensor to hook.  This class
    replaces ONLY the selected block's `attention` method with an equivalent
    explicit self-attention calculation.

    The retained tensor has shape:
        [batch, heads, tokens, tokens]
    and it participates directly in attention @ value, so `.grad` after
    backward is the required d(target)/d(attention).
    """

    def __init__(self, block: torch.nn.Module):
        self.block = block
        self.original_attention = block.attention
        self.attention_probs: Optional[torch.Tensor] = None

        mha = getattr(block, "attn", None)
        if mha is None or not isinstance(mha, torch.nn.MultiheadAttention):
            raise TypeError(
                "Selected CheXzero visual block does not contain "
                "torch.nn.MultiheadAttention as `.attn`"
            )

        self._validate_supported_mha(mha)

        instrument = self

        def attention_with_capture(block_self, x: torch.Tensor) -> torch.Tensor:
            return instrument._forward_attention(block_self, x)

        block.attention = types.MethodType(attention_with_capture, block)

    @staticmethod
    def _validate_supported_mha(mha: torch.nn.MultiheadAttention) -> None:
        if mha.in_proj_weight is None:
            raise NotImplementedError("Separate Q/K/V projection weights are unsupported")
        if getattr(mha, "bias_k", None) is not None:
            raise NotImplementedError("bias_k is unsupported")
        if getattr(mha, "bias_v", None) is not None:
            raise NotImplementedError("bias_v is unsupported")
        if bool(getattr(mha, "add_zero_attn", False)):
            raise NotImplementedError("add_zero_attn=True is unsupported")
        if bool(getattr(mha, "batch_first", False)):
            raise NotImplementedError(
                "This script expects CheXzero's sequence-first MultiheadAttention"
            )

    def reset(self) -> None:
        self.attention_probs = None

    def close(self) -> None:
        self.block.attention = self.original_attention
        self.attention_probs = None

    def _forward_attention(
        self,
        block_self: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Equivalent self-attention for CheXzero's visual ResidualAttentionBlock."""
        mha: torch.nn.MultiheadAttention = block_self.attn

        if x.ndim != 3:
            raise ValueError(f"Expected sequence-first [L,B,E], got {tuple(x.shape)}")

        seq_len, batch_size, embed_dim = x.shape
        num_heads = int(mha.num_heads)
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} not divisible by heads={num_heads}"
            )
        head_dim = embed_dim // num_heads

        # CheXzero uses self-attention: query = key = value = x.
        qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # [L,B,E] -> [B,H,L,D]
        q = q.contiguous().view(seq_len, batch_size, num_heads, head_dim)
        k = k.contiguous().view(seq_len, batch_size, num_heads, head_dim)
        v = v.contiguous().view(seq_len, batch_size, num_heads, head_dim)
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # Scaled dot-product attention.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        attn_mask = getattr(block_self, "attn_mask", None)
        if attn_mask is not None:
            mask = attn_mask.to(dtype=scores.dtype, device=scores.device)
            if mask.ndim == 2:
                if tuple(mask.shape) != (seq_len, seq_len):
                    raise ValueError(
                        f"Unexpected 2D attention mask shape {tuple(mask.shape)}"
                    )
                scores = scores + mask[None, None, :, :]
            elif mask.ndim == 3:
                # Included for completeness; visual CheXzero blocks use no mask.
                if mask.shape[0] == batch_size * num_heads:
                    scores = scores + mask.view(
                        batch_size, num_heads, seq_len, seq_len
                    )
                else:
                    raise ValueError(
                        f"Unexpected 3D attention mask shape {tuple(mask.shape)}"
                    )
            else:
                raise ValueError(
                    f"Unsupported attention mask rank: {mask.ndim}"
                )

        attention_probs = torch.softmax(scores, dim=-1)
        if mha.dropout > 0.0 and mha.training:
            attention_probs = F.dropout(
                attention_probs,
                p=float(mha.dropout),
                training=True,
            )

        # CRITICAL: this exact tensor is then used for attention @ value.
        # Therefore its retained gradient is d(target)/d(attention_probs).
        if attention_probs.requires_grad:
            attention_probs.retain_grad()
        self.attention_probs = attention_probs

        attended = torch.matmul(attention_probs, v)  # [B,H,L,D]

        # [B,H,L,D] -> [L,B,E]
        attended = attended.permute(2, 0, 1, 3).contiguous()
        attended = attended.view(seq_len, batch_size, embed_dim)
        output = F.linear(attended, mha.out_proj.weight, mha.out_proj.bias)
        return output


# -----------------------------------------------------------------------------
# CAM computation
# -----------------------------------------------------------------------------

def normalize_positive_map(cam: torch.Tensor) -> torch.Tensor:
    cam = cam.float()
    maximum = cam.max()
    if torch.isfinite(maximum) and float(maximum) > 0.0:
        return cam / maximum
    return torch.zeros_like(cam)


def compute_attention_gradcam(
    attention: torch.Tensor,
    gradients: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Convert [B,H,N,N] attention + gradients into a 2-D patch map."""
    if attention.ndim != 4:
        raise ValueError(
            f"Expected attention [B,H,N,N], got {tuple(attention.shape)}"
        )
    if gradients.shape != attention.shape:
        raise ValueError(
            f"Gradient shape {tuple(gradients.shape)} != attention "
            f"shape {tuple(attention.shape)}"
        )
    if attention.shape[0] != 1:
        raise ValueError("Extraction currently requires batch size 1")

    # CLS query token = index 0. Patch key tokens = indices 1:.
    cls_attention = attention[0, :, 0, 1:].detach().float()
    cls_gradient = gradients[0, :, 0, 1:].detach().float()

    num_patches = int(cls_attention.shape[-1])
    grid = math.isqrt(num_patches)
    if grid * grid != num_patches:
        raise ValueError(
            f"Cannot reshape {num_patches} visual patch tokens to a square grid"
        )

    per_head_signed = cls_attention * cls_gradient

    signed = per_head_signed.mean(dim=0).reshape(grid, grid)

    # Match the ALBEF-style implementation: ReLU per head, then average heads.
    positive = torch.relu(per_head_signed).mean(dim=0).reshape(grid, grid)
    normalized = normalize_positive_map(positive)

    return {
        "cam_signed_raw": signed.cpu(),
        "cam_positive_raw": positive.cpu(),
        "cam_vis": normalized.cpu(),
    }


def upsample_cam(cam: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(
        cam[None, None].float(),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )[0, 0].clamp(0.0, 1.0).cpu()


# -----------------------------------------------------------------------------
# Data selection
# -----------------------------------------------------------------------------

def find_id_column(frame: pd.DataFrame, explicit: Optional[str]) -> str:
    if explicit:
        if explicit not in frame.columns:
            raise KeyError(
                f"--id_column {explicit!r} not found. Columns: {list(frame.columns)}"
            )
        return explicit

    for candidate in ("image_id", "imageId", "image_name", "id"):
        if candidate in frame.columns:
            return candidate
    return str(frame.columns[0])


def load_selection(
    labels_csv: Path,
    target_labels: Sequence[str],
    id_column: Optional[str],
    max_images: Optional[int],
    positive_only_label: Optional[str],
    selection_csv: Optional[Path],
) -> Tuple[pd.DataFrame, str]:
    labels_frame = pd.read_csv(labels_csv)
    id_col = find_id_column(labels_frame, id_column)
    labels_frame[id_col] = labels_frame[id_col].astype(str)

    missing = [label for label in target_labels if label not in labels_frame.columns]
    if missing:
        raise KeyError(f"Target labels absent from labels CSV: {missing}")

    if selection_csv is not None:
        selected = pd.read_csv(selection_csv)
        selected_id_col = find_id_column(selected, id_column if id_column in selected.columns else None)
        selected_ids = selected[selected_id_col].astype(str).tolist()
        wanted = set(selected_ids)
        frame = labels_frame[labels_frame[id_col].isin(wanted)].copy()

        # Preserve selection_csv order exactly.
        rank = {image_id: index for index, image_id in enumerate(selected_ids)}
        frame["__selection_order__"] = frame[id_col].map(rank)
        frame = frame.sort_values("__selection_order__").drop(
            columns="__selection_order__"
        )

        found = set(frame[id_col].astype(str))
        absent = [image_id for image_id in selected_ids if image_id not in found]
        if absent:
            raise ValueError(
                f"{len(absent)} selected image IDs are absent from labels CSV. "
                f"First examples: {absent[:10]}"
            )
    else:
        frame = labels_frame.copy()

    if positive_only_label is not None:
        if positive_only_label not in frame.columns:
            raise KeyError(
                f"--positive_only_label {positive_only_label!r} absent from labels CSV"
            )
        values = pd.to_numeric(frame[positive_only_label], errors="coerce").fillna(0)
        frame = frame[values == 1].copy()

    if max_images is not None:
        frame = frame.iloc[: int(max_images)].copy()

    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError("No images remain after selection/filtering")

    return frame, id_col


# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------

def extract_one_checkpoint(
    args: argparse.Namespace,
    checkpoint_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    labels_csv = Path(args.labels_csv)
    images_root = Path(args.images_root)
    selection_csv = Path(args.selection_csv) if args.selection_csv else None

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"Images root not found: {images_root}")
    if selection_csv is not None and not selection_csv.is_file():
        raise FileNotFoundError(f"Selection CSV not found: {selection_csv}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    device = torch.device(args.device)
    target_labels = parse_labels(args.target_labels)
    prompt_pairs = load_prompt_pairs(
        target_labels,
        args.positive_template,
        args.negative_template,
        args.prompts_json,
    )

    model = build_chexzero_model(
        checkpoint_path=checkpoint_path,
        device=device,
        context_length=args.context_length,
        pretrained=bool(args.pretrained),
    )

    # ------------------------------------------------------------------
    # Select the visual Transformer layer to explain.
    # ------------------------------------------------------------------
    visual_transformer = getattr(model.visual, "transformer", None)
    if visual_transformer is None:
        raise AttributeError("model.visual.transformer is unavailable")
    blocks = getattr(visual_transformer, "resblocks", None)
    if blocks is None:
        raise AttributeError("model.visual.transformer.resblocks is unavailable")

    num_layers = len(blocks)
    layer_index = int(args.attention_layer)
    if layer_index < 0:
        layer_index = num_layers + layer_index
    if not (0 <= layer_index < num_layers):
        raise ValueError(
            f"--attention_layer resolves to {layer_index}, but model has "
            f"{num_layers} visual blocks"
        )

    instrument = VisualAttentionInstrument(blocks[layer_index])

    image_resolution = int(getattr(model.visual, "input_resolution", 320))
    patch_size = int(model.visual.conv1.kernel_size[0])
    expected_grid = image_resolution // patch_size
    expected_patches = expected_grid * expected_grid

    positional_tokens = int(model.visual.positional_embedding.shape[0])
    if positional_tokens - 1 != expected_patches:
        raise ValueError(
            "Visual positional embedding does not match image_resolution/patch_size: "
            f"tokens={positional_tokens}, expected patches={expected_patches}"
        )

    frame, id_col = load_selection(
        labels_csv=labels_csv,
        target_labels=target_labels,
        id_column=args.id_column,
        max_images=args.max_images,
        positive_only_label=args.positive_only_label,
        selection_csv=selection_csv,
    )

    positive_text, negative_text = encode_prompt_pairs(
        model=model,
        prompts=prompt_pairs,
        labels=target_labels,
        device=device,
        context_length=args.context_length,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = output_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    print("[Method] CheXzero pathology-conditioned visual attention Grad-CAM")
    print("[Target] positive cosine similarity - negative cosine similarity")
    print(
        f"[ViT] layer={layer_index} (1-based {layer_index + 1}/{num_layers}), "
        f"input={image_resolution}, patch={patch_size}, grid={expected_grid}x{expected_grid}"
    )
    print(f"[Data] images={len(frame)} labels={target_labels}")
    for label in target_labels:
        pair = prompt_pairs[label]
        print(
            f"[Prompt] {label!r}: positive={pair['positive']!r}, "
            f"negative={pair['negative']!r}"
        )

    records: List[Dict[str, Any]] = []

    try:
        for _, row in tqdm(
            frame.iterrows(),
            total=len(frame),
            desc="CheXzero attention Grad-CAM",
        ):
            image_id = str(row[id_col])
            output_path = maps_dir / f"{image_id}.pt"

            if output_path.exists() and not args.overwrite:
                record: Dict[str, Any] = {
                    "image_id": image_id,
                    "heatmap_path": str(output_path),
                    "status": "exists_skipped",
                }
                for label in target_labels:
                    record[f"y::{label}"] = float(row[label])
                records.append(record)
                continue

            image_path = resolve_image_path(images_root, image_id)
            image_tensor, original_size_wh = load_chexzero_image(
                image_path=image_path,
                image_resolution=image_resolution,
                pretrained=bool(args.pretrained),
                resize_nonpretrained=bool(args.resize_nonpretrained),
            )
            image_tensor = image_tensor.unsqueeze(0).to(device)

            output: Dict[str, Any] = {
                "__metadata__": {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "original_size_wh": original_size_wh,
                    "checkpoint": str(checkpoint_path),
                    "model": "CheXzero",
                    "method": "chexzero_itc_margin_visual_self_attention_gradcam",
                    "target": "positive_similarity-negative_similarity",
                    "uses_positive_negative_prompts": True,
                    "uses_raw_attention_only": False,
                    "uses_cross_attention": False,
                    "attention_layer_index": layer_index,
                    "attention_layer_number_1based": layer_index + 1,
                    "num_visual_layers": num_layers,
                    "image_resolution": image_resolution,
                    "patch_size": patch_size,
                    "patch_grid": [expected_grid, expected_grid],
                    "output_map_size": int(args.output_map_size),
                    "preprocessing": {
                        "matches_zero_shot_vindr": True,
                        "rgb_input": True,
                        "input_intensity_scale": "0..255",
                        "mean": list(CHEXZERO_CXR_MEAN),
                        "std": list(CHEXZERO_CXR_STD),
                        "pretrained": bool(args.pretrained),
                        "resize_nonpretrained": bool(args.resize_nonpretrained),
                        "model_input_resolution": image_resolution,
                    },
                }
            }

            record = {
                "image_id": image_id,
                "image_path": str(image_path),
                "heatmap_path": str(output_path),
                "status": "saved",
            }

            # Re-run the visual encoder separately for every label because each
            # backward pass consumes the graph and because we want one clean
            # pathology-conditioned gradient per map.
            for label_index, label in enumerate(target_labels):
                instrument.reset()
                model.zero_grad(set_to_none=True)

                image_features = model.encode_image(image_tensor)
                image_features = F.normalize(image_features.float(), dim=-1)

                positive_similarity = (
                    image_features * positive_text[label_index].unsqueeze(0)
                ).sum()
                negative_similarity = (
                    image_features * negative_text[label_index].unsqueeze(0)
                ).sum()

                margin = positive_similarity - negative_similarity
                positive_probability = torch.sigmoid(margin)

                margin.backward()

                attention = instrument.attention_probs
                if attention is None:
                    raise RuntimeError(
                        "Instrumented visual attention tensor was not captured"
                    )
                gradients = attention.grad
                if gradients is None:
                    raise RuntimeError(
                        "No gradient reached the captured attention probabilities. "
                        "This indicates that the instrumented attention tensor was "
                        "not on the score's computational path."
                    )

                cams = compute_attention_gradcam(attention, gradients)
                cams["cam_vis_up"] = upsample_cam(
                    cams["cam_vis"], int(args.output_map_size)
                )

                pair = prompt_pairs[label]
                output[label] = {
                    "ground_truth": float(row[label]),
                    "positive_prompt": pair["positive"],
                    "negative_prompt": pair["negative"],
                    "positive_similarity": float(
                        positive_similarity.detach().cpu()
                    ),
                    "negative_similarity": float(
                        negative_similarity.detach().cpu()
                    ),
                    "margin": float(margin.detach().cpu()),
                    "positive_probability": float(
                        positive_probability.detach().cpu()
                    ),
                    **cams,
                }

                record[f"y::{label}"] = float(row[label])
                record[f"positive_similarity::{label}"] = float(
                    positive_similarity.detach().cpu()
                )
                record[f"negative_similarity::{label}"] = float(
                    negative_similarity.detach().cpu()
                )
                record[f"margin::{label}"] = float(margin.detach().cpu())
                record[f"score::{label}"] = float(
                    positive_probability.detach().cpu()
                )

            torch.save(output, output_path)
            records.append(record)

    finally:
        instrument.close()

    index_path = output_dir / "chexzero_attention_gradcam_index.csv"
    pd.DataFrame(records).to_csv(index_path, index=False)

    manifest = {
        "model": "CheXzero",
        "method": "chexzero_itc_margin_visual_self_attention_gradcam",
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.stem,
        "pretrained": bool(args.pretrained),
        "target": "positive_similarity-negative_similarity",
        "classification_probability": "sigmoid(positive_similarity-negative_similarity)",
        "target_labels": target_labels,
        "prompts": prompt_pairs,
        "attention_layer_index": layer_index,
        "attention_layer_number_1based": layer_index + 1,
        "num_visual_layers": num_layers,
        "image_resolution": image_resolution,
        "patch_size": patch_size,
        "patch_grid": [expected_grid, expected_grid],
        "output_map_size": int(args.output_map_size),
        "num_images": len(frame),
        "labels_csv": str(labels_csv),
        "images_root": str(images_root),
        "selection_csv": None if selection_csv is None else str(selection_csv),
        "index_file": str(index_path),
        "notes": [
            "Raw visual attention alone is not used as the localization map.",
            "The saved CAM is conditioned on the positive-vs-negative CheXzero text margin.",
            "The selected visual block is instrumented so the saved attention probabilities are exactly those used in attention @ value.",
            "ReLU is applied per attention head to A * d(target)/dA, then heads are averaged.",
        ],
    }
    manifest_path = output_dir / "chexzero_attention_gradcam_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"[Output] maps: {maps_dir}")
    print(f"[Output] index: {index_path}")
    print(f"[Output] manifest: {manifest_path}")
    print(f"[Done] {checkpoint_path.name}")

    # Free the checkpoint before the next one is loaded.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.stem,
        "output_dir": str(output_dir),
        "index_file": str(index_path),
        "manifest_file": str(manifest_path),
        "num_images": int(len(frame)),
        "target_labels": list(target_labels),
    }


def resolve_checkpoint_paths(
    model_paths_or_dirs: Sequence[str],
    exts: Tuple[str, ...] = (".pt", ".pth", ".bin"),
    recursive: bool = False,
) -> List[Path]:
    """Expand checkpoint files/directories exactly like the older evaluator."""
    found: List[Path] = []
    for item in model_paths_or_dirs:
        path = Path(item)
        if path.is_dir():
            globber = path.rglob if recursive else path.glob
            for ext in exts:
                found.extend(globber(f"*{ext}"))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"Checkpoint path not found: {path}")

    unique = [Path(value) for value in sorted({str(p.resolve()) for p in found})]
    if not unique:
        raise FileNotFoundError(
            f"No checkpoints found from {list(model_paths_or_dirs)} with extensions {exts}"
        )

    stems = [p.stem for p in unique]
    duplicates = sorted({name for name in stems if stems.count(name) > 1})
    if duplicates:
        raise ValueError(
            "Checkpoint filename stems must be unique because each checkpoint gets "
            f"its own output directory. Duplicate stems: {duplicates}"
        )
    return unique


def extract_all(args: argparse.Namespace) -> None:
    checkpoints = resolve_checkpoint_paths(
        args.model_paths,
        exts=tuple(args.ckpt_exts),
        recursive=bool(args.ckpt_recursive),
    )
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    print(f"[Run] Found {len(checkpoints)} checkpoint(s)")
    for i, path in enumerate(checkpoints, start=1):
        print(f"  [{i:02d}] {path}")

    summaries: List[Dict[str, Any]] = []
    for index, checkpoint_path in enumerate(checkpoints, start=1):
        print("\n" + "=" * 88)
        print(f"[Checkpoint {index}/{len(checkpoints)}] {checkpoint_path}")
        print("=" * 88)
        checkpoint_output = root / checkpoint_path.stem
        summaries.append(
            extract_one_checkpoint(
                args=args,
                checkpoint_path=checkpoint_path,
                output_dir=checkpoint_output,
            )
        )

    summary_csv = root / "chexzero_multi_checkpoint_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)

    combined_manifest = {
        "model": "CheXzero",
        "num_checkpoints": len(checkpoints),
        "checkpoints": summaries,
        "labels_csv": str(Path(args.labels_csv)),
        "images_root": str(Path(args.images_root)),
        "target_labels": parse_labels(args.target_labels),
        "positive_template": args.positive_template,
        "negative_template": args.negative_template,
        "attention_layer": int(args.attention_layer),
        "output_map_size": int(args.output_map_size),
        "pretrained": bool(args.pretrained),
        "summary_csv": str(summary_csv),
    }
    combined_manifest_path = root / "chexzero_multi_checkpoint_manifest.json"
    with combined_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(combined_manifest, handle, indent=2)

    print("\n[Done] All checkpoints processed")
    print(f"[Output] summary: {summary_csv}")
    print(f"[Output] manifest: {combined_manifest_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Extract pathology-conditioned CheXzero visual self-attention "
            "Grad-CAM maps using a positive-minus-negative zero-shot margin."
        ),
    )

    parser.add_argument(
        "--model_paths",
        nargs="+",
        required=True,
        help=(
            "Checkpoint files and/or directories. Directories are expanded to "
            ".pt/.pth/.bin checkpoint files, matching zero_shot_vindr.py."
        ),
    )
    parser.add_argument(
        "--ckpt_exts",
        nargs="+",
        default=[".pt", ".pth", ".bin"],
        help="Checkpoint extensions to discover inside supplied directories.",
    )
    parser.add_argument(
        "--ckpt_recursive",
        action="store_true",
        help="Search supplied checkpoint directories recursively.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help=(
            "Match zero_shot_vindr.py --pretrained: build OpenAI CLIP ViT-B/32 "
            "before loading each CheXzero checkpoint."
        ),
    )
    parser.add_argument(
        "--resize_nonpretrained",
        action="store_true",
        help=(
            "Normally disabled to match the old evaluator exactly. If enabled, "
            "non-pretrained inputs that are not already at the model resolution "
            "are resized with bicubic interpolation."
        ),
    )
    parser.add_argument(
        "--labels_csv",
        required=True,
        help="CSV containing image IDs and pathology labels.",
    )
    parser.add_argument(
        "--images_root",
        required=True,
        help="Directory containing <image_id>.png/.jpg/.jpeg images.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory in which maps/index/manifest are saved.",
    )

    parser.add_argument(
        "--target_labels",
        nargs="+",
        default=["Cardiomegaly", "Pleural effusion"],
        help='Labels to explain, e.g. Cardiomegaly "Pleural effusion".',
    )
    parser.add_argument(
        "--id_column",
        default=None,
        help="Image-ID column. If omitted, common names or the first column are used.",
    )

    parser.add_argument(
        "--positive_template",
        default="{}",
        help="Positive zero-shot prompt template.",
    )
    parser.add_argument(
        "--negative_template",
        default="no {}",
        help="Negative zero-shot prompt template.",
    )
    parser.add_argument(
        "--prompts_json",
        default=None,
        help=(
            "Optional JSON path or inline JSON object overriding prompts per label. "
            "Each label maps to {'positive': ..., 'negative': ...}."
        ),
    )

    parser.add_argument(
        "--context_length",
        type=int,
        default=77,
        help="CheXzero/OpenAI CLIP text context length.",
    )
    parser.add_argument(
        "--attention_layer",
        type=int,
        default=-1,
        help=(
            "Visual Transformer block index. Negative indices follow Python "
            "convention; -1 means the final visual block."
        ),
    )
    parser.add_argument(
        "--output_map_size",
        type=int,
        default=320,
        help=(
            "Spatial size of saved cam_vis_up. The native low-resolution map "
            "is always saved separately. Use 256 here only if your downstream "
            "visualization/evaluation pipeline explicitly requires 256x256."
        ),
    )

    parser.add_argument(
        "--selection_csv",
        default=None,
        help=(
            "Optional CSV defining image IDs to extract, preserving its order. "
            "Useful for extracting the exact same ALBEF qualitative cases."
        ),
    )
    parser.add_argument(
        "--positive_only_label",
        default=None,
        help=(
            "Optional qualitative/debug filter. For unbiased full-dataset/FROC "
            "extraction leave this unset."
        ),
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional deterministic truncation after selection/filtering.",
    )

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    extract_all(parse_args())
