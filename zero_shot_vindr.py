"""
Zero-shot classification (CheXzero) on VinDr PNGs with optional ensembling
and an optional smoke test on a small subset.
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils import data
from tqdm import tqdm

from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

import clip
from model import CLIP
from eval import sigmoid


# -----------------------------
# Dataset: VinDr PNGs ordered by CSV
# -----------------------------
class VinDrPNGDataset(data.Dataset):
    """
    Loads images strictly in the order of image_ids provided (from CSV),
    ensuring alignment with y_true rows.

    Returns dict:
      {'img': tensor[C,H,W], 'image_id': str}
    """

    def __init__(self, image_dir: str, image_ids: List[str], transform=None):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.image_ids = list(image_ids)
        self.transform = transform

        missing = []
        for iid in self.image_ids:
            p = self.image_dir / f"{iid}.png"
            if not p.exists():
                missing.append(str(p))
        if missing:
            raise FileNotFoundError(
                f"[VinDrPNGDataset] Missing {len(missing)} PNG files. Examples:\n"
                + "\n".join(missing[:10])
            )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_id = self.image_ids[idx]
        img_path = self.image_dir / f"{image_id}.png"

        img = Image.open(img_path).convert("RGB")  # (H,W,3) uint8
        arr = np.array(img)
        arr = np.transpose(arr, (2, 0, 1))        # (3,H,W)
        x = torch.from_numpy(arr).float()         # float32, 0..255

        if self.transform:
            x = self.transform(x)

        return {"img": x, "image_id": image_id}


# -----------------------------
# Model loading (matches repo)
# -----------------------------
def load_clip(model_path: str, pretrained: bool = False, context_length: int = 77):
    device = torch.device("cpu")

    if pretrained is False:
        params = {
            "embed_dim": 768,
            "image_resolution": 320,
            "vision_layers": 12,
            "vision_width": 768,
            "vision_patch_size": 16,
            "context_length": context_length,
            "vocab_size": 49408,
            "transformer_width": 512,
            "transformer_heads": 8,
            "transformer_layers": 12,
        }
        model = CLIP(**params)
    else:
        model, _ = clip.load("ViT-B/32", device=device, jit=False)

    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def make_transform(pretrained: bool):
    """
    Matches repo `make()` preprocessing:
      Normalize((101.48761,...), (83.43944,...))
      + Resize(224, BICUBIC) if pretrained
    Assumes input tensor is float32 in [0..255].
    """
    transforms = [
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
    ]
    if pretrained:
        transforms.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
    return Compose(transforms)


def zeroshot_classifier(classnames, templates, model, device: torch.device, context_length: int = 77):
    model.eval()
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, desc="Text embeddings"):
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts, context_length=context_length).to(device)
            class_embeddings = model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


def predict(loader, model, zeroshot_weights, device: torch.device, softmax_eval: bool = True):
    model.eval()
    y_pred = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Image inference"):
            images = batch["img"].to(device)  # (B,3,H,W)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ zeroshot_weights  # (B, L)
            logits_np = logits.detach().cpu().numpy()

            if softmax_eval is False:
                mu = logits_np.mean(axis=1, keepdims=True)
                sd = logits_np.std(axis=1, keepdims=True) + 1e-8
                logits_np = sigmoid((logits_np - mu) / sd)

            y_pred.append(logits_np)

    return np.concatenate(y_pred, axis=0)


def run_single_prediction(
    cxr_labels: List[str],
    template: str,
    model,
    loader,
    device: torch.device,
    softmax_eval: bool = True,
    context_length: int = 77,
):
    zeroshot_weights = zeroshot_classifier(cxr_labels, [template], model, device=device, context_length=context_length)
    return predict(loader, model, zeroshot_weights, device=device, softmax_eval=softmax_eval)


def run_softmax_eval(
    model,
    loader,
    eval_labels: List[str],
    pair_template: Tuple[str, str],
    device: torch.device,
    context_length: int = 77,
):
    pos, neg = pair_template
    pos_pred = run_single_prediction(eval_labels, pos, model, loader, device=device, softmax_eval=True, context_length=context_length)
    neg_pred = run_single_prediction(eval_labels, neg, model, loader, device=device, softmax_eval=True, context_length=context_length)

    # p = exp(pos) / (exp(pos)+exp(neg)) == 1 / (1 + exp(neg-pos))
    return 1.0 / (1.0 + np.exp(neg_pred - pos_pred))


def load_vindr_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "image_id" not in df.columns:
        raise ValueError(f"[VinDr] CSV must contain 'image_id' column. Columns: {list(df.columns)}")
    return df


def build_loader(image_dir: str, image_ids: List[str], pretrained: bool, batch_size: int, num_workers: int, device: torch.device):
    transform = make_transform(pretrained=pretrained)
    dset = VinDrPNGDataset(image_dir=image_dir, image_ids=image_ids, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    return dset, loader


def infer_one_checkpoint(
    model_path: str,
    loader,
    labels: List[str],
    pair_template: Tuple[str, str],
    pretrained: bool,
    context_length: int,
    device: torch.device,
):
    model = load_clip(model_path=model_path, pretrained=pretrained, context_length=context_length)
    model = model.to(device)
    model.eval()
    return run_softmax_eval(model, loader, labels, pair_template, device=device, context_length=context_length)


def ensemble_models_vindr_png(
    model_paths: List[str],
    image_dir: str,
    image_ids: List[str],
    cxr_labels: List[str],
    cxr_pair_template: Tuple[str, str],
    cache_dir: Optional[str] = None,
    save_name: Optional[str] = None,
    pretrained: bool = True,
    context_length: int = 77,
    batch_size: int = 64,
    num_workers: int = 4,
    device: Optional[str] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    model_paths = sorted(model_paths)
    predictions = []

    _, loader = build_loader(
        image_dir=image_dir,
        image_ids=image_ids,
        pretrained=pretrained,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device_t,
    )

    for path in model_paths:
        model_name = Path(path).stem

        cache_path = None
        if cache_dir is not None:
            Path(cache_dir).mkdir(exist_ok=True, parents=True)
            cache_path = Path(cache_dir) / (f"{save_name}_{model_name}.npy" if save_name else f"{model_name}.npy")

        if cache_path is not None and cache_path.exists():
            print(f"[Cache] Loading cached prediction for {model_name} from {cache_path}")
            y_pred = np.load(cache_path)
        else:
            print(f"[Infer] {model_name}: {path}")
            y_pred = infer_one_checkpoint(
                model_path=path,
                loader=loader,
                labels=cxr_labels,
                pair_template=cxr_pair_template,
                pretrained=pretrained,
                context_length=context_length,
                device=device_t,
            )
            if cache_path is not None:
                np.save(cache_path, y_pred)

        predictions.append(y_pred)

    y_pred_avg = np.mean(predictions, axis=0)
    return predictions, y_pred_avg


def write_outputs(out_dir: Path, y_true: np.ndarray, scores: np.ndarray, label_names: List[str], image_ids: List[str], meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "y_true.npy", y_true.astype(np.int32))
    np.save(out_dir / "scores.npy", scores.astype(np.float32))
    (out_dir / "label_names.txt").write_text("\n".join(label_names) + "\n")
    (out_dir / "image_ids.txt").write_text("\n".join(image_ids) + "\n")
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def compute_map_at_k(y_true, scores, k=10):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    N, L = y_true.shape
    ap_list = []

    for i in range(N):
        y = y_true[i]
        s = scores[i]
        pos_idx = np.where(y == 1)[0]
        if len(pos_idx) == 0:
            continue

        order = np.argsort(-s)      # descending
        topk = order[:k]

        hits = 0
        precisions = []
        for rank, idx in enumerate(topk, start=1):
            if y[idx] == 1:
                hits += 1
                precisions.append(hits / rank)

        if len(precisions) == 0:
            ap = 0.0
        else:
            denom = min(len(pos_idx), k)
            ap = float(np.sum(precisions) / denom)
        ap_list.append(ap)

    if len(ap_list) == 0:
        return None
    return float(np.mean(ap_list))


def compute_classification_metrics(y_true, scores, label_names, threshold=0.5):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    N, L = y_true.shape
    assert scores.shape == (N, L)

    metrics = {}

    # ----- ROC–AUC -----
    per_label_auc = {}
    auc_values = []
    for j, label in enumerate(label_names):
        y = y_true[:, j]
        if len(np.unique(y)) < 2:
            per_label_auc[label] = None
            continue
        try:
            auc = roc_auc_score(y, scores[:, j])
            per_label_auc[label] = float(auc)
            auc_values.append(auc)
        except ValueError:
            per_label_auc[label] = None

    metrics["per_label_auc"] = per_label_auc
    metrics["macro_auc"] = float(np.mean(auc_values)) if len(auc_values) > 0 else None

    # micro-AUC
    try:
        metrics["micro_auc"] = float(roc_auc_score(y_true.ravel(), scores.ravel()))
    except ValueError:
        metrics["micro_auc"] = None

    # ----- F1 (global threshold) -----
    y_pred = (scores >= threshold).astype(int)

    per_label_f1 = {}
    f1_values = []
    for j, label in enumerate(label_names):
        y = y_true[:, j]
        y_hat = y_pred[:, j]
        if len(np.unique(y)) < 2:
            per_label_f1[label] = None
            continue
        f1 = f1_score(y, y_hat)
        per_label_f1[label] = float(f1)
        f1_values.append(f1)

    metrics["per_label_f1"] = per_label_f1
    metrics["macro_f1"] = float(np.mean(f1_values)) if len(f1_values) > 0 else None
    metrics["micro_f1"] = float(f1_score(y_true.ravel(), y_pred.ravel()))

    # ----- mAP@10 -----
    metrics["map_at_10"] = compute_map_at_k(y_true, scores, k=10)

    return metrics


# ==========================
# Saving helpers
# ==========================

def save_scores_npz(out_path: Path,
                    image_ids: list,
                    label_names: list,
                    scores: np.ndarray,
                    y_true: np.ndarray = None):
    """
    Compact, fast load for downstream:
      - image_ids (N,)
      - label_names (L,)
      - scores (N,L)
      - y_true (optional) (N,L)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_ids": np.asarray(image_ids, dtype=object),
        "label_names": np.asarray(label_names, dtype=object),
        "scores": np.asarray(scores, dtype=np.float32),
    }
    if y_true is not None:
        payload["y_true"] = np.asarray(y_true, dtype=np.float32)
    np.savez_compressed(out_path, **payload)
    print(f"[Output] Saved per-image scores (npz): {out_path}")


def save_scores_csv(out_path: Path,
                    image_ids: list,
                    label_names: list,
                    scores: np.ndarray,
                    y_true: np.ndarray = None):
    """
    Human-readable wide CSV:
      columns: image_id, score::<label_1>, ..., score::<label_L>
      optionally also y::<label> columns.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(scores, columns=[f"score::{lb}" for lb in label_names])
    df.insert(0, "image_id", image_ids)

    if y_true is not None:
        ydf = pd.DataFrame(y_true, columns=[f"y::{lb}" for lb in label_names])
        df = pd.concat([df, ydf], axis=1)

    df.to_csv(out_path, index=False)
    print(f"[Output] Saved per-image scores (csv): {out_path}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--vindr_root", type=str, required=True, help="Path to vindr_cxr/")
    ap.add_argument("--csv_rel", type=str, default="annotations/image_labels_test.csv")
    ap.add_argument("--img_rel", type=str, default="test")

    ap.add_argument("--model_paths", type=str, nargs="+", required=True)
    ap.add_argument("--pretrained", action="store_true", help="Use CLIP ViT-B/32 base + load CheXzero weights.")
    ap.add_argument("--context_length", type=int, default=77)

    ap.add_argument("--pos_template", type=str, default="{}", help="Positive template for softmax eval.")
    ap.add_argument("--neg_template", type=str, default="no {}", help="Negative template for softmax eval.")

    ap.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit label list (must match CSV columns). If omitted, uses all columns except image_id.",
    )

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")

    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--save_name", type=str, default="vindr")
    ap.add_argument("--out_dir", type=str, default="vindr_outputs")

    # Smoke test
    ap.add_argument("--smoke_test", action="store_true", help="Run a small-subset single-checkpoint test and exit.")
    ap.add_argument("--smoke_n", type=int, default=16, help="Number of images for smoke test.")
    ap.add_argument("--smoke_checkpoint", type=str, default=None, help="Checkpoint path to use for smoke test.")

    args = ap.parse_args()

    vindr_root = Path(args.vindr_root)
    csv_path = vindr_root / args.csv_rel
    img_dir = vindr_root / args.img_rel

    df = load_vindr_csv(str(csv_path))

    if args.labels is None or len(args.labels) == 0:
        label_names = [c for c in df.columns if c != "image_id"]
    else:
        label_names = list(args.labels)
        missing_cols = [c for c in label_names if c not in df.columns]
        if missing_cols:
            raise ValueError(f"[VinDr] Provided labels not found in CSV: {missing_cols}")

    image_ids_all = df["image_id"].astype(str).tolist()
    y_true_all = df[label_names].to_numpy(dtype=np.int32)

    pair_template = (args.pos_template, args.neg_template)

    # device
    if args.device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device_t = torch.device(device_str)

    out_dir = Path(args.out_dir)

    # -------------------------
    # Smoke test path
    # -------------------------
    if args.smoke_test:
        smoke_n = max(1, min(args.smoke_n, len(image_ids_all)))
        image_ids = image_ids_all[:smoke_n]
        y_true = y_true_all[:smoke_n]

        ckpt = args.smoke_checkpoint if args.smoke_checkpoint else sorted(args.model_paths)[0]

        print(f"[Smoke] Using checkpoint: {ckpt}")
        print(f"[Smoke] Images: {smoke_n} | Labels: {len(label_names)} | Device: {device_str}")

        _, loader = build_loader(
            image_dir=str(img_dir),
            image_ids=image_ids,
            pretrained=args.pretrained,
            batch_size=min(args.batch_size, smoke_n),
            num_workers=args.num_workers,
            device=device_t,
        )

        scores = infer_one_checkpoint(
            model_path=ckpt,
            loader=loader,
            labels=label_names,
            pair_template=pair_template,
            pretrained=args.pretrained,
            context_length=args.context_length,
            device=device_t,
        )

        smoke_out = out_dir / "smoke_test"
        smoke_out.mkdir(parents=True, exist_ok=True)
        np.save(smoke_out / "smoke_y_true.npy", y_true.astype(np.int32))
        np.save(smoke_out / "smoke_scores.npy", scores.astype(np.float32))
        (smoke_out / "smoke_label_names.txt").write_text("\n".join(label_names) + "\n")
        (smoke_out / "smoke_image_ids.txt").write_text("\n".join(image_ids) + "\n")

        meta = {
            "mode": "smoke_test",
            "vindr_root": str(vindr_root),
            "csv": str(csv_path),
            "image_dir": str(img_dir),
            "num_images": int(len(image_ids)),
            "num_labels": int(len(label_names)),
            "checkpoint": str(ckpt),
            "pretrained": bool(args.pretrained),
            "context_length": int(args.context_length),
            "pair_template": [args.pos_template, args.neg_template],
            "device": device_str,
            "batch_size": int(min(args.batch_size, smoke_n)),
            "num_workers": int(args.num_workers),
        }
        (smoke_out / "smoke_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

        print(f"[Smoke Done] Wrote smoke outputs to: {smoke_out}")
        print(f"            scores shape={scores.shape} | y_true shape={y_true.shape}")
        return

    # -------------------------
    # Full ensemble path
    # -------------------------
    print(f"[Run] Full inference on {len(image_ids_all)} images, {len(label_names)} labels, device={device_str}")
    preds_list, scores_avg = ensemble_models_vindr_png(
        model_paths=args.model_paths,
        image_dir=str(img_dir),
        image_ids=image_ids_all,
        cxr_labels=label_names,
        cxr_pair_template=pair_template,
        cache_dir=args.cache_dir,
        save_name=args.save_name,
        pretrained=args.pretrained,
        context_length=args.context_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device_str,
    )

    meta = {
        "mode": "full",
        "vindr_root": str(vindr_root),
        "csv": str(csv_path),
        "image_dir": str(img_dir),
        "num_images": int(len(image_ids_all)),
        "num_labels": int(len(label_names)),
        "pretrained": bool(args.pretrained),
        "context_length": int(args.context_length),
        "pair_template": [args.pos_template, args.neg_template],
        "model_paths": list(map(str, sorted(args.model_paths))),
        "cache_dir": args.cache_dir,
        "save_name": args.save_name,
        "device": device_str,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
    }

    write_outputs(
        out_dir=out_dir,
        y_true=y_true_all,
        scores=scores_avg,
        label_names=label_names,
        image_ids=image_ids_all,
        meta=meta,
    )

    print(f"[Done] Wrote outputs to: {out_dir}")
    print(f"       y_true.npy shape={y_true_all.shape}")
    print(f"       scores.npy shape={scores_avg.shape}")

    cls_metrics = compute_classification_metrics(
        y_true=y_true_all,
        scores=scores_avg,
        label_names=label_names,
        threshold=0.5,
    )

    ckpt_name = "ensemble"

    results = {
        "checkpoint": ckpt_name,
        "num_images": int(scores_avg.shape[0]),
        "label_names": list(label_names),
        "classification": cls_metrics,
        "threshold": 0.5,
        "scores_file_npz": None,
        "scores_file_csv": None,
    }

    scores_npz_path = out_dir / f"vindr_zero_shot_scores_{ckpt_name}.npz"
    save_scores_npz(
        out_path=scores_npz_path,
        image_ids=image_ids_all,
        label_names=label_names,
        scores=scores_avg,
        y_true=y_true_all,
    )
    results["scores_file_npz"] = str(scores_npz_path)

    scores_csv_path = out_dir / f"vindr_zero_shot_scores_{ckpt_name}.csv"
    save_scores_csv(
        out_path=scores_csv_path,
        image_ids=image_ids_all,
        label_names=label_names,
        scores=scores_avg,
        y_true=y_true_all,
    )
    results["scores_file_csv"] = str(scores_csv_path)

    # ---- Save JSON metrics ----
    out_path = out_dir / f"vindr_zero_shot_{ckpt_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Output] Saved metrics to: {out_path}")

if __name__ == "__main__":
    main()
