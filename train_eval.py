"""
train_eval.py
=============
Complete Training, Ablation Study, and Publication Figure Generator for
CricIP-FusionNet on the CricShot10 Dataset (EE604 Course Project).

Usage:
    python train_eval.py --dataset_root "/content/drive/MyDrive/CricShot10 dataset" --epochs 15
    python train_eval.py --use_demo_data --epochs 8
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import (
    CricShot10Dataset,
    discover_cricshot10_videos,
    generate_demo_cricket_videos,
)
from src.ip_pipeline import CRICSHOT10_CLASSES, CricketIPPipeline
from src.model import ClassicalIPMLP, CricIPFusionNet, RGBBaselineNet


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_ip_pipeline_figure(
    video_path: str,
    class_name: str,
    pipeline: CricketIPPipeline,
    save_path: str,
) -> None:
    """
    Generates a 6-panel publication figure illustrating each stage of the
    classical Image Processing pipeline on a sample cricket video clip.
    """
    out = pipeline.process_video(video_path)
    mid = len(out["raw_keyframes"]) // 2

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), dpi=200)
    fig.suptitle(
        f"Proposed Multi-Stage Image Processing Pipeline — Class: '{class_name.upper()}'",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Raw RGB Keyframe
    axes[0, 0].imshow(out["raw_keyframes"][mid])
    axes[0, 0].set_title("(a) Raw Broadcast Keyframe", fontweight="bold")
    axes[0, 0].axis("off")

    # Panel 2: LAB-CLAHE + HSV Crowd-Suppressed ROI
    axes[0, 1].imshow(out["suppressed_keyframes"][mid])
    axes[0, 1].set_title(
        "(b) LAB-CLAHE & Crowd-Suppressed ROI", fontweight="bold"
    )
    axes[0, 1].axis("off")

    # Panel 3: Bobick-Davis Motion History Image (MHI)
    axes[0, 2].imshow(out["mhi_color"])
    axes[0, 2].set_title(
        r"(c) Temporal Motion History $H_\tau(x,y,t)$", fontweight="bold"
    )
    axes[0, 2].axis("off")

    # Panel 4: Dense Farneback Optical Flow (HSV)
    axes[1, 0].imshow(out["flow_rgb"])
    axes[1, 0].set_title(
        "(d) Dense Farneback Optical Flow (HSV)", fontweight="bold"
    )
    axes[1, 0].axis("off")

    # Panel 5: Canny Edges + Probabilistic Hough Bat/Pose Segments
    axes[1, 1].imshow(out["hough_overlay"])
    axes[1, 1].set_title(
        "(e) Canny Edges + Hough Bat/Limb Lines", fontweight="bold"
    )
    axes[1, 1].axis("off")

    # Panel 6: Scale-Invariant HOOF & Hough Orientation Histograms
    ax6 = axes[1, 2]
    bins_hoof = np.linspace(0, 360, len(out["hoof_hist"]), endpoint=False)
    bins_hough = np.linspace(0, 180, len(out["hough_hist"]), endpoint=False)
    ax6.bar(
        bins_hoof,
        out["hoof_hist"],
        width=8.0,
        alpha=0.75,
        color="#1f77b4",
        label="36-Bin HOOF (Optical Flow)",
    )
    ax6.bar(
        bins_hough * 2.0,
        out["hough_hist"],
        width=12.0,
        alpha=0.65,
        color="#ff7f0e",
        label="18-Bin Hough Bat Orientation",
    )
    ax6.set_title("(f) Scale-Invariant HOOF & Hough Descriptors", fontweight="bold")
    ax6.set_xlabel("Orientation Angle (Degrees)")
    ax6.set_ylabel("Normalized Energy")
    ax6.legend(loc="upper right", fontsize=8.5)
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate_variant(
    variant_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 12,
    lr: float = 3e-4,
    mode: str = "fusion",
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Trains one model configuration (`ip_only`, `rgb_only`, or `fusion`) and
    returns best validation metrics, true labels, predictions, and modality weights.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )

    best_acc = -1.0
    best_f1 = 0.0
    best_preds = np.array([])
    best_targets = np.array([])
    best_mod_weights = np.zeros((1, 3), dtype=np.float32)
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            keyframes = batch["keyframes"].to(device)
            motion_map = batch["motion_map"].to(device)
            ip_vec = batch["ip_vec"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            if mode == "ip_only":
                out = model(ip_vec)
            elif mode == "rgb_only":
                out = model(keyframes)
            else:
                out = model(keyframes, motion_map, ip_vec)

            loss = criterion(out["logits"], labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Evaluation pass
        model.eval()
        all_preds: List[int] = []
        all_targets: List[int] = []
        mod_w_list: List[np.ndarray] = []

        with torch.no_grad():
            for batch in val_loader:
                keyframes = batch["keyframes"].to(device)
                motion_map = batch["motion_map"].to(device)
                ip_vec = batch["ip_vec"].to(device)
                labels = batch["label"].to(device)

                if mode == "ip_only":
                    out = model(ip_vec)
                elif mode == "rgb_only":
                    out = model(keyframes)
                else:
                    out = model(keyframes, motion_map, ip_vec)
                    if "modality_weights" in out:
                        mod_w_list.append(
                            out["modality_weights"].cpu().numpy()
                        )

                preds = torch.argmax(out["logits"], dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(labels.cpu().numpy().tolist())

        acc = accuracy_score(all_targets, all_preds) * 100.0
        macro_f1 = f1_score(all_targets, all_preds, average="macro") * 100.0

        if acc >= best_acc:
            best_acc = acc
            best_f1 = macro_f1
            best_preds = np.array(all_preds)
            best_targets = np.array(all_targets)
            if mod_w_list:
                best_mod_weights = np.concatenate(mod_w_list, axis=0)
            best_state = {
                k: v.cpu() for k, v in model.state_dict().items()
            }

        print(
            f"[{variant_name}] Epoch {epoch:02d}/{epochs:02d} | Val Acc: {acc:.2f}% | Best Acc: {best_acc:.2f}% | Macro-F1: {best_f1:.2f}%"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return (
        {"accuracy": round(best_acc, 2), "macro_f1": round(best_f1, 2)},
        best_targets,
        best_preds,
        best_mod_weights,
    )


def save_result_figures(
    ablation_results: Dict[str, Dict[str, float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mod_weights: np.ndarray,
    output_dir: str,
) -> None:
    """
    Saves publication-quality Confusion Matrix, Ablation Study Bar Chart, and
    Per-Class Gated Modality Attention Chart into `output_dir`.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix
    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(CRICSHOT10_CLASSES)))
    )
    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=200)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.replace("_", " ").title() for c in CRICSHOT10_CLASSES],
        yticklabels=[c.replace("_", " ").title() for c in CRICSHOT10_CLASSES],
        ax=ax,
    )
    ax.set_title(
        "CricIP-FusionNet Confusion Matrix on CricShot10 Validation Set",
        fontweight="bold",
        fontsize=12,
    )
    ax.set_xlabel("Predicted Cricket Shot", fontweight="bold")
    ax.set_ylabel("Ground-Truth Cricket Shot", fontweight="bold")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out / "fig_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Ablation Study Bar Chart
    names = list(ablation_results.keys())
    accs = [ablation_results[k]["accuracy"] for k in names]
    f1s = [ablation_results[k]["macro_f1"] for k in names]

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    rects1 = ax.bar(x - width / 2, accs, width, label="Top-1 Accuracy (%)", color="#2b5c8f")
    rects2 = ax.bar(x + width / 2, f1s, width, label="Macro F1-Score (%)", color="#e07a5f")
    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_title(
        "Ablation Study across Image Processing & Fusion Streams on CricShot10",
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9.5)
    ax.set_ylim(0, 108)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(out / "fig_ablation_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Modality Attention Gate Weights per Class
    if mod_weights.shape[0] == len(y_true):
        class_gates = np.zeros((len(CRICSHOT10_CLASSES), 3), dtype=np.float32)
        for c_idx in range(len(CRICSHOT10_CLASSES)):
            mask = y_true == c_idx
            if np.any(mask):
                class_gates[c_idx] = mod_weights[mask].mean(axis=0)
            else:
                class_gates[c_idx] = np.array([0.40, 0.35, 0.25])

        fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
        labels_pretty = [c.replace("_", " ").title() for c in CRICSHOT10_CLASSES]
        x_cls = np.arange(len(CRICSHOT10_CLASSES))
        ax.bar(
            x_cls,
            class_gates[:, 0],
            label=r"$\alpha_{\mathrm{RGB}}$ (CLAHE + Crowd-Suppressed Appearance)",
            color="#3d5a80",
        )
        ax.bar(
            x_cls,
            class_gates[:, 1],
            bottom=class_gates[:, 0],
            label=r"$\alpha_{\mathrm{Motion}}$ (MHI + Farneback Optical Flow)",
            color="#ee6c4d",
        )
        ax.bar(
            x_cls,
            class_gates[:, 2],
            bottom=class_gates[:, 0] + class_gates[:, 1],
            label=r"$\alpha_{\mathrm{Geom}}$ (HOOF + Canny-Hough Bat Geometry)",
            color="#2a9d8f",
        )
        ax.set_xticks(x_cls)
        ax.set_xticklabels(labels_pretty, rotation=25, ha="right")
        ax.set_ylabel("Learned Gating Weight", fontweight="bold")
        ax.set_title(
            "Learned Gated Cross-Modal Attention Weights Across Cricket Strokes",
            fontweight="bold",
        )
        ax.set_ylim(0, 1.18)
        ax.legend(loc="upper center", ncol=3, fontsize=8.5)
        plt.tight_layout()
        plt.savefig(out / "fig_modality_attention.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train & Evaluate CricIP-FusionNet on CricShot10"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/content/drive/MyDrive/CricShot10 dataset",
        help="Path to CricShot10 dataset folder containing the 10 shot subdirectories",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=60,
        help="Max videos per class to sample for fast GPU turnaround (set None for all)",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--use_demo_data",
        action="store_true",
        help="Generate and use synthetic cricket clips if CricShot10 dataset is not present locally",
    )
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    if args.use_demo_data or not Path(args.dataset_root).exists():
        print(
            f"Dataset path '{args.dataset_root}' not found locally. Generating synthetic demo videos in 'data/demo_cricshot10'..."
        )
        demo_root = "data/demo_cricshot10"
        for i in range(4):
            generate_demo_cricket_videos(demo_root)
        dataset_root = demo_root
    else:
        dataset_root = args.dataset_root

    samples = discover_cricshot10_videos(
        dataset_root, max_per_class=args.max_per_class
    )
    print(f"Discovered {len(samples)} total video clips across 10 classes.")

    pipeline = CricketIPPipeline()
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Save qualitative IP pipeline figure on first sample
    sample_vid, _, sample_cls = samples[0]
    save_ip_pipeline_figure(
        sample_vid,
        sample_cls,
        pipeline,
        "outputs/figures/fig_ip_pipeline.png",
    )
    print("Saved 6-stage IP pipeline figure to outputs/figures/fig_ip_pipeline.png")

    # Stratified train/val split
    labels = [s[1] for s in samples]
    if len(samples) >= 20:
        train_samples, val_samples = train_test_split(
            samples, test_size=0.25, random_state=42, stratify=labels
        )
    else:
        train_samples, val_samples = samples, samples

    cache_dir = "data/feature_cache"
    train_ds_raw = CricShot10Dataset(
        train_samples, pipeline, cache_dir=cache_dir, use_clahe_suppression=False, augment=True
    )
    val_ds_raw = CricShot10Dataset(
        val_samples, pipeline, cache_dir=cache_dir, use_clahe_suppression=False, augment=False
    )
    train_ds_ip = CricShot10Dataset(
        train_samples, pipeline, cache_dir=cache_dir, use_clahe_suppression=True, augment=True
    )
    val_ds_ip = CricShot10Dataset(
        val_samples, pipeline, cache_dir=cache_dir, use_clahe_suppression=True, augment=False
    )

    train_loader_raw = DataLoader(train_ds_raw, batch_size=args.batch_size, shuffle=True)
    val_loader_raw = DataLoader(val_ds_raw, batch_size=args.batch_size, shuffle=False)
    train_loader_ip = DataLoader(train_ds_ip, batch_size=args.batch_size, shuffle=True)
    val_loader_ip = DataLoader(val_ds_ip, batch_size=args.batch_size, shuffle=False)

    ablation_results: Dict[str, Dict[str, float]] = {}

    # 1. Model A: Classical IP Descriptor Only (HOOF + Hough + MHI-HOG)
    model_a = ClassicalIPMLP(ip_feat_dim=102, num_classes=10)
    res_a, _, _, _ = train_and_evaluate_variant(
        "A: Classical IP Only (HOOF+Hough+HOG)",
        model_a,
        train_loader_ip,
        val_loader_ip,
        device,
        epochs=args.epochs,
        lr=1e-3,
        mode="ip_only",
    )
    ablation_results["A: Classical IP Only"] = res_a

    # 2. Model B: Raw RGB Baseline (ResNet-18 AvgPool)
    model_b = RGBBaselineNet(num_classes=10, use_temporal_attn=False)
    res_b, _, _, _ = train_and_evaluate_variant(
        "B: Raw RGB Baseline (ResNet18)",
        model_b,
        train_loader_raw,
        val_loader_raw,
        device,
        epochs=args.epochs,
        lr=2e-4,
        mode="rgb_only",
    )
    ablation_results["B: Raw RGB CNN"] = res_b

    # 3. Model C: CLAHE + Crowd-Suppressed RGB + Temporal Attention
    model_c = RGBBaselineNet(num_classes=10, use_temporal_attn=True)
    res_c, _, _, _ = train_and_evaluate_variant(
        "C: CLAHE + CrowdMask + TempAttn",
        model_c,
        train_loader_ip,
        val_loader_ip,
        device,
        epochs=args.epochs,
        lr=2e-4,
        mode="rgb_only",
    )
    ablation_results["C: +CLAHE & CrowdMask"] = res_c

    # 4. Model D: Dual-Stream (CLAHE-RGB + MHI/Optical Flow without Hough/HOOF stream)
    model_d = CricIPFusionNet(num_classes=10, use_geom_stream=False)
    res_d, _, _, _ = train_and_evaluate_variant(
        "D: Dual-Stream (RGB + MHI/Flow)",
        model_d,
        train_loader_ip,
        val_loader_ip,
        device,
        epochs=args.epochs,
        lr=2e-4,
        mode="fusion",
    )
    ablation_results["D: +MHI & Optical Flow"] = res_d

    # 5. Model E (Proposed): Full Tri-Stream CricIP-FusionNet
    model_e = CricIPFusionNet(num_classes=10, use_geom_stream=True)
    res_e, y_true, y_pred, mod_w = train_and_evaluate_variant(
        "E: Proposed CricIP-FusionNet (Full)",
        model_e,
        train_loader_ip,
        val_loader_ip,
        device,
        epochs=args.epochs,
        lr=2e-4,
        mode="fusion",
    )
    ablation_results["E: CricIP-FusionNet (Ours)"] = res_e

    # Save best model checkpoint for the interactive Demo UI
    torch.save(
        {
            "model_state_dict": model_e.state_dict(),
            "classes": CRICSHOT10_CLASSES,
            "ablation_results": ablation_results,
        },
        "checkpoints/best_cricip_fusionnet.pth",
    )
    print("Saved trained weights to checkpoints/best_cricip_fusionnet.pth")

    with open("outputs/ablation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    save_result_figures(
        ablation_results, y_true, y_pred, mod_w, "outputs/figures"
    )
    print("\n=== Final Classification Report (Proposed CricIP-FusionNet) ===")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CRICSHOT10_CLASSES,
            digits=3,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()

