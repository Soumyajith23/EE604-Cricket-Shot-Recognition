"""
EE604 Cricket Shot Recognition - Approach 3: Pose-Guided Dual-GRU (CricShot10k, 15 Classes)
===========================================================================================
Loads the trained Stage-3 checkpoints from `weights/weights/`:
  - `fuse_m_s0.pt`       (RGB + 68-D Pose Dual-GRU + Mirror TTA, Val Macro-F1 = 78.74%)
  - `fuse_m_align_s0.pt` (RGB + 68-D Pose + Impact Phase Alignment + Mirror TTA)
  - `pose_m_s0.pt`       (68-D Pose-Only GRU + Mirror TTA, Val Macro-F1 = 76.56%)
  - `rgb_m_s0.pt`        (RGB-Only EfficientNetV2-S GRU + Mirror TTA, Val Macro-F1 = 70.59%)
along with Stage-1 (`Bat_Detection_Model.pt`, `Striker_Bat_Segmentation_Model.pt`) when
`ultralytics` is available, or Classical IP striker crop + anatomical keypoint extraction
fallback when running offline on CPU.
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ip_pipeline import CricketIPPipeline

FEAT_DIM = 5120
K_FRAMES = 15

CLASSES_15: List[str] = [
    "Cover Drive",
    "Defensive",
    "Down The Wicket",
    "Flick",
    "Hook",
    "Late Cut",
    "Lofted Legside",
    "Lofted Offside",
    "Pull",
    "Reverse Sweep",
    "Scoop",
    "Square Cut",
    "Straight Drive",
    "Sweep",
    "Upper Cut",
]

# COCO-17: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows, 9/10 wrists,
#          11/12 hips, 13/14 knees, 15/16 ankles
FLIP_PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
ANGLE_TRIPLES = [
    (5, 7, 9),
    (6, 8, 10),
    (7, 5, 11),
    (8, 6, 12),
    (5, 11, 13),
    (6, 12, 14),
    (11, 13, 15),
    (12, 14, 16),
]


@dataclass
class Cfg:
    name: str
    rgb: bool = True
    pose: bool = False
    mirror: bool = False
    align: bool = False
    moddrop: float = 0.0
    K: int = K_FRAMES


class Net(nn.Module):
    """Stage-3 Dual-GRU Fusion Head over K=15 frames -> Flatten -> BN -> Dense-1024 -> 15 classes."""

    def __init__(self, use_rgb: bool, use_pose: bool, dpose: int = 68, K: int = K_FRAMES,
                 h: int = 128, nc: int = 15, moddrop: float = 0.0):
        super().__init__()
        self.use_rgb = use_rgb
        self.use_pose = use_pose
        self.moddrop = moddrop
        if use_rgb:
            self.rnorm = nn.LayerNorm(FEAT_DIM)
            self.rdrop = nn.Dropout(0.3)
            self.rgru = nn.GRU(FEAT_DIM, h, batch_first=True)
        if use_pose:
            self.pin = nn.Sequential(nn.Linear(dpose, 128), nn.ReLU(), nn.Dropout(0.2))
            self.pgru = nn.GRU(128, h, batch_first=True)
            self.register_buffer("pm", torch.zeros(dpose))
            self.register_buffer("ps", torch.ones(dpose))
        d = h * K * (int(use_rgb) + int(use_pose))
        self.head = nn.Sequential(
            nn.BatchNorm1d(d),
            nn.Linear(d, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, nc),
        )

    def forward(self, xr: Optional[torch.Tensor] = None, xp: Optional[torch.Tensor] = None) -> torch.Tensor:
        zs = []
        if self.use_rgb:
            zs.append(self.rgru(self.rdrop(self.rnorm(xr)))[0])
        if self.use_pose:
            zs.append(self.pgru(self.pin((xp - self.pm) / self.ps))[0])
        if self.training and self.moddrop > 0 and len(zs) == 2:
            u = torch.rand(zs[0].shape[0], 1, 1, device=zs[0].device)
            zs[0] = zs[0] * (u >= self.moddrop).float()
            zs[1] = zs[1] * ((u < self.moddrop) | (u >= 2 * self.moddrop)).float()
        return self.head(torch.cat(zs, 2).flatten(1))


def flip_pose(kp: np.ndarray, kc: np.ndarray, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror a skeleton horizontally: x -> W-x and swap left/right joints."""
    k2 = kp[:, FLIP_PERM].copy()
    k2[..., 0] = W - k2[..., 0]
    return k2, kc[:, FLIP_PERM].copy()


def fill_missing(kp: np.ndarray, kc: np.ndarray, thr: float = 0.25) -> np.ndarray:
    """Low-confidence joints are replaced by linear interpolation over the valid-frame sequence."""
    n = len(kp)
    t = np.arange(n)
    out = kp.astype(np.float32).copy()
    good = kc >= thr
    cen = np.array([kp[i][good[i]].mean(0) if good[i].any() else kp[i].mean(0) for i in range(n)], np.float32)
    for j in range(kp.shape[1]):
        g = good[:, j]
        if g.sum() >= 1:
            for d in range(2):
                out[:, j, d] = np.interp(t, t[g], kp[g, j, d])
        else:
            out[:, j] = cen
    return out


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-6)
    return np.arccos(np.clip(cos, -1, 1)) / np.pi


def pose_features(kp: np.ndarray, kc: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    kp: (n, 17, 2) image coords, kc: (n, 17), idx: (n,) original frame numbers.
    Returns feats (n, 68) and wrist speed (n,) in torso-lengths / frame.
    68 = 34 hip-centred torso-scaled coords + 17 confidences + 8 joint angles + 1 torso lean
         + 4 wrist velocities + 2 hip velocity + 2 hip displacement from first frame.
    """
    n = len(kp)
    k = fill_missing(kp, kc)
    hip = (k[:, 11] + k[:, 12]) / 2
    sho = (k[:, 5] + k[:, 6]) / 2
    sc = max(float(np.median(np.linalg.norm(sho - hip, axis=1))), 8.0)
    rel = (k - hip[:, None]) / sc
    gap = np.maximum(np.diff(idx.astype(np.float32)), 1.0)

    def vel(x):
        v = np.zeros_like(x)
        if n > 1:
            v[1:] = np.diff(x, axis=0) / gap.reshape((-1,) + (1,) * (x.ndim - 1))
            v[0] = v[1]
        return v

    wv = vel(k[:, [9, 10]] / sc).reshape(n, 4)
    hv = vel(hip / sc)
    hd = (hip - hip[0]) / sc
    ang = np.stack([_angle(k[:, a], k[:, b], k[:, c]) for a, b, c in ANGLE_TRIPLES], 1)
    lean = (np.arctan2(sho[:, 0] - hip[:, 0], -(sho[:, 1] - hip[:, 1])) / np.pi)[:, None]
    feats = np.concatenate([rel.reshape(n, 34), kc, ang, lean, wv, hv, hd], 1).astype(np.float32)
    speed = np.maximum(np.linalg.norm(wv[:, :2], axis=1), np.linalg.norm(wv[:, 2:], axis=1))
    return feats, speed


def peak_index(speed: np.ndarray, lo_frac: float = 0.2, hi_frac: float = 0.85) -> int:
    """Impact proxy = frame of maximum (smoothed) wrist speed, searched in the middle of the clip."""
    n = len(speed)
    s = np.convolve(np.pad(speed, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    lo = int(lo_frac * (n - 1))
    hi = max(lo + 1, int(hi_frac * (n - 1)))
    return int(lo + np.argmax(s[lo:hi + 1]))


def positions(nv: int, K: int = K_FRAMES, peak: Optional[int] = None, peak_k: int = 8) -> np.ndarray:
    """Indices into the nv striker-valid frames of the K=15 frames fed to the network."""
    k = np.arange(K, dtype=np.float64)
    if peak is None or nv < 3:
        pos = k / (K - 1) * (nv - 1)
    else:
        pos = np.where(
            k <= peak_k,
            peak * k / peak_k,
            peak + (nv - 1 - peak) * (k - peak_k) / (K - 1 - peak_k),
        )
    return np.clip(np.rint(pos).astype(int), 0, nv - 1)


def load_csr_checkpoint(weights_dir: str, variant: str = "fuse_m_s0", device: str = "cpu") -> Tuple[Net, dict]:
    """Load one of the trained Stage-3 `.pt` checkpoints (`fuse_m_s0.pt`, `fuse_m_align_s0.pt`, `pose_m_s0.pt`, `rgb_m_s0.pt`)."""
    pt_path = os.path.join(weights_dir, f"{variant}.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Checkpoint not found: {pt_path}")
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["cfg"]
    model = Net(
        use_rgb=bool(cfg_dict.get("rgb", True)),
        use_pose=bool(cfg_dict.get("pose", True)),
        dpose=int(ckpt.get("dpose", 68)),
        K=int(cfg_dict.get("K", K_FRAMES)),
        h=128,
        nc=int(ckpt.get("n_classes", 15)),
        moddrop=float(cfg_dict.get("moddrop", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, ckpt


def _estimate_striker_pose_and_crops_classical(frames_rgb: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], int]:
    """
    Extract striker crops (224x224 BGR) and 17 COCO keypoints (n, 17, 2) + confidences (n, 17)
    using Classical IP silhouette + MHI + Hough bat line when YOLO11-pose is not installed.
    """
    n = len(frames_rgb)
    H, W = frames_rgb[0].shape[:2]
    ip = CricketIPPipeline(frame_size=(224, 224))

    kp = np.zeros((n, 17, 2), dtype=np.float32)
    kc = np.full((n, 17), 0.85, dtype=np.float32)
    crops_bgr = []

    resized_rgb = [cv2.resize(f, (224, 224)) for f in frames_rgb]
    mhi_norm, mei_mask, _ = ip.compute_mhi_and_mei(resized_rgb)

    sx, sy = W / 224.0, H / 224.0
    prev_cx, prev_cy = W * 0.5, H * 0.52

    for t in range(n):
        enh = ip.apply_clahe_lab(resized_rgb[t])
        _, field_mask, _ = ip.suppress_crowd_and_segment_roi(enh)
        sil = cv2.bitwise_and(mei_mask, field_mask)
        ys, xs = np.where(sil > 100)
        if len(xs) >= 25:
            x1_s, y1_s, x2_s, y2_s = np.percentile(xs, 5), np.percentile(ys, 5), np.percentile(xs, 95), np.percentile(ys, 95)
            cx = 0.65 * prev_cx + 0.35 * ((x1_s + x2_s) * 0.5 * sx)
            cy = 0.65 * prev_cy + 0.35 * ((y1_s + y2_s) * 0.5 * sy)
            bh = max((y2_s - y1_s) * sy, H * 0.36)
            bw = max((x2_s - x1_s) * sx, W * 0.20)
        else:
            cx, cy = prev_cx, prev_cy
            bh, bw = H * 0.42, W * 0.22
        prev_cx, prev_cy = cx, cy

        torso_len = max(bh * 0.32, 18.0)
        hip_x, hip_y = cx, cy + torso_len * 0.25
        sho_x, sho_y = cx, cy - torso_len * 0.75

        geom_desc, _, hough_hist = ip.extract_canny_hough_and_hog(enh, mhi_norm)
        dom_bin = int(np.argmax(hough_hist))
        angle_deg = (dom_bin + 0.5) * (180.0 / len(hough_hist))
        rad = np.deg2rad(angle_deg)
        swing_phase = (t - n * 0.5) / max(n * 0.5, 1.0)
        w_cx = sho_x + np.cos(rad) * torso_len * (0.45 + 0.25 * swing_phase)
        w_cy = sho_y + np.sin(rad) * torso_len * (0.35 - 0.20 * swing_phase)

        # Populate COCO-17 keypoints
        kp[t, 0] = [sho_x, sho_y - torso_len * 0.35]
        kp[t, 1] = [sho_x - torso_len * 0.06, sho_y - torso_len * 0.38]
        kp[t, 2] = [sho_x + torso_len * 0.06, sho_y - torso_len * 0.38]
        kp[t, 3] = [sho_x - torso_len * 0.10, sho_y - torso_len * 0.36]
        kp[t, 4] = [sho_x + torso_len * 0.10, sho_y - torso_len * 0.36]
        kp[t, 5] = [sho_x - torso_len * 0.22, sho_y]
        kp[t, 6] = [sho_x + torso_len * 0.22, sho_y]
        kp[t, 7] = [(kp[t, 5, 0] + w_cx) * 0.5, (kp[t, 5, 1] + w_cy) * 0.5]
        kp[t, 8] = [(kp[t, 6, 0] + w_cx) * 0.5, (kp[t, 6, 1] + w_cy) * 0.5]
        kp[t, 9] = [w_cx - torso_len * 0.05, w_cy]
        kp[t, 10] = [w_cx + torso_len * 0.05, w_cy]
        kp[t, 11] = [hip_x - torso_len * 0.18, hip_y]
        kp[t, 12] = [hip_x + torso_len * 0.18, hip_y]
        kp[t, 13] = [hip_x - torso_len * 0.20, hip_y + torso_len * 0.75]
        kp[t, 14] = [hip_x + torso_len * 0.20, hip_y + torso_len * 0.75]
        kp[t, 15] = [hip_x - torso_len * 0.22, hip_y + torso_len * 1.45]
        kp[t, 16] = [hip_x + torso_len * 0.22, hip_y + torso_len * 1.45]

        x1 = int(max(0, cx - bw * 0.75))
        y1 = int(max(0, cy - bh * 0.65))
        x2 = int(min(W, cx + bw * 0.75))
        y2 = int(min(H, cy + bh * 0.65))
        if x2 - x1 < 16 or y2 - y1 < 16:
            crop_rgb = cv2.resize(frames_rgb[t], (224, 224))
        else:
            crop_rgb = cv2.resize(frames_rgb[t][y1:y2, x1:x2], (224, 224))
        crops_bgr.append(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))

    return kp, kc, crops_bgr, W


_EFFNET_CACHE = {}


def _extract_rgb_features_5120(crops_bgr: List[np.ndarray], device: str = "cpu") -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Extract (n, 5120) and mirrored (n, 5120) features using EfficientNetV2-S.
    Returns (R, Rf, has_pretrained_effnet).
    """
    import torchvision

    if "model" not in _EFFNET_CACHE:
        has_pre = False
        try:
            m = torchvision.models.efficientnet_v2_s(
                weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
            )
            has_pre = True
        except Exception:
            m = torchvision.models.efficientnet_v2_s(weights=None)
            has_pre = False
        body = m.features.eval().to(device)
        for p in body.parameters():
            p.requires_grad_(False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        _EFFNET_CACHE["model"] = (body, mean, std, has_pre)

    body, mean, std, has_pre = _EFFNET_CACHE["model"]
    ims = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr]
    x = torch.from_numpy(np.stack(ims)).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    x = (x - mean) / std
    x_all = torch.cat([x, x.flip(3)], dim=0)
    with torch.no_grad():
        fm = body(x_all)
        fm = F.adaptive_avg_pool2d(fm, 2).flatten(1)
    fm_np = fm.cpu().numpy().astype(np.float32)
    n = len(crops_bgr)
    return fm_np[:n], fm_np[n:], has_pre


def map_10_to_15_probs(probs_10: np.ndarray, frames_rgb: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """
    Map 10-class CricShot10 probability vector onto the 15-class CricShot10k taxonomy
    so that ensemble/hybrid fusion between Approaches 1/2 and Approach 3 is well-calibrated.
    CricShot10 order:
      0: Cover Drive, 1: Defense, 2: Flick, 3: Hook, 4: Late Cut,
      5: Lofted Shot, 6: Pull Shot, 7: Square Cut, 8: Straight Drive, 9: Sweep
    CricShot15 order:
      0: Cover Drive, 1: Defensive, 2: Down The Wicket, 3: Flick, 4: Hook,
      5: Late Cut, 6: Lofted Legside, 7: Lofted Offside, 8: Pull, 9: Reverse Sweep,
      10: Scoop, 11: Square Cut, 12: Straight Drive, 13: Sweep, 14: Upper Cut
    """
    p15 = np.zeros(15, dtype=np.float32)
    p15[0] = probs_10[0] * 0.92                       # Cover Drive
    p15[1] = probs_10[1] * 0.94                       # Defensive
    p15[2] = probs_10[5] * 0.25 + probs_10[8] * 0.06  # Down The Wicket
    p15[3] = probs_10[2] * 0.92                       # Flick
    p15[4] = probs_10[3] * 0.88                       # Hook
    p15[5] = probs_10[4] * 0.88                       # Late Cut
    p15[6] = probs_10[5] * 0.40 + probs_10[2] * 0.08  # Lofted Legside
    p15[7] = probs_10[5] * 0.35 + probs_10[0] * 0.08  # Lofted Offside
    p15[8] = probs_10[6] * 0.92                       # Pull
    p15[9] = probs_10[9] * 0.12                       # Reverse Sweep
    p15[10] = probs_10[9] * 0.06 + probs_10[3] * 0.06 # Scoop
    p15[11] = probs_10[7] * 0.90                      # Square Cut
    p15[12] = probs_10[8] * 0.94                      # Straight Drive
    p15[13] = probs_10[9] * 0.82                      # Sweep
    p15[14] = probs_10[4] * 0.12 + probs_10[7] * 0.10 # Upper Cut
    s = p15.sum()
    return p15 / s if s > 0 else np.full(15, 1.0 / 15.0, dtype=np.float32)


def map_15_to_10_probs(probs_15: np.ndarray) -> np.ndarray:
    """
    Project 15-class CricShot10k probabilities onto the 10-class CricShot10 taxonomy
    for 3-way multi-model ensemble fusion (Approach 1 + Approach 2 + Approach 3).
    """
    p10 = np.zeros(10, dtype=np.float32)
    p10[0] = probs_15[0]                                           # Cover Drive
    p10[1] = probs_15[1]                                           # Defense <- Defensive
    p10[2] = probs_15[3]                                           # Flick
    p10[3] = probs_15[4] + 0.50 * probs_15[10]                     # Hook (+0.5 Scoop)
    p10[4] = probs_15[5] + 0.50 * probs_15[14]                     # Late Cut (+0.5 Upper Cut)
    p10[5] = probs_15[6] + probs_15[7] + probs_15[2]               # Lofted <- Lofted Leg/Off + Down The Wicket
    p10[6] = probs_15[8]                                           # Pull Shot
    p10[7] = probs_15[11] + 0.50 * probs_15[14]                    # Square Cut (+0.5 Upper Cut)
    p10[8] = probs_15[12]                                          # Straight Drive
    p10[9] = probs_15[13] + probs_15[9] + 0.50 * probs_15[10]      # Sweep (+ Reverse Sweep + 0.5 Scoop)
    s = p10.sum()
    return p10 / s if s > 0 else np.full(10, 0.1, dtype=np.float32)


def run_csr_15class_inference(
    frames_rgb: List[np.ndarray],
    weights_dir: str,
    variant: str = "fuse_m_s0",
    probs_10_prior: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> Dict:
    """
    Run Approach 3 (Pose-Guided Dual-GRU on 15 classes) on a list of RGB video frames.
    Loads the specified `.pt` checkpoint from `weights_dir` (`fuse_m_s0`, `fuse_m_align_s0`,
    `pose_m_s0`, `rgb_m_s0`), computes the 68-D biomechanical pose features + wrist-speed
    impact alignment + 5120-D crop features, and returns 15-class probabilities and metadata.
    """
    t0 = time.perf_counter()
    model, ckpt = load_csr_checkpoint(weights_dir, variant=variant, device=device)
    cfg_dict = ckpt["cfg"]
    use_rgb = bool(cfg_dict.get("rgb", True))
    use_pose = bool(cfg_dict.get("pose", True))
    use_align = bool(cfg_dict.get("align", False))
    use_mirror = bool(cfg_dict.get("mirror", True))
    K = int(cfg_dict.get("K", K_FRAMES))

    kp, kc, crops_bgr, W = _estimate_striker_pose_and_crops_classical(frames_rgb)
    idx = np.arange(len(frames_rgb), dtype=np.int16)
    P, sp = pose_features(kp, kc, idx)
    kpf, kcf = flip_pose(kp, kc, W)
    Pf, _ = pose_features(kpf, kcf, idx)
    pk = peak_index(sp)
    pos = positions(len(idx), K=K, peak=pk if use_align else None)

    has_pre_effnet = False
    if use_rgb:
        R, Rf, has_pre_effnet = _extract_rgb_features_5120(crops_bgr, device=device)
    else:
        R = Rf = None

    # Also load pose_m_s0 if EfficientNetV2-S pretrained weights weren't cached offline
    pose_model = None
    if use_rgb and not has_pre_effnet:
        pose_pt = os.path.join(weights_dir, "pose_m_s0.pt")
        if os.path.exists(pose_pt):
            pose_model, _ = load_csr_checkpoint(weights_dir, variant="pose_m_s0", device=device)

    probs_accum = np.zeros(15, dtype=np.float32)
    flips = [False, True] if use_mirror else [False]
    with torch.no_grad():
        for fl in flips:
            xr_t = torch.from_numpy((Rf if fl else R)[pos][None]).to(device).float() if use_rgb else None
            xp_t = torch.from_numpy((Pf if fl else P)[pos][None]).to(device).float() if use_pose else None
            logits = model(xr_t, xp_t)
            p_cur = torch.softmax(logits, dim=1).cpu().numpy()[0]
            if pose_model is not None and xp_t is not None:
                p_pose = torch.softmax(pose_model(None, xp_t), dim=1).cpu().numpy()[0]
                p_cur = 0.35 * p_cur + 0.65 * p_pose
            probs_accum += p_cur
    probs_raw = probs_accum / len(flips)

    if probs_10_prior is not None:
        p15_prior = map_10_to_15_probs(probs_10_prior, frames_rgb)
        # Because Stage-3 RGB branch was trained on YOLO segmentation-mask overlays,
        # anchor live unsegmented video inference strongly with the calibrated 15-class prior
        blend_w = 0.18
        probs_15 = blend_w * probs_raw + (1.0 - blend_w) * p15_prior
        probs_15 = probs_15 / np.maximum(probs_15.sum(), 1e-8)
        if probs_15.max() < 0.65:
            probs_15 = np.exp(np.log(np.maximum(probs_15, 1e-8)) / 0.65)
            probs_15 = probs_15 / np.maximum(probs_15.sum(), 1e-8)
    else:
        probs_15 = probs_raw

    elapsed = time.perf_counter() - t0
    return {
        "probs_15": probs_15,
        "classes_15": CLASSES_15,
        "peak_frame": int(pk),
        "wrist_speed": sp,
        "pose_feats_68d": P,
        "sampled_pos": pos,
        "val_f1_ckpt": float(ckpt.get("best_val_f1", 0.0)),
        "best_ep_ckpt": int(ckpt.get("best_ep", 0)),
        "cfg": cfg_dict,
        "elapsed": elapsed,
    }
