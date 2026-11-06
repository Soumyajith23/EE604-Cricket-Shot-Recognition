"""
src/dataset.py
==============
PyTorch Dataset, Feature Caching, and Synthetic Demo Clip Generator for CricShot10.

Expected directory structure of CricShot10 dataset:
    <dataset_root>/
        cover/          (*.avi, *.mp4)
        defense/
        flick/
        hook/
        late_cut/
        lofted/
        pull/
        square_cut/
        straight/
        sweep/
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .ip_pipeline import CRICSHOT10_CLASSES, CricketIPPipeline


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_image_tensor(img_uint8: np.ndarray) -> torch.Tensor:
    """
    Converts an (H, W, 3) or (T, H, W, 3) uint8 RGB array to an ImageNet-normalized
    float32 PyTorch tensor of shape (3, H, W) or (T, 3, H, W).
    """
    arr = img_uint8.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    if arr.ndim == 3:
        return torch.from_numpy(np.transpose(arr, (2, 0, 1)).copy())
    elif arr.ndim == 4:
        return torch.from_numpy(np.transpose(arr, (0, 3, 1, 2)).copy())
    raise ValueError(f"Unsupported array shape: {img_uint8.shape}")


def discover_cricshot10_videos(
    dataset_root: str, max_per_class: Optional[int] = None
) -> List[Tuple[str, int, str]]:
    """
    Scans `dataset_root` for the 10 CricShot10 class subdirectories and collects
    all video file paths (.avi, .mp4, .mkv, .mov).

    Returns:
        List of tuples: (video_path, class_index, class_name)
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    samples: List[Tuple[str, int, str]] = []
    valid_exts = {".avi", ".mp4", ".mkv", ".mov"}

    for cls_idx, cls_name in enumerate(CRICSHOT10_CLASSES):
        cls_dir = root / cls_name
        if not cls_dir.exists():
            # Try case-insensitive or space/underscore fallback
            candidates = [
                d
                for d in root.iterdir()
                if d.is_dir()
                and d.name.lower().replace(" ", "_") == cls_name.lower()
            ]
            if candidates:
                cls_dir = candidates[0]
            else:
                continue

        cls_files = sorted(
            [
                str(p)
                for p in cls_dir.rglob("*")
                if p.suffix.lower() in valid_exts
            ]
        )
        if max_per_class is not None:
            cls_files = cls_files[:max_per_class]

        for fpath in cls_files:
            samples.append((fpath, cls_idx, cls_name))

    return samples


class CricShot10Dataset(Dataset):
    """
    PyTorch Dataset for CricShot10 that extracts and optionally caches:
      - raw_keyframes: (T, 3, H, W)
      - suppressed_keyframes: (T, 3, H, W) [CLAHE + HSV Crowd Suppression]
      - motion_composite: (3, H, W) [MHI + Optical Flow + MEI]
      - ip_feature_vector: (102,) [HOOF + Canny-Hough + MHI-HOG + HSV stats]
    """

    def __init__(
        self,
        samples: List[Tuple[str, int, str]],
        pipeline: Optional[CricketIPPipeline] = None,
        cache_dir: Optional[str] = None,
        use_clahe_suppression: bool = True,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.pipeline = pipeline or CricketIPPipeline()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_clahe_suppression = use_clahe_suppression
        self.augment = augment

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _get_cache_path(self, video_path: str, cls_name: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        stem = Path(video_path).stem
        return self.cache_dir / f"{cls_name}_{stem}.npz"

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_path, label, cls_name = self.samples[idx]
        cache_path = self._get_cache_path(video_path, cls_name)

        data_dict = None
        if cache_path is not None and cache_path.exists():
            try:
                loaded = np.load(str(cache_path))
                data_dict = {
                    "raw_keyframes": loaded["raw_keyframes"],
                    "suppressed_keyframes": loaded["suppressed_keyframes"],
                    "motion_composite": loaded["motion_composite"],
                    "ip_feature_vector": loaded["ip_feature_vector"],
                }
            except Exception:
                data_dict = None

        if data_dict is None:
            processed = self.pipeline.process_video(video_path)
            data_dict = {
                "raw_keyframes": processed["raw_keyframes"],
                "suppressed_keyframes": processed["suppressed_keyframes"],
                "motion_composite": processed["motion_composite"],
                "ip_feature_vector": processed["ip_feature_vector"],
            }
            if cache_path is not None:
                np.savez_compressed(
                    str(cache_path),
                    raw_keyframes=data_dict["raw_keyframes"],
                    suppressed_keyframes=data_dict["suppressed_keyframes"],
                    motion_composite=data_dict["motion_composite"],
                    ip_feature_vector=data_dict["ip_feature_vector"],
                )

        keyframes_np = (
            data_dict["suppressed_keyframes"]
            if self.use_clahe_suppression
            else data_dict["raw_keyframes"]
        )
        motion_np = data_dict["motion_composite"]
        ip_vec_np = data_dict["ip_feature_vector"].copy()

        # Optional random horizontal flip augmentation during training
        if self.augment and np.random.rand() < 0.5:
            keyframes_np = keyframes_np[:, :, ::-1, :].copy()
            motion_np = motion_np[:, ::-1, :].copy()

        keyframes_t = normalize_image_tensor(keyframes_np)
        motion_t = normalize_image_tensor(motion_np)
        ip_vec_t = torch.from_numpy(ip_vec_np).float()
        label_t = torch.tensor(label, dtype=torch.long)

        return {
            "keyframes": keyframes_t,
            "motion_map": motion_t,
            "ip_vec": ip_vec_t,
            "label": label_t,
        }


# -----------------------------------------------------------------------------
# Synthetic Cricket Stroke Video Generator for Offline Demo & Pipeline Testing
# -----------------------------------------------------------------------------
def generate_demo_cricket_videos(output_dir: str) -> List[str]:
    """
    Generates realistic 16-frame synthetic cricket broadcast clips (.mp4) with
    distinct bat swing kinematics, stadium turf, clay pitch, crowd background,
    and moving ball/bat trajectories for testing the UI and IP pipeline offline.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Stroke kinematic profiles: (start_angle_deg, end_angle_deg, swing_dx, swing_dy, ball_dir)
    stroke_profiles = {
        "cover": (75, 15, -35, -10, (-45, 25)),
        "defense": (88, 82, 0, 5, (0, 10)),
        "flick": (80, 135, 30, -5, (45, 20)),
        "hook": (45, 165, 38, -32, (50, -35)),
        "late_cut": (60, 10, -40, 12, (-50, -10)),
        "lofted": (85, 145, 15, -42, (10, -55)),
        "pull": (35, 155, 42, -12, (55, -10)),
        "square_cut": (40, 5, -45, 0, (-60, 0)),
        "straight": (88, 65, 2, -25, (0, 45)),
        "sweep": (25, 160, 45, 18, (50, 15)),
    }

    created_files: List[str] = []
    h, w = 224, 224
    num_frames = 16

    for cls_name, (ang0, ang1, sdx, sdy, (bdx, bdy)) in stroke_profiles.items():
        cls_folder = out_path / cls_name
        cls_folder.mkdir(parents=True, exist_ok=True)
        video_file = cls_folder / f"sample_{cls_name}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_file), fourcc, 12.0, (w, h))

        rng = np.random.RandomState(abs(hash(cls_name)) % 10000)
        crowd_texture = rng.randint(40, 180, size=(55, w, 3), dtype=np.uint8)

        for t in range(num_frames):
            alpha = t / float(num_frames - 1)
            frame = np.zeros((h, w, 3), dtype=np.uint8)

            # 1. Stadium crowd top banner
            frame[:55, :] = crowd_texture
            # 2. Green outfield (BGR)
            frame[55:, :] = (45, 135, 55)
            # 3. Tan clay pitch strip in center
            pts_pitch = np.array(
                [[82, 65], [142, 65], [162, 220], [62, 220]], dtype=np.int32
            )
            cv2.fillPoly(frame, [pts_pitch], (135, 175, 200))
            # Crease markings
            cv2.line(frame, (72, 145), (152, 145), (245, 245, 245), 2)

            # 4. Batter silhouette (torso, legs, helmet)
            cx = int(112 + alpha * (sdx * 0.25))
            cy = int(115 + alpha * (sdy * 0.20))
            # Legs (white pads)
            cv2.line(frame, (cx - 6, cy + 12), (cx - 10, cy + 45), (235, 235, 235), 6)
            cv2.line(frame, (cx + 6, cy + 12), (cx + 10, cy + 45), (235, 235, 235), 6)
            # Torso (jersey)
            cv2.rectangle(
                frame, (cx - 10, cy - 16), (cx + 10, cy + 14), (190, 85, 35), -1
            )
            # Helmet
            cv2.circle(frame, (cx, cy - 24), 9, (90, 40, 20), -1)

            # 5. Dynamic Bat Swing
            cur_ang = np.radians(ang0 + alpha * (ang1 - ang0))
            wrist_x = int(cx + 8 + alpha * (sdx * 0.35))
            wrist_y = int(cy - 2 + alpha * (sdy * 0.35))
            bat_len = 38
            tip_x = int(wrist_x + bat_len * np.cos(cur_ang))
            tip_y = int(wrist_y + bat_len * np.sin(cur_ang))

            # Draw arms & wooden cricket bat
            cv2.line(frame, (cx, cy - 8), (wrist_x, wrist_y), (220, 220, 220), 4)
            cv2.line(
                frame, (wrist_x, wrist_y), (tip_x, tip_y), (70, 165, 230), 6, cv2.LINE_AA
            )

            # 6. Moving Cricket Ball
            if alpha < 0.5:
                bx = int(112 - (0.5 - alpha) * 40)
                by = int(195 - alpha * 110)
            else:
                post = (alpha - 0.5) * 2.0
                bx = int(wrist_x + post * bdx)
                by = int(wrist_y + post * bdy)
            if 0 <= bx < w and 0 <= by < h:
                cv2.circle(frame, (bx, by), 4, (30, 30, 220), -1)

            writer.write(frame)

        writer.release()
        created_files.append(str(video_file))

    return created_files

