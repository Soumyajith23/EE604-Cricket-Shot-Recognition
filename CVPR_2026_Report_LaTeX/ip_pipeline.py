"""
src/ip_pipeline.py
==================
Classical & Hybrid Image Processing Pipeline for Cricket Shot Recognition (EE604).

This module implements five core Image Processing stages tailored to broadcast
cricket videos (e.g., CricShot10 dataset):

1. LAB-Space CLAHE Illumination Normalization:
   Enhances local contrast on the luminance channel L* while preserving chromaticity
   to handle variable stadium floodlights, shadows, and day/night broadcasts.

2. HSV Outfield/Pitch Segmentation & Morphological Crowd Suppression:
   Isolates the playing area (green turf + tan cricket pitch) and applies morphological
   closing/opening + spatial ROI weighting to suppress high-frequency background noise
   from stadium spectators and LED advertisement boards.

3. Temporal Motion History Image (MHI) & Motion Energy Image (MEI):
   Implements Bobick & Davis's recursive temporal template H_tau(x, y, t) over sampled
   video frames, compressing a multi-frame cricket stroke into a single grayscale/colorized
   motion-trail representation where pixel intensity encodes recency of motion.

4. Dense Farneback Optical Flow & Scale-Invariant HOOF (Histogram of Oriented Optical Flow):
   Computes dense 2D displacement fields (u, v) between consecutive keyframes, converts
   them into polar coordinates (magnitude M, orientation theta), and aggregates a 36-bin
   magnitude-weighted orientation histogram (HOOF) that characterizes stroke trajectory
   (e.g., upward arc in 'Lofted'/'Hook' vs. horizontal sweep in 'Pull'/'Square Cut').

5. Canny Edge Detection + Probabilistic Hough Transform Bat-Line & HOG Descriptors:
   Extracts structural gradients in the central batter ROI, detects dominant linear segments
   corresponding to the cricket bat and stance geometry via the Probabilistic Hough Transform,
   builds an 18-bin weighted line-orientation histogram + angular entropy feature, and
   computes a compact Histogram of Oriented Gradients (HOG) on the MHI template.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


CRICSHOT10_CLASSES: List[str] = [
    "cover",
    "defense",
    "flick",
    "hook",
    "late_cut",
    "lofted",
    "pull",
    "square_cut",
    "straight",
    "sweep",
]


class CricketIPPipeline:
    """
    End-to-end Image Processing feature extractor and visualizer for cricket shot videos.
    """

    def __init__(
        self,
        frame_size: Tuple[int, int] = (224, 224),
        num_keyframes: int = 8,
        clahe_clip_limit: float = 2.5,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        mhi_tau: float = 8.0,
        mhi_diff_thresh: int = 22,
        hoof_bins: int = 36,
        hough_bins: int = 18,
    ) -> None:
        """
        Args:
            frame_size: Spatial resolution (width, height) to resize frames to.
            num_keyframes: Number of uniformly sampled keyframes per video clip.
            clahe_clip_limit: Contrast limiting threshold for CLAHE on L* channel.
            clahe_grid_size: Tile grid size for local histogram equalization.
            mhi_tau: Decay duration (in frames) for the Motion History Image.
            mhi_diff_thresh: Intensity difference threshold for binary motion mask B(x,y,t).
            hoof_bins: Number of angular bins in [0, 2*pi) for Optical Flow histogram.
            hough_bins: Number of angular bins in [0, pi) for Hough line orientations.
        """
        self.frame_size = frame_size
        self.num_keyframes = num_keyframes
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.mhi_tau = float(mhi_tau)
        self.mhi_diff_thresh = mhi_diff_thresh
        self.hoof_bins = hoof_bins
        self.hough_bins = hough_bins

        # Initialize OpenCV HOGDescriptor (with OpenCV 5.0 Sobel-HOG fallback)
        if hasattr(cv2, "HOGDescriptor"):
            self.hog = cv2.HOGDescriptor(
                _winSize=(64, 64),
                _blockSize=(16, 16),
                _blockStride=(16, 16),
                _cellSize=(8, 8),
                _nbins=9,
            )
        else:
            self.hog = None

    # -------------------------------------------------------------------------
    # Stage 0: Video Keyframe Sampling
    # -------------------------------------------------------------------------
    def sample_keyframes(self, video_path: str) -> List[np.ndarray]:
        """
        Reads a video file (.avi / .mp4) and extracts `num_keyframes` uniformly
        spaced RGB frames focused around the central 80% active stroke window.

        Args:
            video_path: Path to the input video file.

        Returns:
            List of `num_keyframes` RGB uint8 numpy arrays of shape (H, W, 3).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        raw_frames_full: List[np.ndarray] = []
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            raw_frames_full.append(frame_rgb)
        cap.release()

        if len(raw_frames_full) == 0:
            raise ValueError(f"No readable frames found in video: {video_path}")

        # Auto-crop black pillarbox / letterbox bars (e.g. from video editors)
        mid_f = raw_frames_full[len(raw_frames_full) // 2]
        gray_mid = cv2.cvtColor(mid_f, cv2.COLOR_RGB2GRAY)
        col_mean = np.mean(gray_mid, axis=0)
        row_mean = np.mean(gray_mid, axis=1)
        valid_cols = np.where(col_mean > 20.0)[0]
        valid_rows = np.where(row_mean > 20.0)[0]
        c0 = int(valid_cols[0]) if len(valid_cols) > 16 else 0
        c1 = int(valid_cols[-1] + 1) if len(valid_cols) > 16 else mid_f.shape[1]
        r0 = int(valid_rows[0]) if len(valid_rows) > 16 else 0
        r1 = int(valid_rows[-1] + 1) if len(valid_rows) > 16 else mid_f.shape[0]

        raw_frames = [
            cv2.resize(f[r0:r1, c0:c1], self.frame_size, interpolation=cv2.INTER_AREA)
            for f in raw_frames_full
        ]

        # Motion-energy active window selection: locate batter's post-delivery swing peak
        total = len(raw_frames)
        if total > self.num_keyframes + 4:
            diffs = []
            for t in range(1, total):
                # Focus strictly on batter's upper torso & bat swing zone (excluding bottom bowler release)
                g1 = cv2.cvtColor(raw_frames[t][24:148, 56:168], cv2.COLOR_RGB2GRAY).astype(np.float32)
                g0 = cv2.cvtColor(raw_frames[t - 1][24:148, 56:168], cv2.COLOR_RGB2GRAY).astype(np.float32)
                diffs.append(float(np.mean(np.abs(g1 - g0))))
            diffs_arr = np.convolve(np.array(diffs, dtype=np.float32), np.ones(3) / 3.0, mode="same")
            # Batter's stroke occurs after bowler's release (search in post-release window >= 35% of clip)
            search_start = int(0.35 * len(diffs_arr))
            peak_t = search_start + int(np.argmax(diffs_arr[search_start:])) + 1
            win_len = max(self.num_keyframes + 2, int(0.45 * total))
            start_idx = max(0, min(total - win_len, peak_t - int(0.55 * win_len)))
            end_idx = min(total, start_idx + win_len)
            active_frames = raw_frames[start_idx:end_idx]
        else:
            active_frames = raw_frames

        indices = np.linspace(
            0, len(active_frames) - 1, self.num_keyframes, dtype=int
        )
        sampled = [active_frames[i] for i in indices]
        return sampled

    # -------------------------------------------------------------------------
    # Stage 1: LAB-Space CLAHE Illumination Normalization
    # -------------------------------------------------------------------------
    def apply_clahe_lab(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) strictly
        on the Luminance (L*) channel in CIE-LAB color space to prevent hue distortion.

        Args:
            rgb_frame: Input RGB image (H, W, 3), dtype uint8.

        Returns:
            CLAHE-enhanced RGB image (H, W, 3), dtype uint8.
        """
        lab = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size
        )
        l_eq = clahe.apply(l_chan)

        lab_eq = cv2.merge((l_eq, a_chan, b_chan))
        rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return rgb_eq

    # -------------------------------------------------------------------------
    # Stage 2: HSV Field/Pitch Segmentation & Crowd Suppression
    # -------------------------------------------------------------------------
    def suppress_crowd_and_segment_roi(
        self, rgb_frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segments the cricket playing surface (green turf + tan clay pitch) in HSV space
        and constructs a soft spatial-morphological attention mask that suppresses
        background spectator crowds and advertising banners while preserving the batter.

        Args:
            rgb_frame: CLAHE-enhanced RGB image (H, W, 3).

        Returns:
            Tuple of:
                - suppressed_rgb: Crowd-attenuated RGB frame (H, W, 3), uint8.
                - field_mask: Binary playing-area + batter ROI mask (H, W), uint8.
                - pitch_stats: 6-dim vector of HSV color/spatial statistics.
        """
        h, w, _ = rgb_frame.shape
        hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)

        # 1. Green turf mask (Hue roughly 28..95 in OpenCV [0, 180] scale)
        lower_green = np.array([28, 30, 35], dtype=np.uint8)
        upper_green = np.array([95, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # 2. Tan/brown clay pitch strip mask (Hue roughly 10..28, moderate saturation, high value)
        lower_pitch = np.array([10, 20, 80], dtype=np.uint8)
        upper_pitch = np.array([28, 170, 255], dtype=np.uint8)
        pitch_mask = cv2.inRange(hsv, lower_pitch, upper_pitch)

        combined_mask = cv2.bitwise_or(green_mask, pitch_mask)

        # 3. Morphological closing (fill player/bat interior holes) & opening (remove confetti noise)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morphed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel_open)

        # 4. Construct Gaussian Central Batter Prior (batter is typically centered in broadcast view)
        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, h), np.linspace(-1.0, 1.0, w), indexing="ij"
        )
        # Elliptical Gaussian prior slightly shifted vertically toward the crease
        batter_prior = np.exp(-((xx / 0.65) ** 2 + ((yy - 0.05) / 0.75) ** 2))

        # Combine morphological field mask with batter prior so the batter's upper torso/helmet
        # is never clipped even when standing in front of sight-screen or stands
        soft_mask = np.clip(
            0.45 * (morphed.astype(np.float32) / 255.0) + 0.75 * batter_prior,
            0.20,
            1.0,
        )
        soft_mask = cv2.GaussianBlur(soft_mask, (11, 11), 0)

        suppressed_rgb = np.clip(
            rgb_frame.astype(np.float32) * soft_mask[..., None], 0, 255
        ).astype(np.uint8)

        # Compute 6-dim scene descriptor (green ratio, pitch ratio, mean/std HSV in central ROI)
        green_ratio = float(np.mean(green_mask > 0))
        pitch_ratio = float(np.mean(pitch_mask > 0))
        central_hsv = hsv[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        mean_h = float(np.mean(central_hsv[:, :, 0]) / 180.0)
        mean_s = float(np.mean(central_hsv[:, :, 1]) / 255.0)
        mean_v = float(np.mean(central_hsv[:, :, 2]) / 255.0)
        std_v = float(np.std(central_hsv[:, :, 2]) / 255.0)

        pitch_stats = np.array(
            [green_ratio, pitch_ratio, mean_h, mean_s, mean_v, std_v],
            dtype=np.float32,
        )
        binary_display_mask = (soft_mask * 255.0).astype(np.uint8)
        return suppressed_rgb, binary_display_mask, pitch_stats

    # -------------------------------------------------------------------------
    # Stage 3: Temporal Motion History Image (MHI) & Motion Energy Image (MEI)
    # -------------------------------------------------------------------------
    def compute_mhi_and_mei(
        self, frames: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes Bobick & Davis's Motion History Image H_tau(x, y, t) and Motion Energy
        Image E_tau(x, y, t) across a sequence of preprocessed RGB keyframes:

            H_tau(x, y, t) = tau                         if Psi(x, y, t) == 1
                           = max(0, H_tau(x, y, t-1)-1)  otherwise

        where Psi(x, y, t) is the morphologically filtered binary frame-difference silhouette.

        Args:
            frames: List of RGB frames of shape (H, W, 3).

        Returns:
            Tuple of:
                - mhi_norm: Normalized grayscale MHI in [0, 255], shape (H, W), uint8.
                - mei_mask: Binary Motion Energy Image in {0, 255}, shape (H, W), uint8.
                - mhi_color: Jet/Turbo-colormapped RGB visualization of MHI (H, W, 3), uint8.
        """
        h, w, _ = frames[0].shape
        mhi = np.zeros((h, w), dtype=np.float32)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        gray_frames = [
            cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (5, 5), 0)
            for f in frames
        ]

        for t in range(1, len(gray_frames)):
            diff = cv2.absdiff(gray_frames[t], gray_frames[t - 1])
            _, motion_mask = cv2.threshold(
                diff, self.mhi_diff_thresh, 1, cv2.THRESH_BINARY
            )
            # Clean isolated speckle noise via morphological opening
            motion_mask = cv2.morphologyEx(
                motion_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
            )
            # Suppress top broadcast crowd banner (top 18%) and bottom TV scoreboard ticker (bottom 18%)
            motion_mask[: int(0.18 * h), :] = 0
            motion_mask[int(0.82 * h) :, :] = 0

            # Recursive Bobick-Davis update
            mhi = np.where(
                motion_mask > 0,
                self.mhi_tau,
                np.maximum(0.0, mhi - 1.0),
            )

        mhi_norm = np.clip((mhi / self.mhi_tau) * 255.0, 0, 255).astype(np.uint8)
        mei_mask = np.where(mhi_norm > 0, 255, 0).astype(np.uint8)

        mhi_color_bgr = cv2.applyColorMap(mhi_norm, cv2.COLORMAP_TURBO)
        # Keep zero-motion background dark
        mhi_color_bgr[mhi_norm == 0] = 0
        mhi_color_rgb = cv2.cvtColor(mhi_color_bgr, cv2.COLOR_BGR2RGB)

        return mhi_norm, mei_mask, mhi_color_rgb

    # -------------------------------------------------------------------------
    # Stage 4: Dense Farneback Optical Flow & Scale-Invariant HOOF Descriptor
    # -------------------------------------------------------------------------
    def compute_optical_flow_and_hoof(
        self, frames: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes Dense Farneback Optical Flow across consecutive keyframes, compensates
        for global camera pan (by subtracting median background motion), and extracts a
        scale-invariant Histogram of Oriented Optical Flow (HOOF) in the batter ROI.

        Args:
            frames: List of RGB keyframes (H, W, 3).

        Returns:
            Tuple of:
                - hoof_descriptor: Normalized (hoof_bins + 4)-dim motion vector.
                - flow_rgb_vis: HSV-encoded dense optical flow visualization (H, W, 3), uint8.
                - flow_field: Averaged (u, v) optical flow field (H, W, 2), float32.
        """
        h, w, _ = frames[0].shape
        gray_seq = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]

        # Focus HOOF accumulation on the central batter ROI [15%..85% H, 20%..80% W]
        r0, r1 = int(0.15 * h), int(0.85 * h)
        c0, c1 = int(0.20 * w), int(0.80 * w)

        hoof_hist = np.zeros(self.hoof_bins, dtype=np.float32)
        accum_flow = np.zeros((h, w, 2), dtype=np.float32)
        peak_mag_sum = -1.0
        peak_flow = np.zeros((h, w, 2), dtype=np.float32)
        step_magnitudes: List[float] = []

        for t in range(1, len(gray_seq)):
            flow = cv2.calcOpticalFlowFarneback(
                gray_seq[t - 1],
                gray_seq[t],
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

            # Global camera-pan compensation using perimeter border median flow
            border_u = np.concatenate([flow[:15, :, 0].ravel(), flow[-15:, :, 0].ravel()])
            border_v = np.concatenate([flow[:15, :, 1].ravel(), flow[-15:, :, 1].ravel()])
            flow[:, :, 0] -= np.median(border_u)
            flow[:, :, 1] -= np.median(border_v)

            accum_flow += flow
            roi_flow = flow[r0:r1, c0:c1]
            mag, ang = cv2.cartToPolar(
                roi_flow[:, :, 0], roi_flow[:, :, 1], angleInDegrees=False
            )

            # Filter out sub-pixel camera jitter
            active = mag > 1.0
            if np.any(active):
                hist, _ = np.histogram(
                    ang[active],
                    bins=self.hoof_bins,
                    range=(0.0, 2.0 * np.pi),
                    weights=mag[active],
                )
                hoof_hist += hist.astype(np.float32)

            curr_mag_sum = float(np.sum(mag))
            step_magnitudes.append(curr_mag_sum / float(mag.size))
            if curr_mag_sum > peak_mag_sum:
                peak_mag_sum = curr_mag_sum
                peak_flow = flow.copy()

        # Scale-invariant L2 normalization of HOOF histogram
        hoof_norm = hoof_hist / (np.linalg.norm(hoof_hist) + 1e-6)

        # Temporal kinematics summary statistics (mean speed, peak speed, temporal peak location, verticality ratio)
        mean_speed = float(np.mean(step_magnitudes)) if step_magnitudes else 0.0
        max_speed = float(np.max(step_magnitudes)) if step_magnitudes else 0.0
        peak_time_norm = (
            float(np.argmax(step_magnitudes)) / max(1, len(step_magnitudes) - 1)
            if step_magnitudes
            else 0.5
        )
        avg_flow = accum_flow / max(1, len(gray_seq) - 1)
        vert_horiz_ratio = float(
            np.mean(np.abs(avg_flow[r0:r1, c0:c1, 1]))
            / (np.mean(np.abs(avg_flow[r0:r1, c0:c1, 0])) + 1e-5)
        )

        kinematic_stats = np.array(
            [
                np.tanh(mean_speed / 5.0),
                np.tanh(max_speed / 10.0),
                peak_time_norm,
                np.tanh(vert_horiz_ratio),
            ],
            dtype=np.float32,
        )
        hoof_descriptor = np.concatenate([hoof_norm, kinematic_stats], axis=0)

        # Create HSV color-wheel optical flow visualization from the peak stroke frame pair
        mag_vis, ang_vis = cv2.cartToPolar(
            peak_flow[:, :, 0], peak_flow[:, :, 1], angleInDegrees=True
        )
        hsv_vis = np.zeros((h, w, 3), dtype=np.uint8)
        hsv_vis[:, :, 0] = (ang_vis / 2.0).astype(np.uint8)
        hsv_vis[:, :, 1] = 255
        hsv_vis[:, :, 2] = cv2.normalize(
            mag_vis, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        flow_rgb_vis = cv2.cvtColor(hsv_vis, cv2.COLOR_HSV2RGB)

        return hoof_descriptor, flow_rgb_vis, avg_flow

    # -------------------------------------------------------------------------
    # Stage 5: Canny Edge + Probabilistic Hough Transform Bat/Pose Geometry
    # -------------------------------------------------------------------------
    def extract_canny_hough_and_hog(
        self, peak_rgb: np.ndarray, mhi_norm: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies adaptive Canny edge detection and Probabilistic Hough Transform within
        the batter ROI to estimate dominant bat & limb orientations, and extracts a
        spatial HOG descriptor from the Motion History Image.

        Args:
            peak_rgb: Preprocessed RGB keyframe at the stroke impact/peak (H, W, 3).
            mhi_norm: Grayscale Motion History Image (H, W), uint8.

        Returns:
            Tuple of:
                - geom_descriptor: (hough_bins + 2 + 36)-dim vector combining Hough line
                  orientation histogram, angular entropy, dominant angle, and MHI-HOG.
                - hough_overlay: RGB image with Canny edges & detected Hough lines overlaid.
                - hough_hist_norm: 18-bin normalized line angle histogram in [0, 180) deg.
        """
        h, w, _ = peak_rgb.shape
        gray = cv2.cvtColor(peak_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

        # Otsu/median-based automatic Canny thresholds
        med_val = float(np.median(blurred))
        lower_t = int(max(25, 0.66 * med_val))
        upper_t = int(min(220, 1.33 * med_val))
        edges = cv2.Canny(blurred, lower_t, upper_t)

        # Restrict Hough line detection to the batter's central action zone
        # (excluding top 18% stadium crowd and bottom 20% broadcast TV scoreboard overlay)
        roi_mask = np.zeros_like(edges)
        r0, r1 = int(0.18 * h), int(0.80 * h)
        c0, c1 = int(0.22 * w), int(0.78 * w)
        roi_mask[r0:r1, c0:c1] = 255
        roi_edges = cv2.bitwise_and(edges, roi_mask)

        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1.0,
            theta=np.pi / 180.0,
            threshold=22,
            minLineLength=16,
            maxLineGap=8,
        )

        hough_hist = np.zeros(self.hough_bins, dtype=np.float32)
        overlay = (peak_rgb.astype(np.float32) * 0.45).astype(np.uint8)
        # Render Canny edges in cyan
        overlay[roi_edges > 0] = [0, 210, 255]

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = [int(v) for v in np.asarray(line).ravel()[:4]]
                dx = float(x2 - x1)
                dy = float(y2 - y1)
                length = float(np.hypot(dx, dy))
                if length < 12.0:
                    continue

                # Orientation in [0, 180) degrees
                angle_deg = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0
                # Suppress strictly horizontal pitch-crease lines (<8 deg or >172 deg)
                if angle_deg < 8.0 or angle_deg > 172.0:
                    continue
                bin_idx = min(
                    self.hough_bins - 1,
                    int((angle_deg / 180.0) * self.hough_bins),
                )
                hough_hist[bin_idx] += length

                # Highlight strong straight segments (candidate bat/limb axes) in gold/red
                color = (255, 215, 0) if length > 28.0 else (50, 255, 120)
                cv2.line(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        hough_hist_norm = hough_hist / (np.linalg.norm(hough_hist) + 1e-6)
        prob_dist = hough_hist / (np.sum(hough_hist) + 1e-6)
        angular_entropy = float(
            -np.sum(prob_dist * np.log(prob_dist + 1e-8))
            / np.log(self.hough_bins)
        )
        dominant_bin_norm = float(np.argmax(hough_hist)) / float(self.hough_bins)

        # Compute compact HOG on 64x64 resized MHI + Edge composite
        mhi_edge_composite = cv2.addWeighted(mhi_norm, 0.7, roi_edges, 0.3, 0)
        mhi_64 = cv2.resize(
            mhi_edge_composite, (64, 64), interpolation=cv2.INTER_AREA
        )
        if self.hog is not None:
            raw_hog = self.hog.compute(mhi_64).ravel()  # 576-dim (16 blocks x 36 dims)
            pooled_hog = raw_hog.reshape(16, 36).mean(axis=0).astype(np.float32)
        else:
            gx = cv2.Sobel(mhi_64, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(mhi_64, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.hypot(gx, gy)
            ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
            bins = np.clip((ang / 20.0).astype(np.int32), 0, 8)
            cells = []
            for r in range(2):
                for c in range(2):
                    sub_b = bins[r * 32 : (r + 1) * 32, c * 32 : (c + 1) * 32].ravel()
                    sub_m = mag[r * 32 : (r + 1) * 32, c * 32 : (c + 1) * 32].ravel()
                    cells.append(np.bincount(sub_b, weights=sub_m, minlength=9)[:9])
            pooled_hog = np.concatenate(cells, axis=0).astype(np.float32)
        pooled_hog = pooled_hog / (np.linalg.norm(pooled_hog) + 1e-6)

        geom_descriptor = np.concatenate(
            [
                hough_hist_norm,
                np.array([angular_entropy, dominant_bin_norm], dtype=np.float32),
                pooled_hog,
            ],
            axis=0,
        )
        return geom_descriptor, overlay, hough_hist_norm

    # -------------------------------------------------------------------------
    # Complete End-to-End Video Processing Entrypoint
    # -------------------------------------------------------------------------
    def process_video(self, video_path: str) -> Dict[str, np.ndarray]:
        """
        Executes all five Image Processing stages on a video clip and returns both
        numerical feature tensors (for CricIP-FusionNet) and visual intermediate maps
        (for the CVPR report figures and interactive demo UI).

        Args:
            video_path: Path to the input cricket video (.avi / .mp4).

        Returns:
            Dictionary containing:
                - 'raw_keyframes': (T, H, W, 3) uint8
                - 'clahe_keyframes': (T, H, W, 3) uint8
                - 'suppressed_keyframes': (T, H, W, 3) uint8
                - 'roi_mask': (H, W) uint8
                - 'mhi_gray': (H, W) uint8
                - 'mei_mask': (H, W) uint8
                - 'mhi_color': (H, W, 3) uint8
                - 'flow_rgb': (H, W, 3) uint8
                - 'hough_overlay': (H, W, 3) uint8
                - 'motion_composite': (H, W, 3) uint8 (3-channel IP motion map for Stream 2 CNN)
                - 'hoof_hist': (36,) float32
                - 'hough_hist': (18,) float32
                - 'ip_feature_vector': (102,) float32 [40 HOOF+Kinematics + 56 Hough+HOG + 6 PitchStats]
        """
        raw_frames = self.sample_keyframes(video_path)
        clahe_frames = [self.apply_clahe_lab(f) for f in raw_frames]

        suppressed_frames: List[np.ndarray] = []
        pitch_stats_list: List[np.ndarray] = []
        mid_idx = len(clahe_frames) // 2
        mid_roi_mask = np.zeros(self.frame_size[::-1], dtype=np.uint8)

        for idx, cf in enumerate(clahe_frames):
            supp, mask, pstats = self.suppress_crowd_and_segment_roi(cf)
            suppressed_frames.append(supp)
            pitch_stats_list.append(pstats)
            if idx == mid_idx:
                mid_roi_mask = mask

        pitch_stats_avg = np.mean(np.stack(pitch_stats_list, axis=0), axis=0)

        # Stage 3: MHI & MEI
        mhi_gray, mei_mask, mhi_color = self.compute_mhi_and_mei(suppressed_frames)

        # Stage 4: Dense Optical Flow & HOOF
        hoof_desc, flow_rgb, avg_flow = self.compute_optical_flow_and_hoof(
            suppressed_frames
        )

        # Stage 5: Canny-Hough Bat/Pose Geometry & MHI-HOG
        geom_desc, hough_overlay, hough_hist = self.extract_canny_hough_and_hog(
            suppressed_frames[mid_idx], mhi_gray
        )

        # Build a 3-channel Spatio-Temporal IP Motion Composite image for Stream-2 CNN:
        # Channel 0: Normalized MHI, Channel 1: Flow Magnitude, Channel 2: Canny/MEI Silhouette
        flow_gray = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2GRAY)
        motion_composite = np.stack([mhi_gray, flow_gray, mei_mask], axis=-1)

        # Full 102-dimensional explicit Image Processing feature vector:
        # 40 (HOOF + kinematics) + 56 (18 Hough + 2 entropy/dom + 36 HOG) + 6 (Pitch HSV stats) = 102 dims
        ip_feature_vector = np.concatenate(
            [hoof_desc, geom_desc, pitch_stats_avg], axis=0
        ).astype(np.float32)

        return {
            "raw_keyframes": np.stack(raw_frames, axis=0),
            "clahe_keyframes": np.stack(clahe_frames, axis=0),
            "suppressed_keyframes": np.stack(suppressed_frames, axis=0),
            "roi_mask": mid_roi_mask,
            "mhi_gray": mhi_gray,
            "mei_mask": mei_mask,
            "mhi_color": mhi_color,
            "flow_rgb": flow_rgb,
            "hough_overlay": hough_overlay,
            "motion_composite": motion_composite,
            "hoof_hist": hoof_desc[: self.hoof_bins],
            "hough_hist": hough_hist,
            "ip_feature_vector": ip_feature_vector,
        }

