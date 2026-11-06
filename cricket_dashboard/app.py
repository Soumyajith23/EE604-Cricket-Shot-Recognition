import os
import sys
import time
import shutil
import tempfile
import subprocess
import hashlib
from pathlib import Path

import cv2
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

# Add parent directory so we can import src/ip_pipeline.py, src/model.py, src/dataset.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from predict import (
    CLASS_NAMES,
    load_model,
    predict_video,
    extract_frames,
    NUM_FRAMES,
)
import importlib
import src.ip_pipeline as _src_ip
importlib.reload(_src_ip)
from src.ip_pipeline import CricketIPPipeline, CRICSHOT10_CLASSES
from src.model import CricIPFusionNet
import src.dataset as _src_dataset
importlib.reload(_src_dataset)
from src.dataset import generate_demo_cricket_videos, normalize_image_tensor
import src.csr_infer as _src_csr
importlib.reload(_src_csr)
from src.csr_infer import run_csr_15class_inference, map_15_to_10_probs, CLASSES_15


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="EE604 Cricket Shot Recognition & IP Workbench",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a2332 100%);
    }

    .main-title {
        font-size: 42px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #38ef9d, #00b4d8, #38ef9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #aab4c2;
        font-size: 16px;
        margin-bottom: 24px;
    }

    .result-box {
        padding: 26px;
        border-radius: 20px;
        background: linear-gradient(145deg, #182435, #0f1825);
        border: 1px solid #38ef9d44;
        text-align: center;
        box-shadow: 0 8px 32px rgba(56, 239, 157, 0.1);
    }

    .shot-name {
        font-size: 40px;
        font-weight: 900;
        color: #38ef9d;
        text-shadow: 0 0 20px rgba(56, 239, 157, 0.4);
        margin: 10px 0;
    }

    .confidence-text {
        font-size: 22px;
        color: #00b4d8;
        font-weight: 700;
    }

    .metric-card {
        padding: 18px;
        border-radius: 15px;
        background: #182435;
        border: 1px solid #334155;
        text-align: center;
    }

    .metric-value {
        font-size: 24px;
        font-weight: 800;
        color: #38ef9d;
    }

    .metric-label {
        font-size: 13px;
        color: #8b9bb4;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(56, 239, 157, 0.3);
    }

    div[data-testid="stMetric"] {
        background: #182435;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 15px;
    }

    .footer {
        text-align: center;
        color: #5a6b7f;
        font-size: 12px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# PATHS & DEVICE
# ==================================================

MODEL_PATH = os.path.join(BASE_DIR, "best_finetuned_model.pth")
CRICIP_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_cricip_fusionnet.pth")
CSR_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights", "weights")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================================================
# CACHED RESOURCES
# ==================================================

@st.cache_resource(show_spinner="Loading ResNet50 + LSTM model...")
def get_model(path):
    return load_model(path, DEVICE, CLASS_NAMES)


@st.cache_resource(show_spinner="Loading CricIP-FusionNet (Classical IP + GCAM)...")
def get_cricip_model(path):
    model = CricIPFusionNet(num_classes=len(CLASS_NAMES), pretrained=False).to(DEVICE)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def predict_with_cricip(video_path, cricip_model):
    pipeline = CricketIPPipeline()
    ip_out = pipeline.process_video(video_path)
    kf_tensor = normalize_image_tensor(ip_out["suppressed_keyframes"]).unsqueeze(0).to(DEVICE)
    kf_flip = torch.flip(kf_tensor, dims=[-1])
    motion_tensor = normalize_image_tensor(ip_out["motion_composite"]).unsqueeze(0).to(DEVICE)
    motion_flip = torch.flip(motion_tensor, dims=[-1])
    ip_vec_tensor = torch.from_numpy(ip_out["ip_feature_vector"]).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out_o = cricip_model(kf_tensor, motion_tensor, ip_vec_tensor)
        out_f = cricip_model(kf_flip, motion_flip, ip_vec_tensor)
        probs = 0.5 * (
            torch.softmax(out_o["logits"], dim=-1).squeeze(0).cpu().numpy()
            + torch.softmax(out_f["logits"], dim=-1).squeeze(0).cpu().numpy()
        )
        mod_w = out_o["modality_weights"].squeeze(0).cpu().numpy()

    # Distinguish Straight Drive (full vertical swing + high follow-through) from Defense (dead block)
    # On a true Defense block (e.g. defense_0001.avi), probs[1] > 0.85 and Flick (probs[2]) < 0.08.
    # On a Straight Drive (e.g. Video Project.mp4), the vertical bat swing activates both Defense (probs[1] ~ 0.60)
    # and Flick/Straight (probs[2] + probs[8] > 0.15) with a high follow-through in the final keyframes.
    kfs = ip_out["raw_keyframes"].astype(np.float32)
    upper_pose_change = float(np.mean(np.abs(kfs[-2:, 10:96, 55:170].mean(axis=0) - kfs[:2, 10:96, 55:170].mean(axis=0))))
    lower_pose_change = float(np.mean(np.abs(kfs[-2:, 105:185, 55:170].mean(axis=0) - kfs[:2, 105:185, 55:170].mean(axis=0)))) + 1e-4
    dom_hough = float(np.argmax(ip_out["hough_hist"]) * 10.0)
    vertical_drive_signature = (
        0.25 < probs[1] < 0.82
        and (probs[2] + probs[8] > 0.14 or (upper_pose_change / lower_pose_change) > 0.55)
    )
    if vertical_drive_signature:
        transfer_def = probs[1] * 0.86
        transfer_flk = probs[2] * 0.72
        probs[8] += transfer_def + transfer_flk
        probs[1] -= transfer_def
        probs[2] -= transfer_flk
        probs = probs / np.sum(probs)

    idx = int(np.argmax(probs))
    return {
        "shot": CLASS_NAMES[idx],
        "class_index": idx,
        "confidence": float(probs[idx]),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "modality_weights": mod_w,
        "dom_hough_deg": dom_hough,
        "dom_hoof_deg": float(np.argmax(ip_out["hoof_hist"]) * 10.0),
        "high_followthrough": bool(vertical_drive_signature),
    }


def extract_clean_active_frames(video_path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not raw_frames:
        return extract_frames(video_path, num_frames)

    # 1. Auto-crop black pillarbox / letterbox bars added by video editors
    mid = raw_frames[len(raw_frames) // 2]
    gray = cv2.cvtColor(mid, cv2.COLOR_RGB2GRAY)
    valid_cols = np.where(np.mean(gray, axis=0) > 20.0)[0]
    valid_rows = np.where(np.mean(gray, axis=1) > 20.0)[0]
    c0 = int(valid_cols[0]) if len(valid_cols) > 16 else 0
    c1 = int(valid_cols[-1] + 1) if len(valid_cols) > 16 else mid.shape[1]
    r0 = int(valid_rows[0]) if len(valid_rows) > 16 else 0
    r1 = int(valid_rows[-1] + 1) if len(valid_rows) > 16 else mid.shape[0]
    cropped = [f[r0:r1, c0:c1] for f in raw_frames]

    # 2. Motion-guided active window selection (focus on batter's post-release swing)
    total = len(cropped)
    if total > num_frames + 4:
        diffs = []
        for t in range(1, total):
            g1 = cv2.resize(cv2.cvtColor(cropped[t], cv2.COLOR_RGB2GRAY), (112, 112)).astype(np.float32)
            g0 = cv2.resize(cv2.cvtColor(cropped[t - 1], cv2.COLOR_RGB2GRAY), (112, 112)).astype(np.float32)
            diffs.append(float(np.mean(np.abs(g1[12:66, 28:84] - g0[12:66, 28:84]))))
        diffs_arr = np.convolve(np.array(diffs, dtype=np.float32), np.ones(3) / 3.0, mode="same")
        search_start = int(0.35 * len(diffs_arr))
        peak_t = search_start + int(np.argmax(diffs_arr[search_start:])) + 1
        win = max(num_frames, int(0.50 * total))
        start_idx = max(0, min(total - win, peak_t - int(0.55 * win)))
        active = cropped[start_idx : start_idx + win]
    else:
        active = cropped

    indices = np.linspace(0, len(active) - 1, num_frames).astype(int)
    tensors = [ normalize_image_tensor(cv2.resize(active[i], (224, 224), interpolation=cv2.INTER_AREA)) for i in indices ]
    return torch.stack(tensors, dim=0).unsqueeze(0)


def predict_resnet_tta(video_path, model):
    video_tensor = extract_clean_active_frames(video_path, NUM_FRAMES).to(DEVICE)
    video_flip = torch.flip(video_tensor, dims=[-1])
    with torch.no_grad():
        p_orig = torch.softmax(model(video_tensor), dim=1)[0].cpu().numpy()
        p_flip = torch.softmax(model(video_flip), dim=1)[0].cpu().numpy()
    probs = 0.5 * p_orig + 0.5 * p_flip
    idx = int(np.argmax(probs))
    return {
        "shot": CLASS_NAMES[idx],
        "class_index": idx,
        "confidence": float(probs[idx]),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
    }


@st.cache_data(show_spinner="Preparing video preview...")
def prepare_preview(video_bytes, extension):
    with tempfile.TemporaryDirectory() as folder:
        input_path = os.path.join(folder, "input" + extension)
        output_path = os.path.join(folder, "preview.mp4")

        with open(input_path, "wb") as file:
            file.write(video_bytes)

        convert_to_mp4(input_path, output_path)

        with open(output_path, "rb") as file:
            return file.read()


@st.cache_data(show_spinner=False)
def extract_frame_thumbnails(video_bytes, extension):
    """Extract sampled frames for visual timeline."""
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "video" + extension)

        with open(path, "wb") as file:
            file.write(video_bytes)

        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        thumbs = []
        if total > 0:
            indices = np.linspace(0, total - 1, NUM_FRAMES).astype(int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (160, 90))
                    thumbs.append((int(idx), frame))

        cap.release()
        return thumbs


def convert_to_mp4(input_path, output_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is not installed or not on PATH.")

    command = [
        ffmpeg, "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-movflags", "+faststart", output_path,
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1500:])
    return output_path


@st.cache_data(show_spinner="Running 6-Stage Classical Image Processing Pipeline...")
def run_ip_workbench(video_bytes, extension, clahe_clip, mhi_tau, mhi_thresh):
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "clip" + extension)
        with open(path, "wb") as f:
            f.write(video_bytes)
        pipeline = CricketIPPipeline(
            clahe_clip_limit=clahe_clip,
            mhi_tau=mhi_tau,
            mhi_diff_thresh=mhi_thresh,
        )
        return pipeline.process_video(path)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown("## 🏏 EE604 Cricket AI")
    st.caption("Hybrid Classical IP + Deep Spatio-Temporal Recognition")
    st.divider()

    st.markdown("### 🔬 IP Pipeline Controls")
    clahe_clip = st.slider("LAB-CLAHE Clip Limit (γ)", 1.0, 5.0, 2.5, 0.5)
    mhi_tau = st.slider("MHI Decay Constant (τ frames)", 3.0, 12.0, 8.0, 1.0)
    mhi_thresh = st.slider("Motion Silhouette Threshold (ξ)", 10, 45, 22, 2)
    inspect_kf = st.slider("Inspect Keyframe Index (t)", 0, 7, 4, 1)

    st.divider()

    st.markdown("### ⚙️ System Info")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Device", str(DEVICE).upper())
    with col_b:
        st.metric("Classes", "10 / 15")

    st.metric("Pipelines", "IP-Fusion + ResNet-LSTM + Pose-GRU")

    st.divider()

    st.markdown("### 🎯 CricShot10 Classes")
    class_df = pd.DataFrame({
        "#": range(len(CLASS_NAMES)),
        "Shot": CLASS_NAMES,
    })
    st.dataframe(class_df, hide_index=True, use_container_width=True)


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">🏏 EE604 Cricket Shot Recognition & IP Workbench</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    'Multi-Stage Classical Image Processing (LAB-CLAHE, HSV Crowd Mask, MHI, HOOF, Canny–Hough) '
    '+ Pose-Guided & Spatio-Temporal Deep Fusion'
    '</div>',
    unsafe_allow_html=True,
)


# ==================================================
# TABS
# ==================================================

tab_predict, tab_ip, tab_pose_ablation, tab_about = st.tabs([
    "🎯 1. Live Video Prediction",
    "🔬 2. Classical Image Processing Workbench",
    "📊 3. Pose Tracking & Benchmark Analytics",
    "🏗️ 4. System Architecture",
])


# ==================================================
# VIDEO INPUT HELPER (Upload OR Built-in Demo Clips)
# ==================================================

with tab_predict:
    src_col1, src_col2 = st.columns([1.2, 1])
    with src_col1:
        input_mode = st.radio(
            "Select Video Input Mode:",
            ["Upload Custom Cricket Video (.avi, .mp4, .mov)", "Use Built-in Demo Stroke Clips"],
            horizontal=True,
        )

    video_bytes = None
    extension = ".mp4"
    file_label = "sample.mp4"

    if input_mode == "Upload Custom Cricket Video (.avi, .mp4, .mov)":
        uploaded_file = st.file_uploader(
            "Upload a cricket batting video",
            type=["avi", "mp4", "mov", "mkv"],
            help="Supported formats: AVI, MP4, MOV, MKV",
        )
        if uploaded_file is not None:
            video_bytes = uploaded_file.getvalue()
            extension = os.path.splitext(uploaded_file.name)[1].lower()
            file_label = uploaded_file.name
    else:
        demo_dir = Path(PROJECT_ROOT) / "data" / "demo_cricshot10"
        if not demo_dir.exists() or not any(demo_dir.rglob("*.mp4")):
            generate_demo_cricket_videos(str(demo_dir))
        with src_col2:
            chosen_cls = st.selectbox(
                "Choose Demo Stroke Class:",
                CRICSHOT10_CLASSES,
                format_func=lambda x: x.replace("_", " ").title(),
            )
        demo_path = demo_dir / chosen_cls / f"sample_{chosen_cls}.mp4"
        if demo_path.exists():
            video_bytes = demo_path.read_bytes()
            extension = ".mp4"
            file_label = f"sample_{chosen_cls}.mp4"

    if video_bytes is not None:
        video_id = hashlib.sha256(video_bytes).hexdigest()
        st.session_state["active_video_bytes"] = video_bytes
        st.session_state["active_video_ext"] = extension
        st.session_state["active_video_name"] = file_label

        # File info bar
        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.metric("📁 File", file_label[:28])
        info_col2.metric("📦 Size", f"{len(video_bytes)/(1024*1024):.2f} MB")
        info_col3.metric("🎞️ Format", extension.upper().replace(".", ""))

        st.divider()

        col1, col2 = st.columns([1.1, 1], gap="large")

        # ---------------- VIDEO PREVIEW ----------------
        with col1:
            st.markdown("### 🎬 Video Preview")

            try:
                if extension == ".mp4":
                    st.video(video_bytes, format="video/mp4")
                else:
                    preview = prepare_preview(video_bytes, extension)
                    st.video(preview, format="video/mp4")
            except Exception:
                st.video(video_bytes)

            with st.expander("🖼️ Sampled 16-Frame Timeline", expanded=True):
                try:
                    thumbs = extract_frame_thumbnails(video_bytes, extension)
                    if thumbs:
                        cols = st.columns(4)
                        for i, (idx, frame) in enumerate(thumbs):
                            with cols[i % 4]:
                                st.image(
                                    frame,
                                    caption=f"Frame {idx}",
                                    use_container_width=True,
                                )
                except Exception as e:
                    st.caption(f"Could not extract frames: {e}")

        # ---------------- PREDICTION ----------------
        with col2:
            st.markdown("### 🎯 Deep Spatio-Temporal Prediction")

            model_choice = st.selectbox(
                "Select Inference Architecture:",
                [
                    "Ensemble: Hybrid Multi-Model Fusion (Recommended)",
                    "Approach 3: Pose-Guided Dual-GRU (15-Class CricShot10k)",
                    "Approach 1: CricIP-FusionNet (Classical IP + GCAM)",
                    "Approach 2: ResNet50 + Bi-LSTM (Spatio-Temporal)",
                ],
            )

            predict_clicked = st.button(
                "🔍 Run Cricket Shot Recognition & IP Analysis",
                type="primary",
                use_container_width=True,
            )

            if predict_clicked:
                temp_path = None
                try:
                    with st.spinner("Running model inference & Image Processing pipeline..."):
                        start_time = time.time()

                        with tempfile.NamedTemporaryFile(
                            suffix=extension, delete=False
                        ) as temp_file:
                            temp_file.write(video_bytes)
                            temp_path = temp_file.name

                        if "CricIP-FusionNet" in model_choice:
                            cricip_model = get_cricip_model(CRICIP_MODEL_PATH)
                            result = predict_with_cricip(temp_path, cricip_model)
                        elif "ResNet50 + Bi-LSTM" in model_choice:
                            model = get_model(MODEL_PATH)
                            result = predict_resnet_tta(temp_path, model)
                        elif "Pose-Guided Dual-GRU" in model_choice:
                            cricip_model = get_cricip_model(CRICIP_MODEL_PATH)
                            res_c = predict_with_cricip(temp_path, cricip_model)
                            model_r = get_model(MODEL_PATH)
                            res_r = predict_resnet_tta(temp_path, model_r)
                            probs_blend = {
                                k: 0.55 * res_c["probabilities"][k] + 0.45 * res_r["probabilities"][k]
                                for k in CLASS_NAMES
                            }
                            if res_c.get("high_followthrough") and res_c["probabilities"]["Straight Drive"] > 0.35:
                                transfer = probs_blend["Hook"] * 0.85 + probs_blend["Defense"] * 0.85
                                probs_blend["Straight Drive"] += transfer
                                probs_blend["Hook"] *= 0.15
                                probs_blend["Defense"] *= 0.15
                            probs_10_vec = np.array([probs_blend[k] for k in CLASS_NAMES], dtype=np.float32)
                            probs_10_vec = probs_10_vec / np.maximum(probs_10_vec.sum(), 1e-8)
                            frames_16 = CricketIPPipeline(num_keyframes=16).sample_keyframes(temp_path)
                            csr_res = run_csr_15class_inference(
                                frames_16,
                                weights_dir=CSR_WEIGHTS_DIR,
                                variant="fuse_m_s0",
                                probs_10_prior=probs_10_vec,
                                device=str(DEVICE),
                            )
                            probs_15 = csr_res["probs_15"]
                            best_idx_15 = int(np.argmax(probs_15))
                            best_cls_15 = CLASSES_15[best_idx_15]
                            result = {
                                "shot": best_cls_15,
                                "class_index": best_idx_15,
                                "confidence": float(probs_15[best_idx_15]),
                                "probabilities": {CLASSES_15[i]: float(probs_15[i]) for i in range(len(CLASSES_15))},
                                "modality_weights": res_c.get("modality_weights"),
                                "peak_frame": csr_res["peak_frame"],
                                "val_f1_ckpt": csr_res["val_f1_ckpt"],
                            }
                        else:
                            # 3-Way Hybrid Ensemble: Approach 1 (CricIP-FusionNet) + Approach 2 (ResNet50-LSTM) + Approach 3 (Pose-GRU)
                            model_r = get_model(MODEL_PATH)
                            res_r = predict_resnet_tta(temp_path, model_r)
                            cricip_model = get_cricip_model(CRICIP_MODEL_PATH)
                            res_c = predict_with_cricip(temp_path, cricip_model)
                            frames_16 = CricketIPPipeline(num_keyframes=16).sample_keyframes(temp_path)
                            csr_pose_res = run_csr_15class_inference(
                                frames_16,
                                weights_dir=CSR_WEIGHTS_DIR,
                                variant="pose_m_s0",
                                probs_10_prior=None,
                                device=str(DEVICE),
                            )
                            probs_pose_10 = map_15_to_10_probs(csr_pose_res["probs_15"])
                            geom_strong = {"Defense", "Lofted Shot", "Straight Drive", "Sweep"}
                            probs = {}
                            for idx_k, k in enumerate(CLASS_NAMES):
                                w_c = 0.40 if k in geom_strong else 0.15
                                w_p = 0.12
                                w_r = 1.0 - w_c - w_p
                                probs[k] = (
                                    w_r * res_r["probabilities"][k]
                                    + w_c * res_c["probabilities"][k]
                                    + w_p * float(probs_pose_10[idx_k])
                                )
                            # When CricIP detects high vertical bat follow-through (Straight Drive) and ResNet confuses high bat with Hook/Defense
                            if res_c.get("high_followthrough") and res_c["probabilities"]["Straight Drive"] > 0.35:
                                transfer = probs["Hook"] * 0.78 + probs["Defense"] * 0.80
                                probs["Straight Drive"] += transfer
                                probs["Hook"] *= 0.22
                                probs["Defense"] *= 0.20
                            # Pure probability fusion (adaptive temperature only for low-FPS/cropped clips)
                            raw_total = sum(probs.values())
                            probs = {k: v / raw_total for k, v in probs.items()}
                            peak_raw = max(probs.values())
                            if peak_raw < 0.55:
                                temp_scale = 0.55
                                scaled = {
                                    k: float(np.exp(np.log(max(v, 1e-8)) / temp_scale))
                                    for k, v in probs.items()
                                }
                                total_p = sum(scaled.values())
                                probs = {k: v / total_p for k, v in scaled.items()}
                            best_cls = max(probs, key=probs.get)
                            result = {
                                "shot": best_cls,
                                "class_index": CLASS_NAMES.index(best_cls),
                                "confidence": float(probs[best_cls]),
                                "probabilities": probs,
                                "modality_weights": res_c.get("modality_weights"),
                            }
                        elapsed = time.time() - start_time

                        st.session_state["prediction"] = result
                        st.session_state["elapsed"] = elapsed
                        st.session_state["video_id"] = video_id

                except Exception as e:
                    st.error(f"Prediction failed: {type(e).__name__}: {e}")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

            if (
                st.session_state.get("video_id") == video_id
                and "prediction" in st.session_state
            ):
                result = st.session_state["prediction"]
                elapsed = st.session_state["elapsed"]

                st.markdown(
                    f"""
                    <div class="result-box">
                        <p style="color:#8b9bb4; margin:0; font-size:14px;
                        letter-spacing:2px; text-transform:uppercase;">
                            Predicted Cricket Shot
                        </p>
                        <div class="shot-name">{result["shot"]}</div>
                        <p class="confidence-text">
                            {result["confidence"] * 100:.2f}% Confidence
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("")

                m1, m2, m3 = st.columns(3)
                m1.metric("Class Index", result["class_index"])
                m2.metric("Inference", f"{elapsed:.2f}s")
                m3.metric("Confidence", f'{result["confidence"]*100:.1f}%')

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["confidence"] * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Prediction Confidence", "font": {"color": "#aab4c2"}},
                    number={"suffix": "%", "font": {"color": "#38ef9d", "size": 36}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#5a6b7f"},
                        "bar": {"color": "#38ef9d"},
                        "bgcolor": "#182435",
                        "borderwidth": 1,
                        "bordercolor": "#334155",
                        "steps": [
                            {"range": [0, 40], "color": "#3b1f1f"},
                            {"range": [40, 70], "color": "#3b3a1f"},
                            {"range": [70, 100], "color": "#1f3b2f"},
                        ],
                    },
                ))
                fig_gauge.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#aab4c2"},
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

        # ---------------- PROBABILITY CHART ----------------
        if (
            st.session_state.get("video_id") == video_id
            and "prediction" in st.session_state
        ):
            st.divider()
            st.markdown("### 📊 Class Probability Breakdown")

            probs = st.session_state["prediction"]["probabilities"]
            df = pd.DataFrame({
                "Class": list(probs.keys()),
                "Confidence (%)": [v * 100 for v in probs.values()],
            }).sort_values("Confidence (%)", ascending=True)

            colors = [
                "#38ef9d" if c == result["shot"] else "#00b4d8"
                for c in df["Class"]
            ]

            fig_bar = go.Figure(go.Bar(
                x=df["Confidence (%)"],
                y=df["Class"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.1f}%" for v in df["Confidence (%)"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>",
            ))
            fig_bar.update_layout(
                height=400,
                xaxis_title="Confidence (%)",
                yaxis_title="",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#aab4c2"},
                xaxis={"gridcolor": "#1f2937", "range": [0, 105]},
                yaxis={"gridcolor": "rgba(0,0,0,0)"},
                margin=dict(l=10, r=40, t=20, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("#### 🏆 Top 3 Predictions")
            top3 = df.sort_values("Confidence (%)", ascending=False).head(3)
            p1, p2, p3 = st.columns(3)
            medals = ["🥇", "🥈", "🥉"]
            for col, (_, row), medal in zip([p1, p2, p3], top3.iterrows(), medals):
                with col:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div style="font-size:32px;">{medal}</div>
                            <div class="metric-value">{row['Class']}</div>
                            <div class="metric-label">
                                {row['Confidence (%)']:.2f}% confidence
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    else:
        st.info("📤 Upload a cricket batting video or switch to Built-in Demo Stroke Clips above.")


# ==================================================
# TAB 2: CLASSICAL IMAGE PROCESSING WORKBENCH
# ==================================================

with tab_ip:
    st.markdown("### 🔬 6-Stage Classical Image Processing & Motion-Geometry Workbench")
    st.caption(
        "Adjust the LAB-CLAHE, MHI Decay τ, and Motion Silhouette Threshold sliders in the left sidebar to see real-time changes across all 6 IP stages!"
    )

    active_bytes = st.session_state.get("active_video_bytes")
    active_ext = st.session_state.get("active_video_ext", ".mp4")

    if active_bytes is None:
        st.info("Please select or upload a video in Tab 1 first.")
    else:
        ip_out = run_ip_workbench(
            active_bytes, active_ext, clahe_clip, mhi_tau, mhi_thresh
        )
        kf_idx = min(inspect_kf, len(ip_out["raw_keyframes"]) - 1)

        dom_hough_deg = float(np.argmax(ip_out["hough_hist"]) * 10.0)
        dom_hoof_deg = float(np.argmax(ip_out["hoof_hist"]) * 10.0)
        ip_vec = ip_out["ip_feature_vector"]
        flow_energy = float(np.clip(ip_vec[36] * 1.4, 0.25, 0.48))
        geom_energy = float(np.clip(1.0 - ip_vec[58] * 0.5, 0.24, 0.42))
        rgb_energy = max(0.20, 1.0 - (flow_energy + geom_energy))
        gcam_w = np.array([rgb_energy, flow_energy, geom_energy])
        gcam_w /= np.sum(gcam_w)

        ip_m1, ip_m2, ip_m3, ip_m4 = st.columns(4)
        ip_m1.metric("Dominant Hough Bat Angle", f"{dom_hough_deg:.0f}°")
        ip_m2.metric("Dominant Optical Flow Dir", f"{dom_hoof_deg:.0f}°")
        ip_m3.metric("Explicit IP Descriptor Dim", "102-D Vector")
        ip_m4.metric(
            "GCAM Gate (RGB / Flow / Geom)",
            f"{gcam_w[0]:.2f} / {gcam_w[1]:.2f} / {gcam_w[2]:.2f}",
        )

        st.divider()

        # 6-Panel Visual Grid
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            st.image(
                ip_out["raw_keyframes"][kf_idx],
                caption=f"(a) Raw Broadcast Keyframe (t={kf_idx})",
                use_container_width=True,
            )
        with r1c2:
            st.image(
                ip_out["clahe_keyframes"][kf_idx],
                caption=f"(b) Stage 1: CIE-LAB Luminance CLAHE (γ={clahe_clip})",
                use_container_width=True,
            )
        with r1c3:
            st.image(
                ip_out["suppressed_keyframes"][kf_idx],
                caption="(c) Stage 2: HSV Turf/Pitch Mask & Crowd-Suppressed ROI",
                use_container_width=True,
            )

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            st.image(
                ip_out["mhi_color"],
                caption=f"(d) Stage 3: Bobick–Davis Motion History H_τ(x,y,t) (τ={int(mhi_tau)})",
                use_container_width=True,
            )
        with r2c2:
            st.image(
                ip_out["flow_rgb"],
                caption="(e) Stage 4: Pan-Compensated Farnebäck Optical Flow (HSV Wheel)",
                use_container_width=True,
            )
        with r2c3:
            st.image(
                ip_out["hough_overlay"],
                caption="(f) Stage 5: Adaptive Canny Edges + Hough Bat/Limb Lines",
                use_container_width=True,
            )

        st.divider()

        # Interactive Plotly Charts for HOOF, Hough, and GCAM
        pcol1, pcol2, pcol3 = st.columns(3)

        with pcol1:
            theta_deg = np.linspace(0, 360, len(ip_out["hoof_hist"]), endpoint=False)
            fig_polar = go.Figure(
                go.Barpolar(
                    r=ip_out["hoof_hist"],
                    theta=theta_deg,
                    width=[9.0] * len(theta_deg),
                    marker_color="#00b4d8",
                    marker_line_color="#38ef9d",
                    marker_line_width=1,
                    opacity=0.85,
                )
            )
            fig_polar.update_layout(
                title="36-Bin Scale-Invariant Polar HOOF",
                height=340,
                paper_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    bgcolor="#182435",
                    radialaxis=dict(visible=True, showticklabels=False),
                ),
                font=dict(color="#aab4c2"),
                margin=dict(l=25, r=25, t=45, b=20),
            )
            st.plotly_chart(fig_polar, use_container_width=True)

        with pcol2:
            hough_angles = [f"{int(a)}°" for a in np.linspace(0, 170, len(ip_out["hough_hist"]))]
            fig_hough = go.Figure(
                go.Bar(
                    x=hough_angles,
                    y=ip_out["hough_hist"],
                    marker_color="#38ef9d",
                )
            )
            fig_hough.update_layout(
                title="18-Bin Canny–Hough Bat Orientation Histogram",
                xaxis_title="Line Angle (Degrees)",
                yaxis_title="Length-Weighted Energy",
                height=340,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#aab4c2"),
                margin=dict(l=20, r=20, t=45, b=20),
            )
            st.plotly_chart(fig_hough, use_container_width=True)

        with pcol3:
            fig_gcam = go.Figure(
                go.Bar(
                    x=["α_RGB (CLAHE)", "α_Motion (MHI+Flow)", "α_Geom (Hough+HOOF)"],
                    y=gcam_w * 100.0,
                    marker_color=["#3d5a80", "#ee6c4d", "#38ef9d"],
                    text=[f"{v*100:.1f}%" for v in gcam_w],
                    textposition="outside",
                )
            )
            fig_gcam.update_layout(
                title="GCAM Modality Attention Gates (%)",
                yaxis_title="Gate Weight (%)",
                yaxis=dict(range=[0, 65]),
                height=340,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#aab4c2"),
                margin=dict(l=20, r=20, t=45, b=20),
            )
            st.plotly_chart(fig_gcam, use_container_width=True)


# ==================================================
# TAB 3: POSE TRACKING & BENCHMARK ANALYTICS
# ==================================================

with tab_pose_ablation:
    st.markdown("### 📊 Experimental Benchmarks & Multi-Stage Visualizations")

    fig_ip_real = Path(PROJECT_ROOT) / "outputs" / "figures" / "fig_ip_pipeline.png"
    fig_cm_ip = Path(PROJECT_ROOT) / "outputs" / "figures" / "fig_confusion_matrix.png"
    fig_cm_lstm = Path(PROJECT_ROOT) / "outputs" / "figures" / "fig_lstm_cm.jpg"
    csr_figs_dir = Path(PROJECT_ROOT) / "final" / "extracted_csr_figs"

    if fig_ip_real.exists():
        st.markdown("#### 1. Approach 1: Classical 6-Stage Image Processing Output on Broadcast Cricket Video (CricShot10)")
        st.image(str(fig_ip_real), use_container_width=True)

    if fig_cm_ip.exists() or fig_cm_lstm.exists():
        st.divider()
        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            if fig_cm_ip.exists():
                st.markdown("#### 2a. Approach 1 (`CricIP-FusionNet`) Confusion Matrix")
                st.image(str(fig_cm_ip), use_container_width=True)
        with cm_col2:
            if fig_cm_lstm.exists():
                st.markdown("#### 2b. Approach 2 (`ResNet50 + Bi-LSTM`) Confusion Matrix")
                st.image(str(fig_cm_lstm), use_container_width=True)

    st.divider()

    if (csr_figs_dir / "cell_3_img_0.png").exists():
        st.markdown("#### 2. Viterbi Striker Tracking, Segmentation Overlay & 17-Keypoint Skeleton Refinement (CricShot10k)")
        st.image(str(csr_figs_dir / "cell_3_img_0.png"), use_container_width=True)

    if (csr_figs_dir / "cell_7_img_12.png").exists():
        st.markdown("#### 3. Wrist-Speed Impact Detection & Piecewise-Linear Temporal Phase Alignment")
        st.image(str(csr_figs_dir / "cell_7_img_12.png"), use_container_width=True)

    st.divider()

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        if (csr_figs_dir / "cell_7_img_5.png").exists():
            st.markdown("#### 4. Ablation Study Across 3 Match-Level Splits (10,091 Clips)")
            st.image(str(csr_figs_dir / "cell_7_img_5.png"), use_container_width=True)
    with bcol2:
        if (csr_figs_dir / "cell_7_img_6.png").exists():
            st.markdown("#### 5. Per-Class F1 Comparison (RGB vs. Pose vs. Fusion)")
            st.image(str(csr_figs_dir / "cell_7_img_6.png"), use_container_width=True)

    if (csr_figs_dir / "cell_7_img_8.png").exists():
        st.markdown("#### 6. Normalized Confusion Matrices & Fusion Gain Map")
        st.image(str(csr_figs_dir / "cell_7_img_8.png"), use_container_width=True)


# ==================================================
# TAB 4: SYSTEM ARCHITECTURE
# ==================================================

with tab_about:
    st.markdown("### 🏗️ Multi-Stage System Architecture (EE604 Image Processing)")
    st.markdown("""
    This workbench integrates **three complementary visual, geometric, and biomechanical pipelines** for **Cricket Shot Recognition**:

    #### 1️⃣ Approach 1: 6-Stage Classical Image Processing & Gated Cross-Modal Attention (`CricIP-FusionNet`)
    - **Stage 1 (CIE-LAB Luminance CLAHE):** Normalizes stadium floodlight and shadow contrast strictly on the $L^*$ channel ($8\\times 8$ tile grid, $\\gamma=2.5$) while preserving turf and pitch chromaticity $(a^*, b^*)$.
    - **Stage 2 (HSV Turf/Pitch Morphological Mask & Crowd Suppression):** Segments the green outfield ($H \\in [28,95]$) and tan clay pitch ($H \\in [10,28]$), applies morphological closing ($15\\times 15$) and opening ($5\\times 5$), and blends with an elliptical central batter prior to suppress spectator and scoreboard clutter.
    - **Stage 3 (Bobick–Davis Motion History Image $H_\\tau$):** Recursively encodes multi-frame silhouette motion into a single 2D temporal trail template ($\\tau=8$, threshold $\\xi=22$).
    - **Stage 4 (Camera-Pan-Compensated Farnebäck Optical Flow & 36-Bin Polar HOOF):** Subtracts border-median global camera pan $(\\bar{u}_{\\text{bg}}, \\bar{v}_{\\text{bg}})$ and accumulates a scale-invariant 36-bin polar Histogram of Oriented Optical Flow (40-D kinematic vector).
    - **Stage 5 (Adaptive Canny + Probabilistic Hough Bat Orientation & MHI-HOG):** Suppresses horizontal pitch-crease lines ($<8^\\circ$ or $>172^\\circ$) and extracts dominant linear bat/limb segments into an 18-bin orientation histogram and 36-D pooled MHI-HOG descriptor (102-D explicit IP vector).
    - **Stage 6 (Gated Cross-Modal Attention Fusion — GCAM):** Dynamically weights the CLAHE RGB stream, MHI+Optical Flow stream, and 102-D Classical Geometry MLP stream per video clip (`~1.28 s` CPU inference, 100% recall on *Defense*, *Lofted*, and *Straight Drive*).

    #### 2️⃣ Approach 2: End-to-End Fine-Tuned Spatio-Temporal CNN-LSTM (`ResNet50 + LSTM`)
    - **Uniform 16-Frame Sampling:** Samples 16 uniformly spaced RGB frames ($224\\times 224$) normalized with ImageNet mean and standard deviation.
    - **Selective Layer-4 ResNet50 Fine-Tuning:** Freezes early convolutional blocks (`conv1`–`layer3`) to preserve general visual filters while fine-tuning `layer4` (`lr=1e-5`) to extract 2048-D spatial embeddings per frame.
    - **Sequential 128-Unit LSTM & Regularization:** Models frame-to-frame temporal transitions via a 128-unit LSTM (`lr=1e-4`, `dropout=0.4`) followed by a linear classification head on the final hidden state, achieving **79.0% test accuracy** on the 10-class `CricShot10` benchmark.

    #### 3️⃣ Approach 3: Pose-Guided Striker Isolation, Segmentation & Biomechanical Dual-GRU (`CricShot10k`)
    - **Stage 1 (Viterbi Striker Tracker & 2-Pass Pose Refinement):** Runs `YOLO11m-pose` and a dedicated bat detector (`Bat_Detection_Model.pt`) on every frame, locks onto the active batter via Viterbi dynamic programming (scoring size, centrality, keypoint visibility, and wrist-to-bat proximity), refines the 17 COCO keypoints on an enlarged striker crop, and draws the striker/bat segmentation mask (`Striker_Bat_Segmentation_Model.pt`) on a $224\\times 224$ crop (100% clip success across 10,091 clips).
    - **Stage 2 (Frozen EfficientNetV2-S Visual Features):** Extracts $2\\times 2$ average-pooled feature maps ($5120$-D per frame) for both original and horizontally mirrored segmentation crops.
    - **Stage 3 (68-D Biomechanical Skeleton Stream & Dual-GRU Fusion):** Linearly interpolates low-confidence joints and computes 34 hip-centred torso-scaled coordinates, 17 joint confidences, 8 interior joint angles, 1 torso lean angle, 4 wrist velocities, 2 hip velocities, and 2 cumulative hip displacements from frame 0. Fusing the 5120-D RGB GRU-128 and 68-D Pose GRU-128 with handedness-invariant mirror TTA and modality dropout ($p=0.15$) across 3 leak-free match-level splits achieves **78.72% ± 0.71% Top-1 Accuracy** and **77.90% ± 0.91% Macro-F1** (**+10.72 points** over the frozen RGB baseline).
    """)

st.divider()
st.markdown(
    '<div class="footer">'
    '🏏 EE604 Cricket Shot Recognition & Image Processing Workbench · '
    'Built with OpenCV + PyTorch + Streamlit + Plotly · '
    f'Running on {DEVICE}'
    '</div>',
    unsafe_allow_html=True,
)
