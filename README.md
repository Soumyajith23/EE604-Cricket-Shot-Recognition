# 🏏 Cricket Shot Recognition from Broadcast Video (EE604 Image Processing Course Project)

**Authors (Indian Institute of Technology Kanpur — EE604, 2026):**
- **Nonit Gupta** (230712)
- **Om Jee Singh** (230718)
- **Bipin Kumar Jaiswal** (230300)
- **Nishant Singh** (230706)
- **Soumyajit Halder** (231028)

---

## 📌 Project Overview

Recognizing fine-grained cricket batting shots from television broadcast videos is challenging due to harsh stadium floodlights/shadows, animated crowd banners and scoreboards, camera panning, multiple players on the pitch (bowler, umpire, wicketkeeper, non-striker), and left/right batter handedness.

In this project, we systematically design, evaluate, and unify **three progressive approaches** along with a **3-Way Hybrid Multi-Model Ensemble** deployed in an interactive **4-Tab Streamlit Workbench (`cricket_dashboard/app.py`)**:

1. **Approach 1 — 6-Stage Classical Image Processing + Gated Cross-Modal Attention (`CricIP-FusionNet`, `src/ip_pipeline.py` & `src/model.py`):**
   - **Stage 1:** CIE-LAB Luminance CLAHE ($\gamma=2.5$, $8\times8$ tiles) for shadow/floodlight normalization without hue distortion.
   - **Stage 2:** HSV Turf ($H\in[28,95]$) & Pitch ($H\in[10,28]$) Morphological Segmentation + Gaussian Batter ROI Prior for spectator/scoreboard suppression.
   - **Stage 3:** Bobick–Davis Recursive Motion History Image ($H_\tau(x,y,t)$, $\tau=8$, $\xi=22$).
   - **Stage 4:** Border-Median Camera-Pan-Compensated Farnebäck Dense Optical Flow & 36-Bin Scale-Invariant Polar HOOF (40-D vector).
   - **Stage 5:** Adaptive Canny Edge + Probabilistic Hough Bat Angle Histogram (18-Bin, suppressing horizontal pitch creases $<8^\circ$ or $>172^\circ$) + 36-D MHI-HOG (102-D explicit IP descriptor).
   - **Stage 6:** Gated Cross-Modal Attention Module (GCAM) dynamically fusing RGB, MHI+Flow, and 102-D Classical Geometry streams (`~1.28 s` CPU inference; **100% recall** on *Defense*, *Lofted*, and *Straight Drive*).

2. **Approach 2 — End-to-End Spatio-Temporal `ResNet50 + LSTM` (`cricket_dashboard/model.py` & `predict.py`):**
   - Uniformly samples $T=16$ RGB frames ($224\times224$), freezes `conv1`–`layer3` of an ImageNet-pretrained ResNet50 while fine-tuning `layer4` ($\text{lr}=10^{-5}$), and models sequential transitions via a 128-unit LSTM (`dropout=0.4`, $\text{lr}=10^{-4}$), achieving **79.0% test accuracy** on the 10-class `CricShot10` benchmark (284 test clips).

3. **Approach 3 — Pose-Guided Viterbi Striker Isolation & Biomechanical Dual-GRU Fusion (`src/csr_infer.py` & `CricShot10k_Pose_DualGRU_Experiments.ipynb`):**
   - **Stage 1 (`cricpre.py`):** Runs `YOLO11m-pose` + a dedicated bat detector on every frame and solves a **Viterbi dynamic programming search** to isolate the striker from bowlers, umpires, and non-strikers (99.8% clip tracking success across 10,091 videos), followed by a 2-pass high-resolution pose refinement and striker$\cup$bat segmentation overlay.
   - **Stage 2 (`cricfeat.py`):** Extracts frozen `EfficientNetV2-S` $2\times2$-pooled features ($5120$-D per frame) for original and horizontally mirrored crops.
   - **Stage 3 (`cricfuse.py`):** Extracts a **68-D torso-normalized biomechanical skeleton descriptor** (34 hip-centered coordinates, 17 confidences, 8 interior joint angles, 1 torso lean angle, 4 wrist velocities, 2 hip velocities, 2 cumulative hip displacements) and fuses a Pose GRU-128 with an RGB GRU-128 under handedness-invariant mirror augmentation and modality dropout ($p=0.15$). Across 3 leak-free match-level splits on the 15-class **CricShot10k** benchmark (10,091 clips), the 68-D skeleton stream raises Macro-F1 from **67.18%** (frozen RGB baseline) to **75.93%** alone (**+8.75 pts**) and **77.90%** (**78.72% accuracy**, **+10.72 pts** paired gain) when fused.

---

## 🔗 Datasets, Pretrained Weights & External Repositories Used

- **CricShot10 Dataset (1,894 clips, 10 classes):** [Google Drive Folder](https://drive.google.com/drive/folders/1DPHURwQk5R8blgjM8VNz6Q68LqckxljX) | Original Baseline Repo: [engaak/cricshot10](https://github.com/engaak/cricshot10)
- **CricShot10k Dataset, Detection Weights & Shards (10,091 clips, 15 classes):** [Kaggle Dataset (`nonitgupta/crickshot10k`)](https://www.kaggle.com/datasets/nonitgupta/crickshot10k)
- **Pose & Backbone Frameworks Used:**
  - Ultralytics YOLO11-Pose & Segmentation: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
  - PyTorch Torchvision (`ResNet18`, `ResNet50`, `EfficientNetV2-S`): [https://github.com/pytorch/vision](https://github.com/pytorch/vision)

---

## 📂 Repository Structure

```text
├── cricket_dashboard/
│   ├── app.py                                      # Unified 4-Tab Streamlit Dashboard (Ensemble + Approaches 1, 2, 3)
│   ├── model.py                                    # Approach 2: ResNet50 + 128-Unit LSTM Architecture
│   ├── predict.py                                  # Approach 2: Video Frame Extraction & Inference Engine
│   └── requirements.txt                            # Dashboard dependencies
├── src/
│   ├── ip_pipeline.py                              # Approach 1: 6-Stage Classical Image Processing Pipeline (102-D)
│   ├── model.py                                    # Approach 1: CricIP-FusionNet & Gated Cross-Modal Attention (GCAM)
│   ├── csr_infer.py                                # Approach 3: 68-D Biomechanical Pose + Dual-GRU Inference & 15->10 Mapping
│   ├── dataset.py                                  # CricShot10 Dataset Loader & Synthetic Stroke Generator
│   ├── nonit_cricpre.py                            # Approach 3 Stage 1: Viterbi Striker Tracker & 2-Pass YOLO11m-Pose
│   └── nonit_cell6.py                              # Approach 3 Stage 3: 3-Split Match-Level Dual-GRU Training & Ablation
├── weights/weights/
│   └── pose_m_s0.pt                                # Approach 3 Trained 68-D Pose GRU Checkpoint
├── outputs/figures/                                # Confusion Matrices, IP Pipeline Figures & Ablation Plots
├── CVPR_2026_Report_LaTeX/                         # Complete 28-Page CVPR 2026 LaTeX Report Source & Figures
├── CricShot10_Training_and_Experiments.ipynb       # Colab Notebook for Approach 1 (CricIP-FusionNet Ablations)
├── CricShot10k_Pose_DualGRU_Experiments.ipynb      # Notebook for Approach 3 (10,091-Clip CricShot10k Experiments)
└── train_eval.py                                   # CLI Training & Evaluation Script for CricShot10
```

---

## 🚀 How to Run the Interactive 4-Tab Dashboard

```bash
pip install -r requirements.txt
streamlit run cricket_dashboard/app.py
```
