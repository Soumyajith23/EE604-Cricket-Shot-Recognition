"""
src/model.py
============
Neural Architectures for Cricket Shot Recognition (EE604 Course Project).

Implements:
1. ClassicalIPMLP (Baseline 1):
   Operates purely on the 102-D handcrafted Image Processing descriptor
   (36-bin HOOF + 4 kinematics + 18-bin Canny-Hough bat orientation + 2 angular stats
   + 36-D MHI-HOG + 6-D HSV pitch statistics).

2. RGBBaselineNet (Baseline 2):
   Standard ResNet-18 frame encoder with temporal average/attention pooling over
   sampled video keyframes (without explicit optical flow or Hough geometry streams).

3. CricIPFusionNet (Proposed Novel Architecture):
   A Tri-Stream Hybrid Image Processing & Deep Spatio-Temporal Network that integrates:
     - Stream 1 (Appearance Stream): Temporal-Attention ResNet-18 over CLAHE-normalized,
       crowd-suppressed RGB keyframes.
     - Stream 2 (Motion Template Stream): Convolutional encoder over the 3-channel
       Bobick-Davis Motion History Image (MHI) + Dense Farneback Optical Flow composite.
     - Stream 3 (Explicit IP Geometric Stream): Residual MLP over the 102-D scale-invariant
       HOOF, Canny-Hough bat orientation, and MHI-HOG vector.
   Fused via a novel Gated Cross-Modal Attention Module (GCAM) that dynamically predicts
   per-clip modality weights (alpha_rgb, alpha_motion, alpha_geom).
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TemporalAttentionPool(nn.Module):
    """
    Learns frame-level importance weights across T video keyframes so that the peak
    bat-ball impact frames receive higher weight than preparatory stance frames.
    """

    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.Tanh(),
            nn.Linear(feat_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (B, T, D).

        Returns:
            pooled: Tensor of shape (B, D).
            weights: Temporal attention weights of shape (B, T).
        """
        scores = self.attn(x).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=-1)  # (B, T)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)
        return pooled, weights


class ClassicalIPMLP(nn.Module):
    """
    Model A: Classifier trained strictly on the 102-D Classical Image Processing
    feature vector (HOOF + Canny-Hough Bat Angles + MHI-HOG + HSV Pitch Stats).
    """

    def __init__(
        self,
        ip_feat_dim: int = 102,
        embed_dim: int = 128,
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(ip_feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, ip_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder(ip_vec)
        logits = self.classifier(feat)
        return {"logits": logits, "features": feat}


class RGBBaselineNet(nn.Module):
    """
    Model B / C: Single-Stream ResNet-18 Video Keyframe Classifier.
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        use_temporal_attn: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.feat_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.use_temporal_attn = use_temporal_attn
        if use_temporal_attn:
            self.temp_pool = TemporalAttentionPool(self.feat_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, num_classes),
        )

    def forward(self, keyframes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            keyframes: Tensor of shape (B, T, 3, H, W).
        """
        b, t, c, h, w = keyframes.shape
        flat = keyframes.view(b * t, c, h, w)
        frame_feats = self.backbone(flat).view(b, t, self.feat_dim)

        if self.use_temporal_attn:
            vid_feat, temp_weights = self.temp_pool(frame_feats)
        else:
            vid_feat = frame_feats.mean(dim=1)
            temp_weights = torch.full(
                (b, t), 1.0 / t, device=keyframes.device
            )

        logits = self.head(vid_feat)
        return {
            "logits": logits,
            "features": vid_feat,
            "temporal_weights": temp_weights,
        }


class GatedCrossModalAttention(nn.Module):
    """
    Novel Gated Cross-Modal Attention Module (GCAM).

    Given three aligned modality embeddings:
      - e_rgb    in R^D (CLAHE-enhanced, crowd-suppressed appearance stream)
      - e_motion in R^D (MHI + Dense Optical Flow spatial-temporal template stream)
      - e_geom   in R^D (Explicit HOOF + Canny-Hough Bat Orientation + HOG stream)

    computes adaptive per-instance gating weights alpha = Softmax(W_g [e_rgb || e_motion || e_geom])
    in R^3 and applies a residual cross-modality context refinement before classification.
    """

    def __init__(self, embed_dim: int = 256, num_modalities: int = 3) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities

        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_modalities),
        )

        self.context_proj = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(
        self,
        e_rgb: torch.Tensor,
        e_motion: torch.Tensor,
        e_geom: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused_embedding: (B, D)
            modality_weights: (B, 3) where columns are [alpha_rgb, alpha_motion, alpha_geom]
        """
        concat = torch.cat([e_rgb, e_motion, e_geom], dim=-1)  # (B, 3D)
        gate_logits = self.gate_net(concat)  # (B, 3)
        alpha = F.softmax(gate_logits, dim=-1)  # (B, 3)

        # Residual-anchored cross-modal gating: preserves pretrained appearance
        # features while injecting gated motion & classical IP geometric cues
        weighted_aux = (
            alpha[:, 1:2] * e_motion + alpha[:, 2:3] * e_geom
        )
        interaction = self.context_proj(concat)  # (B, D)

        fused = F.layer_norm(
            e_rgb + 0.45 * weighted_aux + 0.25 * interaction, (self.embed_dim,)
        )
        return fused, alpha


class CricIPFusionNet(nn.Module):
    """
    Proposed Method: CricIP-FusionNet (Tri-Stream Image Processing & Deep Gated Network).
    """

    def __init__(
        self,
        ip_feat_dim: int = 102,
        embed_dim: int = 256,
        num_classes: int = 10,
        pretrained: bool = True,
        use_geom_stream: bool = True,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.use_geom_stream = use_geom_stream
        self.embed_dim = embed_dim

        # Stream 1: Appearance Backbone (ResNet-18 on CLAHE + Crowd-Suppressed Keyframes)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        rgb_backbone = models.resnet18(weights=weights)
        rgb_dim = rgb_backbone.fc.in_features  # 512
        rgb_backbone.fc = nn.Identity()
        self.rgb_backbone = rgb_backbone
        self.temp_pool = TemporalAttentionPool(rgb_dim)
        self.rgb_proj = nn.Sequential(
            nn.Linear(rgb_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Stream 2: Motion Template Backbone (Lightweight ResNet-18 / Conv Encoder on MHI+Flow composite)
        motion_backbone = models.resnet18(weights=weights)
        motion_dim = motion_backbone.fc.in_features
        motion_backbone.fc = nn.Identity()
        self.motion_backbone = motion_backbone
        self.motion_proj = nn.Sequential(
            nn.Linear(motion_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Stream 3: Explicit Classical IP Geometric Descriptor Encoder (HOOF + Canny-Hough + HOG)
        self.geom_encoder = nn.Sequential(
            nn.Linear(ip_feat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Novel Gated Cross-Modal Attention Fusion
        self.gcam = GatedCrossModalAttention(embed_dim=embed_dim, num_modalities=3)

        # Final Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(
        self,
        keyframes: torch.Tensor,
        motion_map: torch.Tensor,
        ip_vec: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            keyframes: (B, T, 3, H, W) normalized CLAHE + crowd-suppressed keyframes.
            motion_map: (B, 3, H, W) normalized [MHI, OpticalFlow, MEI] composite image.
            ip_vec: (B, 102) explicit classical IP descriptor (HOOF + Hough + HOG + HSV).

        Returns:
            Dict containing:
                - 'logits': (B, num_classes)
                - 'fused_features': (B, embed_dim)
                - 'modality_weights': (B, 3) [alpha_rgb, alpha_motion, alpha_geom]
                - 'temporal_weights': (B, T) keyframe importance weights
        """
        b, t, c, h, w = keyframes.shape

        # 1. Appearance Stream
        flat_rgb = keyframes.view(b * t, c, h, w)
        rgb_frame_feats = self.rgb_backbone(flat_rgb).view(b, t, -1)
        rgb_pooled, temp_weights = self.temp_pool(rgb_frame_feats)
        e_rgb = self.rgb_proj(rgb_pooled)

        # 2. Motion Template Stream (MHI + Optical Flow)
        motion_raw = self.motion_backbone(motion_map)
        e_motion = self.motion_proj(motion_raw)

        # 3. Classical IP Geometric Stream (HOOF + Canny-Hough + HOG)
        if self.use_geom_stream:
            e_geom = self.geom_encoder(ip_vec)
        else:
            e_geom = torch.zeros_like(e_rgb)

        # 4. Gated Cross-Modal Attention Fusion
        fused, modality_weights = self.gcam(e_rgb, e_motion, e_geom)
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "fused_features": fused,
            "modality_weights": modality_weights,
            "temporal_weights": temp_weights,
        }

