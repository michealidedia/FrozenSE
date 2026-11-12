"""
Segmentation losses.

    total = TverskyLoss(alpha, beta) + focal_weight * FocalLoss(focal_alpha, gamma)

which is an ADDITIVE COMPOUND of two separate terms, not Focal Tversky Loss.
Focal Tversky Loss (Abraham & Khan, 2019) is a single term:

    FTL = (1 - TverskyIndex(alpha, beta)) ** gamma

A NOTE ON alpha AND beta
------------------------
Following Salehi et al. (2017):

    TI = TP / (TP + alpha*FN + beta*FP)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig


def _flatten_binary(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.shape[1] != 1:
        raise ValueError(f"Binary losses expect C=1, got logits {tuple(logits.shape)}")
    if targets.shape[1] != 1:
        raise ValueError(f"Binary losses expect C=1, got targets {tuple(targets.shape)}")
    return logits.reshape(logits.shape[0], -1), targets.reshape(targets.shape[0], -1)


def tversky_index(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    beta: float,
    smooth: float = 1e-6,
    per_sample: bool = True,
) -> torch.Tensor:
    """Tversky index. Computed per sample by default.

    Per-sample computation keeps the loss independent of batch composition.
    """
    probs = torch.sigmoid(logits)
    p, t = _flatten_binary(probs, targets)

    tp = (p * t).sum(dim=1)
    fn = ((1 - p) * t).sum(dim=1)
    fp = (p * (1 - t)).sum(dim=1)

    ti = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return ti if per_sample else ti.mean()


class TverskyLoss(nn.Module):
    """1 - TI(alpha, beta)."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (1.0 - tversky_index(logits, targets, self.alpha, self.beta, self.smooth)).mean()


class FocalTverskyLoss(nn.Module):
    """(1 - TI(alpha, beta)) ** gamma  -- Abraham & Khan (2019).
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5,
                 gamma: float = 1.33, smooth: float = 1e-6):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ti = tversky_index(logits, targets, self.alpha, self.beta, self.smooth)
        return ((1.0 - ti).clamp_min(1e-8) ** self.gamma).mean()


class FocalLoss(nn.Module):
    """Pixelwise focal loss on top of BCE (Lin et al., 2017)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1,
                              torch.as_tensor(self.alpha, device=logits.device),
                              torch.as_tensor(1 - self.alpha, device=logits.device))
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p, t = _flatten_binary(probs, targets)
        inter = (p * t).sum(dim=1)
        dice = (2 * inter + self.smooth) / (p.sum(dim=1) + t.sum(dim=1) + self.smooth)
        return (1 - dice).mean()


class TverskyPlusFocalLoss(nn.Module):
    """Tversky(alpha, beta) + w * Focal(focal_alpha, gamma).
    """

    def __init__(self, alpha: float = 0.4, beta: float = 0.6,
                 focal_alpha: float = 0.20, gamma: float = 2.0,
                 focal_weight: float = 1.0, smooth: float = 1e-6):
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta, smooth)
        self.focal = FocalLoss(focal_alpha, gamma)
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.tversky(logits, targets) + self.focal_weight * self.focal(logits, targets)

    def components(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Individual terms, for the loss-curve figure."""
        with torch.no_grad():
            tv = self.tversky(logits, targets)
            fc = self.focal(logits, targets)
        return {
            "tversky": tv.item(),
            "focal": fc.item(),
            "total": (tv + self.focal_weight * fc).item(),
        }


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)
        self.bw, self.dw = bce_weight, dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bw * self.bce(logits, targets) + self.dw * self.dice(logits, targets)


# --------------------------------------------------------------------------
# Auxiliary signed-distance supervision
# --------------------------------------------------------------------------
def signed_distance_target(mask: torch.Tensor, clip: float = 20.0) -> torch.Tensor:
    """Compute a normalised signed distance field from a binary mask.

    Negative inside the object, positive outside, clipped and scaled to
    [-1, 1].  Used as the TARGET for the auxiliary head; requires SciPy.
    """
    from scipy.ndimage import distance_transform_edt
    import numpy as np

    out = torch.zeros_like(mask)
    arr = mask.detach().cpu().numpy()
    for b in range(arr.shape[0]):
        m = arr[b, 0] > 0.5
        if m.any() and (~m).any():
            inside = distance_transform_edt(m)
            outside = distance_transform_edt(~m)
            sdf = outside - inside
        else:
            sdf = np.full(m.shape, clip, dtype=np.float32)
        sdf = np.clip(sdf, -clip, clip) / clip
        out[b, 0] = torch.from_numpy(sdf.astype("float32")).to(mask.device)
    return out


class SegmentationWithAuxSDF(nn.Module):
    """Primary segmentation loss + auxiliary SDF regression.
    """

    def __init__(self, primary: nn.Module, sdf_weight: float = 0.5):
        super().__init__()
        self.primary = primary
        self.sdf_weight = sdf_weight
        self.sdf_criterion = nn.L1Loss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sdf_pred: torch.Tensor | None = None) -> torch.Tensor:
        loss = self.primary(logits, targets)
        if sdf_pred is not None:
            sdf_target = signed_distance_target(targets)
            loss = loss + self.sdf_weight * self.sdf_criterion(sdf_pred, sdf_target)
        return loss


# --------------------------------------------------------------------------
def build_loss(cfg: LossConfig) -> nn.Module:
    """Instantiate the loss named in the config. No silent substitutions."""
    name = cfg.name
    if name == "tversky":
        return TverskyLoss(cfg.tversky_alpha, cfg.tversky_beta, cfg.smooth)
    if name == "focal_tversky":
        return FocalTverskyLoss(cfg.tversky_alpha, cfg.tversky_beta,
                                cfg.focal_gamma, cfg.smooth)
    if name == "tversky_plus_focal":
        return TverskyPlusFocalLoss(cfg.tversky_alpha, cfg.tversky_beta,
                                    cfg.focal_alpha, cfg.focal_gamma,
                                    cfg.focal_weight, cfg.smooth)
    if name == "dice":
        return DiceLoss(cfg.smooth)
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "bce_dice":
        return BCEDiceLoss(smooth=cfg.smooth)
    raise ValueError(f"Unknown loss name {name!r}")


def describe_loss(cfg: LossConfig) -> str:
    """One-line description suitable for pasting into the Methods section."""
    a, b, g = cfg.tversky_alpha, cfg.tversky_beta, cfg.focal_gamma
    if cfg.name == "focal_tversky":
        return (f"Focal Tversky Loss, FTL = (1 - TI)^gamma with "
                f"TI = TP/(TP + {a}*FN + {b}*FP), gamma = {g} "
                f"(Abraham & Khan, 2019).")
    if cfg.name == "tversky_plus_focal":
        return (f"Compound objective L = L_Tversky + {cfg.focal_weight} * L_Focal, "
                f"where L_Tversky = 1 - TP/(TP + {a}*FN + {b}*FP) and L_Focal is the "
                f"pixelwise focal loss with alpha = {cfg.focal_alpha}, gamma = {g}. "
                f"Note that alpha weights false negatives and beta false positives, "
                f"so the configuration "
                f"{'favours recall' if a > b else 'favours precision'}.")
    if cfg.name == "tversky":
        return f"Tversky loss, 1 - TP/(TP + {a}*FN + {b}*FP)."
    return f"{cfg.name} loss."
