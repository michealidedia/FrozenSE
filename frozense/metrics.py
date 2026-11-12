"""
Segmentation metrics, per-core aggregation and uncertainty.

1. EMPTY MASKS. Patches whose distance metrics are undefined (empty prediction
   or empty ground truth) are counted and reported, never dropped silently.

2. PER-CORE AGGREGATION. Patches drawn from the same core are not independent,
   so metrics are aggregated per core first, and all inference is over cores.

3. UNCERTAINTY. Bootstrap confidence intervals over cores, and paired Wilcoxon
   signed-rank tests between models evaluated on the same cores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    from scipy.stats import wilcoxon
    _SCIPY = True
except ImportError:  # pragma: no cover
    _SCIPY = False


EmptyCase = Literal["both_empty", "gt_empty_pred_not", "gt_not_pred_empty", "both_present"]


# --------------------------------------------------------------------------
# Per-patch primitives
# --------------------------------------------------------------------------
def dice_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """Dice. Two empty masks score 1.0 (a correct prediction of absence)."""
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    denom = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * np.logical_and(pred_b, gt_b).sum() + smooth) / (denom + smooth))


def iou_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred_b, gt_b).sum()
    if union == 0:
        return 1.0
    return float((np.logical_and(pred_b, gt_b).sum() + smooth) / (union + smooth))


def classify_case(pred: np.ndarray, gt: np.ndarray) -> EmptyCase:
    p, g = pred.any(), gt.any()
    if not p and not g:
        return "both_empty"
    if not g and p:
        return "gt_empty_pred_not"
    if g and not p:
        return "gt_not_pred_empty"
    return "both_present"


def _surface_points(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of a binary mask."""
    eroded = binary_erosion(mask, border_value=0)
    return np.argwhere(np.logical_and(mask, ~eroded))


def surface_distances(pred: np.ndarray, gt: np.ndarray,
                      spacing_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric surface distances in millimetres.

    Returns (distances from GT surface to prediction, distances from
    prediction surface to GT).  Both arrays are empty if either surface is.
    """
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    if not pred_b.any() or not gt_b.any():
        return np.array([]), np.array([])

    # Distance transform of the complement gives, at each pixel, the distance
    # to the nearest foreground pixel of the other mask.
    dt_pred = distance_transform_edt(~pred_b) * spacing_mm
    dt_gt = distance_transform_edt(~gt_b) * spacing_mm

    gt_surf = _surface_points(gt_b)
    pred_surf = _surface_points(pred_b)
    if gt_surf.size == 0 or pred_surf.size == 0:
        return np.array([]), np.array([])

    d_gt_to_pred = dt_pred[gt_surf[:, 0], gt_surf[:, 1]]
    d_pred_to_gt = dt_gt[pred_surf[:, 0], pred_surf[:, 1]]
    return d_gt_to_pred, d_pred_to_gt


def hd95_and_asd(pred: np.ndarray, gt: np.ndarray,
                 spacing_mm: float) -> tuple[float, float]:
    """95th-percentile Hausdorff distance and average surface distance, in mm.

    Returns (nan, nan) when either mask is empty; the caller decides how to
    treat those cases via the empty-mask policy.
    """
    d1, d2 = surface_distances(pred, gt, spacing_mm)
    if d1.size == 0 or d2.size == 0:
        return float("nan"), float("nan")
    hd95 = float(max(np.percentile(d1, 95), np.percentile(d2, 95)))
    asd = float((d1.mean() + d2.mean()) / 2.0)
    return hd95, asd


# --------------------------------------------------------------------------
# Accumulation
# --------------------------------------------------------------------------
@dataclass
class PatchResult:
    core_id: str
    dice: float
    iou: float
    hd95_mm: float
    asd_mm: float
    case: EmptyCase


@dataclass
class CoreResult:
    core_id: str
    n_patches: int
    dice: float
    iou: float
    hd95_mm: float          # nan if no patch in this core had a defined value
    asd_mm: float
    n_defined_distance: int
    case_counts: dict[str, int]


@dataclass
class EvaluationResult:
    """Everything needed to report a model's performance."""

    model_name: str
    patches: list[PatchResult]
    cores: list[CoreResult]
    spacing_mm: float
    empty_mask_policy: str
    case_counts: dict[str, int] = field(default_factory=dict)

    # ---- persistence ----
    # Long cross-validation runs outlive a free-tier session (Kaggle stops
    # at 12 h, Colab disconnects sooner), so every fold's result is saved as
    # soon as it exists and reloaded instead of being recomputed. NaN, which
    # marks an undefined surface distance, is stored as null so the files
    # stay valid JSON.
    def to_dict(self) -> dict:
        nan_to_none = lambda v: None if isinstance(v, float) and not np.isfinite(v) else v
        return {
            "model_name": self.model_name,
            "spacing_mm": self.spacing_mm,
            "empty_mask_policy": self.empty_mask_policy,
            "case_counts": dict(self.case_counts),
            "cores": [{k: nan_to_none(v) for k, v in vars(c).items()} for c in self.cores],
            "patches": [{k: nan_to_none(v) for k, v in vars(p).items()} for p in self.patches],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvaluationResult":
        none_to_nan = lambda v: float("nan") if v is None else v
        cores = [CoreResult(**{k: none_to_nan(v) for k, v in c.items()}) for c in d["cores"]]
        patches = [PatchResult(**{k: none_to_nan(v) for k, v in p.items()})
                   for p in d["patches"]]
        return cls(model_name=d["model_name"], patches=patches, cores=cores,
                   spacing_mm=d["spacing_mm"], empty_mask_policy=d["empty_mask_policy"],
                   case_counts=dict(d.get("case_counts", {})))

    # ---- point estimates over CORES ----
    def mean(self, metric: str) -> float:
        vals = [getattr(c, metric) for c in self.cores]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def std(self, metric: str) -> float:
        vals = [getattr(c, metric) for c in self.cores]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")

    def values(self, metric: str) -> np.ndarray:
        return np.array([getattr(c, metric) for c in self.cores], dtype=float)

    def bootstrap_ci(self, metric: str, n: int = 2000,
                     level: float = 0.95, seed: int = 0) -> tuple[float, float]:
        """Percentile bootstrap CI over cores."""
        vals = self.values(metric)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            return float("nan"), float("nan")
        rng = np.random.default_rng(seed)
        means = np.array([
            rng.choice(vals, size=vals.size, replace=True).mean() for _ in range(n)
        ])
        lo = float(np.percentile(means, 100 * (1 - level) / 2))
        hi = float(np.percentile(means, 100 * (1 + level) / 2))
        return lo, hi

    def summary_row(self, n_boot: int = 2000, level: float = 0.95,
                    seed: int = 0, spacing_mm: float | None = None) -> dict:
        sp = spacing_mm or self.spacing_mm
        row: dict[str, object] = {"model": self.model_name, "n_cores": len(self.cores)}
        for m, label in [("dice", "Dice"), ("iou", "IoU")]:
            lo, hi = self.bootstrap_ci(m, n_boot, level, seed)
            row[f"{label}"] = self.mean(m)
            row[f"{label}_sd"] = self.std(m)
            row[f"{label}_ci_low"] = lo
            row[f"{label}_ci_high"] = hi
        for m, label in [("hd95_mm", "HD95"), ("asd_mm", "ASD")]:
            lo, hi = self.bootstrap_ci(m, n_boot, level, seed)
            row[f"{label}_mm"] = self.mean(m)
            row[f"{label}_mm_sd"] = self.std(m)
            row[f"{label}_mm_ci_low"] = lo
            row[f"{label}_mm_ci_high"] = hi
            row[f"{label}_px"] = self.mean(m) / sp if sp else float("nan")
        row.update({f"n_{k}": v for k, v in self.case_counts.items()})
        row["empty_mask_policy"] = self.empty_mask_policy
        return row


class MetricAccumulator:
    """Collects per-patch results and aggregates them per core."""

    def __init__(self, model_name: str, spacing_mm: float,
                 empty_mask_policy: str = "report_separately",
                 patch_diagonal_mm: float | None = None):
        if not _SCIPY:
            raise ImportError("scipy is required for surface distance metrics")
        self.model_name = model_name
        self.spacing_mm = spacing_mm
        self.policy = empty_mask_policy
        self.patch_diagonal_mm = patch_diagonal_mm
        self._patches: list[PatchResult] = []

    def add_batch(self, probs: np.ndarray, targets: np.ndarray,
                  core_ids: Sequence[str], threshold: float = 0.5) -> None:
        """probs and targets: (B, 1, H, W) arrays of floats."""
        preds = (probs >= threshold).astype(np.uint8)
        gts = (targets >= 0.5).astype(np.uint8)

        for i, core_id in enumerate(core_ids):
            p, g = preds[i, 0], gts[i, 0]
            case = classify_case(p, g)
            hd95, asd = hd95_and_asd(p, g, self.spacing_mm)

            if case == "both_empty":
                # A correct prediction of "no tumour here" is a perfect result,
                # and its surface distance is legitimately zero.
                hd95, asd = 0.0, 0.0
            elif case in ("gt_empty_pred_not", "gt_not_pred_empty"):
                if self.policy == "worst_case":
                    worst = self.patch_diagonal_mm or float("nan")
                    hd95, asd = worst, worst
                # "report_separately" and "drop_silently" both leave nan here;
                # they differ in whether the counts are surfaced (they always
                # are in this implementation) and in documentation intent.

            self._patches.append(PatchResult(
                core_id=core_id,
                dice=dice_score(p, g),
                iou=iou_score(p, g),
                hd95_mm=hd95,
                asd_mm=asd,
                case=case,
            ))

    def result(self) -> EvaluationResult:
        by_core: dict[str, list[PatchResult]] = {}
        for pr in self._patches:
            by_core.setdefault(pr.core_id, []).append(pr)

        cores: list[CoreResult] = []
        for core_id, items in sorted(by_core.items()):
            hd = [x.hd95_mm for x in items if np.isfinite(x.hd95_mm)]
            ad = [x.asd_mm for x in items if np.isfinite(x.asd_mm)]
            counts: dict[str, int] = {}
            for x in items:
                counts[x.case] = counts.get(x.case, 0) + 1
            cores.append(CoreResult(
                core_id=core_id,
                n_patches=len(items),
                dice=float(np.mean([x.dice for x in items])),
                iou=float(np.mean([x.iou for x in items])),
                hd95_mm=float(np.mean(hd)) if hd else float("nan"),
                asd_mm=float(np.mean(ad)) if ad else float("nan"),
                n_defined_distance=len(hd),
                case_counts=counts,
            ))

        totals: dict[str, int] = {}
        for pr in self._patches:
            totals[pr.case] = totals.get(pr.case, 0) + 1

        return EvaluationResult(
            model_name=self.model_name,
            patches=self._patches,
            cores=cores,
            spacing_mm=self.spacing_mm,
            empty_mask_policy=self.policy,
            case_counts=totals,
        )


# --------------------------------------------------------------------------
# Model comparison
# --------------------------------------------------------------------------
def paired_wilcoxon(a: EvaluationResult, b: EvaluationResult,
                    metric: str = "dice") -> dict:
    """Paired Wilcoxon signed-rank test over cores common to both models.
    """
    map_a = {c.core_id: getattr(c, metric) for c in a.cores}
    map_b = {c.core_id: getattr(c, metric) for c in b.cores}
    shared = sorted(set(map_a) & set(map_b))

    x = np.array([map_a[c] for c in shared], dtype=float)
    y = np.array([map_b[c] for c in shared], dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    # The descriptive columns are filled in every case, so a table built from
    # several comparisons always has them -- only the test itself may be
    # absent.
    row = {
        "model_a": a.model_name, "model_b": b.model_name, "metric": metric,
        "n_pairs": int(x.size),
        "mean_a": float(x.mean()) if x.size else float("nan"),
        "mean_b": float(y.mean()) if y.size else float("nan"),
        "median_difference": float(np.median(x - y)) if x.size else float("nan"),
    }

    if x.size < 5:
        return {**row, "statistic": float("nan"), "p_value": float("nan"),
                "note": "too few paired cores for a meaningful test"}

    if np.allclose(x, y):
        return {**row, "statistic": 0.0, "p_value": 1.0, "note": "identical values"}

    stat, p = wilcoxon(x, y)
    return {**row, "statistic": float(stat), "p_value": float(p)}


def comparison_table(results: Sequence[EvaluationResult],
                     reference: str,
                     metric: str = "dice") -> list[dict]:
    """Compare every model against `reference` with paired tests."""
    ref = next((r for r in results if r.model_name == reference), None)
    if ref is None:
        raise ValueError(f"No result named {reference!r}")
    return [paired_wilcoxon(ref, r, metric) for r in results
            if r.model_name != reference]


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, for the family of comparisons.

    A comparison with no valid test (p = NaN, e.g. too few paired cores) is
    not part of the family: it stays NaN and does not count towards the
    family size. This matters. Handled naively, Python's max(0.0, nan)
    returns 0.0, which turns an untestable comparison into p_holm = 0 --
    reported as SIGNIFICANT -- while also over-correcting every real test.
    """
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.size, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    n = valid.size
    running = 0.0
    for rank, idx in enumerate(valid[np.argsort(p[valid])]):
        running = max(running, (n - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()
