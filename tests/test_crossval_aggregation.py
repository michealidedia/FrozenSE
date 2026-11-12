"""Exercise the cross-validation aggregation without torch.

Builds synthetic per-fold EvaluationResults over the real 193-core/6-slide
layout and checks pooling, the two kinds of standard deviation, paired tests
and Holm correction.

Run:  python3 -I tests/test_crossval_aggregation.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                              # noqa: E402

from frozense.config import ExperimentConfig, SplitConfig       # noqa: E402
from frozense.data import Core, make_split, n_folds             # noqa: E402
from frozense.metrics import CoreResult, EvaluationResult       # noqa: E402
from frozense.crossval import (                                 # noqa: E402
    CVResult, pool_folds, cv_summary_table, cv_comparison_table,
    format_cv_table, format_comparison,
)

REAL_COUNTS = {"slide001": 36, "slide002": 29, "slide003": 28,
               "slide005": 31, "slide006": 44, "slide007": 25}


def build_cores():
    out = []
    for slide, n in REAL_COUNTS.items():
        for i in range(1, n + 1):
            cid = f"{slide}_core{i:03d}"
            out.append(Core(cid, f"/i/{cid}.jpg", f"/l/{cid}_c.png", slide))
    return out


def fake_result(model, core_ids, base_dice, rng):
    """Per-core scores drawn around `base_dice`, a few with undefined HD95."""
    cores = []
    for k, cid in enumerate(core_ids):
        d = float(np.clip(rng.normal(base_dice, 0.08), 0.0, 1.0))
        undefined = (k % 17 == 0)      # mimics an all-empty ground-truth core
        cores.append(CoreResult(
            core_id=cid, n_patches=20, dice=d, iou=d / (2 - d),
            hd95_mm=float("nan") if undefined else float(rng.uniform(0.02, 0.4)),
            asd_mm=float("nan") if undefined else float(rng.uniform(0.005, 0.1)),
            n_defined_distance=0 if undefined else 20,
            case_counts={"both_present": 20},
        ))
    return EvaluationResult(model_name=model, patches=[], cores=cores,
                            spacing_mm=0.00025,
                            empty_mask_policy="report_separately",
                            case_counts={"both_present": 20 * len(core_ids)})


def build_cv(model, cores, base_dice, seed, fold_offsets):
    """One CVResult built by running the real splitter over all 6 folds."""
    rng = np.random.default_rng(seed)
    fold_results, manifests = [], []
    for fold in range(n_folds(cores)):
        split = make_split(cores, SplitConfig(fold=fold, n_train_cores=35))
        manifests.append(split.manifest())
        ids = [c.core_id for c in split.test]
        fold_results.append(
            fake_result(model, ids, base_dice + fold_offsets[fold], rng)
        )
    pooled = pool_folds(fold_results, model)
    return CVResult(model, fold_results, manifests,
                    [{"epochs_run": 40}] * len(fold_results), pooled,
                    seconds=120.0)


def check(label, condition, detail=""):
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        raise AssertionError(label + " " + detail)


def main() -> int:
    cores = build_cores()
    cfg = ExperimentConfig()

    # Slide-dependent offsets: a real staining batch effect, so the
    # across-fold SD should be clearly larger than seed noise.
    offsets = [0.00, -0.05, 0.04, -0.09, 0.06, -0.02]

    print("pooling")
    a = build_cv("proposed_frozense_sam", cores, 0.63, 1, offsets)
    check("pooled covers all 193 cores", len(a.pooled.cores) == 193,
          f"got {len(a.pooled.cores)}")
    check("six folds recorded", len(a.fold_results) == 6)
    check("no duplicate core in pooled",
          len({c.core_id for c in a.pooled.cores}) == 193)

    # Pooled mean is a core-weighted mean; the fold mean is unweighted.
    # They differ because slides have different core counts -- worth
    # reporting both, and worth not confusing them.
    print(f"  pooled Dice {a.pooled.mean('dice'):.4f}   "
          f"fold mean {a.fold_mean('dice'):.4f} +/- {a.fold_std('dice'):.4f}")
    check("pooled and fold means differ (slides are unequal sizes)",
          abs(a.pooled.mean("dice") - a.fold_mean("dice")) > 1e-6)
    check("across-fold SD reflects the injected slide effect",
          a.fold_std("dice") > 0.03, f"fold SD {a.fold_std('dice'):.4f}")
    check("core SD is larger than fold SD",
          a.pooled.std("dice") > a.fold_std("dice"),
          f"core {a.pooled.std('dice'):.4f} vs fold {a.fold_std('dice'):.4f}")

    print("\ndisjointness guard")
    try:
        pool_folds([a.fold_results[0], a.fold_results[0]], "dup")
        check("pooling overlapping folds raises", False)
    except AssertionError as exc:
        check("pooling overlapping folds raises",
              "not disjoint" in str(exc))

    print("\nsummary table")
    b = build_cv("baseline_unet_pretrained", cores, 0.66, 2, offsets)
    c = build_cv("baseline_unet_scratch", cores, 0.52, 3, offsets)
    d = build_cv("baseline_mobilesam_native", cores, 0.31, 4, offsets)

    rows = cv_summary_table([a, b, c, d], cfg)
    check("one row per model", len(rows) == 4)
    for r in rows:
        check(f"  {r['model']}: has CI and both SDs",
              all(np.isfinite(r[k]) for k in
                  ("Dice", "Dice_ci_low", "Dice_ci_high",
                   "Dice_core_sd", "Dice_fold_sd")))
        check(f"  {r['model']}: CI brackets the point estimate",
              r["Dice_ci_low"] <= r["Dice"] <= r["Dice_ci_high"])
        check(f"  {r['model']}: records 6 folds and held-out slides",
              r["n_folds"] == 6 and len(r["held_out_slides"]) == 6)
        check(f"  {r['model']}: 35 training cores every fold",
              set(r["train_cores_per_fold"]) == {35})
    print()
    print(format_cv_table(rows))

    print("\npaired comparison against the proposed model")
    comp = cv_comparison_table([a, b, c, d], reference="proposed_frozense_sam")
    check("three comparisons", len(comp) == 3)
    for r in comp:
        check(f"  vs {r['model_b']}: paired over real cores",
              r["n_pairs"] > 150, f"n={r['n_pairs']}")
        check(f"  vs {r['model_b']}: Holm p >= raw p",
              r["p_holm"] >= r["p_value"] - 1e-12)
    print()
    print(format_comparison(comp))

    weaker = next(r for r in comp if r["model_b"] == "baseline_unet_scratch")
    check("scratch U-Net is detected as worse",
          weaker["median_difference"] > 0 and weaker["significant_at_0.05"])

    print("\nserialisation")
    out = os.path.join(os.environ.get("FROZENSE_TEST_DIR", os.path.join(
        tempfile.gettempdir(), "frozense_tests")), "cv_out", "cv_result.json")
    path = a.save(out)
    import json
    with open(path) as fh:
        payload = json.load(fh)
    check("saved per-core scores for all 193", len(payload["per_core"]) == 193)
    check("saved six manifests", len(payload["manifests"]) == 6)
    check("manifests name the held-out slide",
          [m["groups"]["test"][0] for m in payload["manifests"]]
          == sorted(REAL_COUNTS))

    print("\nall checks passed")
    return 0


def test_holm_with_untestable_comparisons():
    """A NaN p-value must stay NaN and must not enlarge the family."""
    from frozense.metrics import holm_bonferroni
    print("\nHolm-Bonferroni with an untestable comparison")
    adj = holm_bonferroni([0.01, float("nan"), 0.04])
    check("NaN stays NaN (was silently turned into 0.0 = 'significant')",
          np.isnan(adj[1]), str(adj))
    check("family size counts only real tests (2, not 3)",
          np.isclose(adj[0], 0.02) and np.isclose(adj[2], 0.04), str(adj))
    check("all-NaN input stays all NaN", all(np.isnan(holm_bonferroni([np.nan, np.nan]))))
    check("standard case unchanged",
          np.allclose(holm_bonferroni([0.01, 0.02, 0.03]), [0.03, 0.04, 0.04]))


if __name__ == "__main__":
    rc = main()
    test_holm_with_untestable_comparisons()
    raise SystemExit(rc)
