"""Pooled metrics, paired tests and patch accounting for every model, from the saved runs.

    python analysis/compute_statistics.py RUNS_DIR [OUT_DIR]
"""
import csv
import json
import os
import sys

import numpy as np
from scipy import stats

RUNS = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join("results", "tables")
os.makedirs(OUT, exist_ok=True)

MODELS = {  # key: (run folder, display name)
    "uni": ("pfm_uni", "FrozenSE-UNI"),
    "phikon": ("pfm_phikon", "FrozenSE-Phikon"),
    "sam": ("improved_skip_partial", "FrozenSE-SAM"),
    "unet_pre": ("baseline_unet_pretrained_r2", "U-Net (ImageNet)"),
    "unet_scr": ("baseline_unet_scratch_r2", "U-Net (scratch)"),
    "nnunet": ("nnunet_2d", "nnU-Net (2D)"),
    "zero": ("baseline_mobilesam_native", "MobileSAM (zero-shot)"),
    "orig": ("proposed_frozense_sam", "FrozenSE-SAM, fully frozen, no skips"),
    "uni_lin": ("pfm_uni_linear_head", "UNI + linear head"),
    "uni_frz": ("pfm_uni_fully_frozen", "FrozenSE-UNI, fully frozen encoder"),
}


def folds(key):
    folder = MODELS[key][0]
    if key == "nnunet":
        return [json.load(open(os.path.join(RUNS, folder, f"fold{k}", "result.json"))) for k in range(6)]
    return [json.load(open(os.path.join(RUNS, folder, f"fold{k}", "fold_result.json"))) for k in range(6)]


def boot(x, n=2000, seed=0):
    """Percentile bootstrap 95% CI of the mean, resampling cores."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


D = {}
for key in MODELS:
    cores, tumour, cases, fold_dice = {}, {}, {}, []
    for f in folds(key):
        r = f["result"]
        fold_dice.append(float(np.mean([c["dice"] for c in r["cores"]])))
        for c in r["cores"]:
            cores[c["core_id"]] = c
        for p in r["patches"]:
            cases[p["case"]] = cases.get(p["case"], 0) + 1
            if p["case"] in ("both_present", "gt_not_pred_empty"):       # label contains tumour
                tumour.setdefault(p["core_id"], []).append(p["dice"])
    ids = sorted(cores)
    D[key] = {"ids": ids, "cores": cores, "dice": np.array([cores[i]["dice"] for i in ids]),
              "tumour": {i: float(np.mean(v)) for i, v in tumour.items()}, "cases": cases, "fold": fold_dice}
assert all(D[k]["ids"] == D["sam"]["ids"] for k in D), "models were not scored on the same cores"
IDS = D["sam"]["ids"]


def metric(key, field, scale=1.0):
    v = np.array([D[key]["cores"][i][field] for i in IDS if D[key]["cores"][i].get(field) is not None], float) * scale
    v = v[~np.isnan(v)]
    return (float(v.mean()),) + boot(v)


def pair(a, b, tumour=False):
    if tumour:
        ids = sorted(set(D[a]["tumour"]) & set(D[b]["tumour"]))
        x = np.array([D[a]["tumour"][i] for i in ids]); y = np.array([D[b]["tumour"][i] for i in ids])
    else:
        x, y = D[a]["dice"], D[b]["dice"]
    d = x - y
    lo, hi = boot(d)
    return {"model_a": MODELS[a][1], "model_b": MODELS[b][1], "n_cores": len(d), "mean_diff_pp": d.mean() * 100,
            "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "median_diff_pp": float(np.median(d)) * 100,
            "p_wilcoxon": float(stats.wilcoxon(x, y).pvalue), "cores_a_better": int((d > 0).sum())}


def holm(ps):
    order = np.argsort(ps); adj = [0.0] * len(ps); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i])); adj[i] = running
    return adj


def write(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


# ---- main results
main = []
for key in ("uni", "phikon", "sam", "unet_pre", "unet_scr", "nnunet", "zero", "orig", "uni_lin", "uni_frz"):
    d, lo, hi = (float(D[key]["dice"].mean()),) + boot(D[key]["dice"])
    iou, hd, asd = metric(key, "iou"), metric(key, "hd95_mm", 1000), metric(key, "asd_mm", 1000)
    main.append({"model": MODELS[key][1], "n_cores": len(IDS), "dice_pct": d * 100, "dice_ci_low": lo * 100,
                 "dice_ci_high": hi * 100, "iou_pct": iou[0] * 100, "iou_ci_low": iou[1] * 100,
                 "iou_ci_high": iou[2] * 100, "hd95_um": hd[0], "hd95_ci_low": hd[1], "hd95_ci_high": hd[2],
                 "asd_um": asd[0], "slide_mean_dice_pct": np.mean(D[key]["fold"]) * 100,
                 "slide_sd_dice_pct": np.std(D[key]["fold"], ddof=1) * 100})
write("main_results.csv", main)

# ---- paired tests; Holm within each family
families = {
    "FrozenSE-UNI vs trained baselines": [("uni", b) for b in ("unet_pre", "unet_scr", "nnunet")],
    "FrozenSE-Phikon vs trained baselines": [("phikon", b) for b in ("unet_pre", "unet_scr", "nnunet")],
    "FrozenSE-SAM vs trained baselines": [("sam", b) for b in ("unet_pre", "unet_scr", "nnunet")],
    "between encoders": [("uni", "phikon"), ("uni", "sam"), ("phikon", "sam")],
    "FrozenSE components on UNI": [("uni", "uni_lin"), ("uni", "uni_frz")],
}
tests = []
for fam, comps in families.items():
    res = [pair(a, b) for a, b in comps]
    for r, h in zip(res, holm([r["p_wilcoxon"] for r in res])):
        tests.append({"family": fam, **r, "p_holm": h})
for a, b in (("sam", "zero"), ("sam", "orig"), ("unet_scr", "unet_pre"), ("nnunet", "unet_pre"), ("zero", "unet_pre"),
             ("uni_lin", "unet_pre"), ("uni_frz", "unet_pre"), ("uni_frz", "uni_lin")):
    tests.append({"family": "descriptive (unadjusted)", **pair(a, b), "p_holm": ""})
write("paired_tests.csv", tests)

# ---- tumour-containing and tumour-free patches
acc = []
for key in ("uni", "phikon", "sam", "unet_pre", "unet_scr", "nnunet", "zero", "uni_lin", "uni_frz"):
    c = D[key]["cases"]; free = c.get("both_empty", 0) + c.get("gt_empty_pred_not", 0)
    acc.append({"model": MODELS[key][1], "tumour_patches_tumour_predicted": c.get("both_present", 0),
                "tumour_patches_none_predicted": c.get("gt_not_pred_empty", 0),
                "tumour_free_correctly_empty": c.get("both_empty", 0),
                "tumour_free_correctly_empty_pct": 100 * c.get("both_empty", 0) / free,
                "tumour_free_false_positive": c.get("gt_empty_pred_not", 0),
                "dice_on_tumour_patches_pct": 100 * float(np.mean(list(D[key]["tumour"].values()))),
                "n_cores_with_tumour": len(D[key]["tumour"])})
write("patch_accounting.csv", acc)

# ---- per slide and per core
slides = ["slide001", "slide002", "slide003", "slide005", "slide006", "slide007"]
write("per_slide_dice.csv", [{"model": MODELS[k][1], **{s: D[k]["fold"][j] * 100 for j, s in enumerate(slides)}}
                             for k in MODELS])
write("per_core_dice.csv", [{"core_id": i, "slide": i.split("_")[0],
                             **{MODELS[k][1]: D[k]["cores"][i]["dice"] for k in MODELS}} for i in IDS])

summary = {"main": main, "paired": tests, "patch_accounting": acc,
           "fold_dice": {k: D[k]["fold"] for k in MODELS}}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)

print(f"{'model':38s} Dice % [95% CI]")
for r in main:
    print(f"{r['model']:38s} {r['dice_pct']:5.1f} [{r['dice_ci_low']:.1f}, {r['dice_ci_high']:.1f}]")
print(f"\ntables written to {OUT}")
