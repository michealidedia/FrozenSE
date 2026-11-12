"""    python analysis/compute_statistics.py RUNS_DIR results/tables
    python analysis/make_figures.py RUNS_DIR figures [results/tables] [--tiff]

RUNS_DIR is the unpacked results/raw_runs.zip. Each figure is written as a vector PDF
(fonts embedded) and a 600 dpi PNG at full-page width (190 mm); --tiff adds an LZW TIFF.
Figure 2 shows an H&E patch from the dataset and is not regenerated here.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image
from scipy import stats

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
RES, OUT = ARGS[0], ARGS[1]
TABLES = ARGS[2] if len(ARGS) > 2 else os.path.join("results", "tables")
TIFF = "--tiff" in sys.argv
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#6b6b6b", "axes.linewidth": 0.7, "xtick.color": "#4a4a4a",
    "ytick.color": "#4a4a4a", "axes.labelcolor": "#262626", "text.color": "#262626",
    "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "savefig.dpi": 300,
    "figure.dpi": 150, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
INK, MUTED = "#262626", "#6b6b6b"
# Validated categorical palette (fixed order; colour follows the model).
C = {"final": "#C55A11", "unet_pre": "#2F5597", "unet_scr": "#2A9D8F",
     "zeroshot": "#8E6BB8", "original": "#B8860B"}
LABEL = {"final": "FrozenSE-SAM (final)", "unet_pre": "U-Net, ImageNet encoder",
         "unet_scr": "U-Net, from scratch", "zeroshot": "MobileSAM, zero-shot",
         "original": "FrozenSE-SAM (original design)"}
RUN = {"final": "improved_skip_partial", "unet_pre": "baseline_unet_pretrained_r2",
       "unet_scr": "baseline_unet_scratch_r2", "zeroshot": "baseline_mobilesam_native",
       "original": "proposed_frozense_sam", "skip": "improved_skip",
       "se_off": "ablation_se_off", "aux_sdf": "ablation_aux_sdf",
       "ftl": "loss_focal_tversky", "dice": "loss_dice",
       "unet_pre_r1": "baseline_unet_pretrained", "unet_scr_r1": "baseline_unet_scratch"}
SLIDES = ["slide001", "slide002", "slide003", "slide005", "slide006", "slide007"]
CORES = [36, 29, 28, 31, 44, 25]


def load_cv(name, subset=False):
    f = "cv_result_folds024.json" if subset else "cv_result.json"
    return json.load(open(os.path.join(RES, name, f)))


def per_core(cv):
    return {c["core_id"]: c for c in cv["per_core"]}


def boot_mean_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


facts = {}

# ------------------------------------------------------------------ paired stats
def paired(a_name, b_name, subset=False):
    A, B = per_core(load_cv(a_name, subset)), per_core(load_cv(b_name, subset))
    ids = sorted(set(A) & set(B))
    a = np.array([A[i]["dice"] for i in ids]); b = np.array([B[i]["dice"] for i in ids])
    d = a - b
    lo, hi = boot_mean_ci(d)
    w = stats.wilcoxon(a, b)
    return {"n": len(ids), "mean_a": a.mean(), "mean_b": b.mean(), "mean_diff": d.mean(),
            "ci": (lo, hi), "median_diff": float(np.median(d)), "p": float(w.pvalue),
            "frac_a_ge_b": float((d >= 0).mean()), "frac_a_gt_b": float((d > 0).mean())}

pairs = {
    "final_vs_unet_pre": paired(RUN["final"], RUN["unet_pre"]),
    "final_vs_unet_scr": paired(RUN["final"], RUN["unet_scr"]),
    "final_vs_zeroshot": paired(RUN["final"], RUN["zeroshot"]),
    "skip_vs_original": paired(RUN["skip"], RUN["original"]),
    "final_vs_skip": paired(RUN["final"], RUN["skip"]),
    "final_vs_original": paired(RUN["final"], RUN["original"]),
    "seoff_vs_final": paired(RUN["se_off"], RUN["final"], True),
    "auxsdf_vs_final": paired(RUN["aux_sdf"], RUN["final"], True),
    "ftl_vs_final": paired(RUN["ftl"], RUN["final"], True),
    "dice_vs_final": paired(RUN["dice"], RUN["final"], True),
}
facts["pairs"] = pairs

# per-slide Dice and fold means
fold = {k: load_cv(RUN[k])["fold_dice"] for k in ("final", "unet_pre", "unet_scr", "zeroshot", "original", "skip")}
facts["fold_dice"] = fold
readers = [4.527777777777778, 5.0344827586206895, 4.535714285714286, 4.645161290322581,
           5.0227272727272725, 5.16]
facts["readers_r"] = {k: float(np.corrcoef(readers, v)[0, 1]) for k, v in fold.items()}

# validation means (selection criterion)
def val_mean(name):
    return float(np.mean([h["best_val_score"] for h in load_cv(name)["histories"]]))
facts["val_mean"] = {k: val_mean(RUN[k]) for k in ("original", "skip", "final", "unet_pre", "unet_scr")}

# learning curve
import csv
lc = list(csv.DictReader(open(os.path.join(RES, "data_efficiency.csv"))))
curve = {}
for r in lc:
    curve.setdefault(r["model"], {}).setdefault(int(r["n_train_cores"]), {})[int(r["fold"])] = float(r["val_dice"])
facts["curve"] = curve


# ------------------------------------------------------------------ helpers
TARGET_W_IN = 7.48   # 190 mm, full-page width
TARGET_DPI = 600


def save(fig, name):
    """Vector PDF (fonts embedded) plus 600 dpi TIFF and PNG at full-page width, no metadata."""
    import io
    name = os.path.splitext(name)[0]
    kw = dict(bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(os.path.join(OUT, name + ".pdf"), metadata={"Creator": None, "Producer": None,
                "CreationDate": None}, **kw)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100, **kw)
    w_in = Image.open(buf).size[0] / 100.0
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=TARGET_DPI * TARGET_W_IN / w_in, metadata={"Software": None}, **kw)
    im = Image.open(buf); im.load(); im = im.convert("RGB")
    if TIFF:
        im.save(os.path.join(OUT, name + ".tif"), compression="tiff_lzw", dpi=(TARGET_DPI, TARGET_DPI))
    im.save(os.path.join(OUT, name + ".png"), dpi=(TARGET_DPI, TARGET_DPI), optimize=True)
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec="#4a4a4a", fs=7.2, bold=False, lw=0.8, tc=INK, hatch=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.012",
                       fc=fc, ec=ec, lw=lw, hatch=hatch)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color="#4a4a4a", lw=0.9, style="-|>", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=8))


# ------------------------------------------------------------------ Figure 1: design
fig, ax = plt.subplots(figsize=(6.8, 3.7))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.0, 0.99, "(a) Leave-one-slide-out partition of the 193 labelled cores", fontsize=8.2,
        fontweight="bold", va="top")
ax.text(0.625, 0.99, "(b) Inside every fold", fontsize=8.2, fontweight="bold", va="top")
gx, gw, gy0, gh = 0.085, 0.084, 0.80, 0.088
for j, (s, n) in enumerate(zip(SLIDES, CORES)):
    ax.text(gx + j * gw + gw / 2, gy0 + 0.03, f"{s}\n{n} cores", ha="center", va="bottom",
            fontsize=6.4, color=INK, linespacing=1.15)
role_fc = {"Test": "#3f3f3f", "Val": "#9b9b9b", "Train": "#e4e4e4"}
role_tc = {"Test": "white", "Val": "white", "Train": INK}
for f in range(6):
    y = gy0 - (f + 1) * gh
    ax.text(gx - 0.01, y + gh / 2, f"Fold {f}", ha="right", va="center", fontsize=6.8)
    for j in range(6):
        role = "Test" if j == f else ("Val" if j == (f + 1) % 6 else "Train")
        ax.add_patch(Rectangle((gx + j * gw + 0.003, y + 0.008), gw - 0.006, gh - 0.016,
                               fc=role_fc[role], ec="none"))
        ax.text(gx + j * gw + gw / 2, y + gh / 2, role, ha="center", va="center",
                fontsize=6.2, color=role_tc[role])
ax.text(gx, gy0 - 6 * gh - 0.03,
        "Every core is tested exactly once (pooled n = 193).\n"
        "A test slide is scored once, after training and checkpoint selection.",
        fontsize=6.5, color=MUTED, va="top", linespacing=1.3)
rx, rw = 0.625, 0.37
steps = [("Training pool: the four remaining\nslides (118 to 136 cores)", "#f3f3f3", "#7f7f7f"),
         ("35 cores (fixed seed), 35 tumour-biased\n512 × 512 patches each: 1,225 patches", "#fbe9dc", "#C55A11"),
         ("Validation slide: early stopping,\nlearning-rate schedule, checkpoint choice", "#e9eef7", "#2F5597"),
         ("Test slide: 20 random patches per core,\nscored once (Dice, IoU, HD95, ASD)", "#ececec", "#3f3f3f")]
ys = [0.80, 0.625, 0.45, 0.275]
for (t, fc, ec), y in zip(steps, ys):
    box(ax, rx, y, rw, 0.115, t, fc, ec=ec, fs=6.4)
for y0, y1 in zip(ys[:-1], ys[1:]):
    arrow(ax, rx + rw / 2, y0, rx + rw / 2, y1 + 0.115)
box(ax, rx, 0.05, rw, 0.13, "Same folds, patches and scoring for every\nmodel: three FrozenSE encoders, two U-Nets,\nnnU-Net and zero-shot MobileSAM",
    "#ffffff", ec="#9b9b9b", fs=6.3)
ax.plot([0.60, 0.60], [0.04, 0.95], color="#d9d9d9", lw=0.8)
save(fig, "Figure1")

# ------------------------------------------------------------------ Figure 3: architecture
fig, ax = plt.subplots(figsize=(6.8, 4.9))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
FZ, FT, DEC = "#ededed", "#fbe1cf", "#dfe7f5"
ex, ew, eh = 0.01, 0.30, 0.07
dx, dw, dh = 0.565, 0.43, 0.118
rows = {"in": 0.90, "rs": 0.79, "pe": 0.655, "s1": 0.505, "s2": 0.395, "s3": 0.305, "s4": 0.215, "nk": 0.125}
enc = [("in", "Input patch: 3 × 512 × 512 (H&E)", "#ffffff"),
       ("rs", "Bilinear resize to 3 × 1024 × 1024;\nImageNet mean/SD normalisation", "#ffffff"),
       ("pe", "Patch embedding (frozen)\n64 × 256 × 256", FZ),
       ("s1", "TinyViT stage 1 (frozen)\n128 × 128 × 128", FZ),
       ("s2", "TinyViT stage 2 (frozen)\n160 × 64 × 64", FZ),
       ("s3", "TinyViT stage 3 (frozen)\n320 × 64 × 64", FZ),
       ("s4", "TinyViT stage 4 (fine-tuned)\n320 × 64 × 64", FT),
       ("nk", "Neck (fine-tuned)\n256 × 64 × 64", FT)]
for key, t, fc in enc:
    box(ax, ex, rows[key], ew, eh, t, fc, fs=6.4)
keys = [k for k, _, _ in enc]
for k0, k1 in zip(keys[:-1], keys[1:]):
    arrow(ax, ex + ew / 2, rows[k0], ex + ew / 2, rows[k1] + eh)
cy = {k: rows[k] + eh / 2 for k in rows}
drow = {"up1": cy["s1"] - dh / 2, "up2": cy["pe"] - dh / 2, "up3": 0.80}
dec = [("up1", "Up-stage 1: ConvTranspose 2 × 2, stride 2\n→ 192 × 128 × 128; concatenate skip 1\n2 × [Conv 3 × 3, BN, ReLU]; SE (r = 16)", DEC),
       ("up2", "Up-stage 2: ConvTranspose 2 × 2, stride 2\n→ 96 × 256 × 256; concatenate skip 2\n2 × [Conv 3 × 3, BN, ReLU]; SE (r = 16)", DEC),
       ("up3", "Up-stage 3: ConvTranspose 2 × 2, stride 2\n→ 48 × 512 × 512; 2 × [Conv 3 × 3, BN, ReLU]; SE\nHead: Conv 1 × 1 → 1 × 512 × 512, sigmoid", DEC)]
for key, t, fc in dec:
    box(ax, dx, drow[key], dw, dh, t, fc, fs=6.2)
cx = dx + dw / 2
for k0, k1 in (("up1", "up2"), ("up2", "up3")):
    arrow(ax, cx, drow[k0] + dh, cx, drow[k1])
arrow(ax, cx, drow["up3"] + dh, cx, 0.955)
ax.text(cx, 0.962, "Binary tumour mask (threshold 0.5), 512 × 512", ha="center", va="bottom", fontsize=6.6,
        fontweight="bold")
ny = cy["nk"]
ax.plot([ex + ew, cx], [ny, ny], color="#2F5597", lw=1.1)
arrow(ax, cx, ny, cx, drow["up1"], color="#2F5597", lw=1.1)
ax.text((ex + ew + cx) / 2, ny + 0.012, "encoder features 256 × 64 × 64", fontsize=6.4, color="#22427a",
        ha="center", va="bottom")
for src, lab in (("s1", "skip 1: stage-1 output, 128 ch"), ("pe", "skip 2: patch embedding, 64 ch")):
    y = cy[src]
    arrow(ax, ex + ew, y, dx, y, color="#C55A11", lw=1.1, ls="--")
    ax.text((ex + ew + dx) / 2, y + 0.012, lab, fontsize=6.3, color="#7a3a0b", ha="center", va="bottom")
ax.text(0.335, 0.40, "Encoder: MobileSAM TinyViT, 5.74 M parameters.\n"
        "Stage 4 and neck fine-tuned (3.15 M) at 5$\\times$10$^{-5}$;\n"
        "BatchNorm statistics fixed; stages 1 to 3 frozen.", fontsize=6.2, color=MUTED, va="top",
        linespacing=1.35)
ax.text(cx + 0.02, 0.40, "Decoder: 1.44 M parameters,\ntrained from scratch\nat 5$\\times$10$^{-4}$.", fontsize=6.2,
        color=MUTED, va="top", linespacing=1.35)
for i, (fc, t) in enumerate(((FZ, "frozen"), (FT, "fine-tuned"), (DEC, "trained from scratch"))):
    ax.add_patch(Rectangle((0.01 + i * 0.17, 0.02), 0.022, 0.03, fc=fc, ec="#4a4a4a", lw=0.6))
    ax.text(0.038 + i * 0.17, 0.035, t, fontsize=6.6, va="center")
ax.text(0.99, 0.035, "Total 7.19 M parameters, 4.59 M trainable", fontsize=6.6, va="center", ha="right")
save(fig, "Figure3")

# ------------------------------------------------------------------ Figure 5: development and ablations
fig = plt.figure(figsize=(6.8, 2.75))
gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.45], wspace=0.6)
a = fig.add_subplot(gs[0, 0])
steps = [("Original\ndesign", "original"), ("+ skip\nconnections", "skip"), ("+ stage-4\nfine-tuning", "final")]
vm = facts["val_mean"]
tm = {"original": float(load_cv(RUN["original"])["pooled_dice"]), "skip": float(load_cv(RUN["skip"])["pooled_dice"]),
      "final": float(load_cv(RUN["final"])["pooled_dice"])}
xs = np.arange(3)
a.plot(xs, [vm[k] * 100 for _, k in steps], "-o", color="#2F5597", lw=1.6, ms=5, mec="white", label="Validation (used to select)")
a.plot(xs, [tm[k] * 100 for _, k in steps], "-s", color=C["final"], lw=1.6, ms=5, mec="white", label="Test (reported only)")
for x0, (_, k) in zip(xs, steps):
    a.text(x0, vm[k] * 100 - 0.3, f"{vm[k]*100:.1f}", ha="center", va="top", fontsize=6.5)
    a.text(x0, tm[k] * 100 + 0.3, f"{tm[k]*100:.1f}", ha="center", va="bottom", fontsize=6.5)
a.set_xticks(xs); a.set_xticklabels([s for s, _ in steps], fontsize=6.6)
a.set_xlim(-0.35, 2.35)
a.set_ylabel("Dice (%)"); a.set_ylim(60.5, 67.5); a.grid(axis="y"); a.set_axisbelow(True)
a.legend(loc="upper left", frameon=False, fontsize=6.3)
a.set_title("(a) Development, 6 folds", fontsize=8)
a = fig.add_subplot(gs[0, 1])
abl = [("SE blocks removed", "seoff_vs_final", "0.30"), ("Auxiliary SDF head added", "auxsdf_vs_final", "<0.001"),
       ("Focal Tversky loss", "ftl_vs_final", "<0.001"), ("Dice loss", "dice_vs_final", "0.002")]
for i, (t, k, ph) in enumerate(abl):
    pz = pairs[k]; y = len(abl) - 1 - i
    m, lo, hi = pz["mean_diff"] * 100, pz["ci"][0] * 100, pz["ci"][1] * 100
    col = "#2F5597" if m > 0 else "#7f7f7f"
    a.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
    a.plot(m, y, "o", ms=6, color=col, mec="white", mew=1.1, zorder=3)
    a.text(1.75, y, f"{m:+.2f} ({lo:+.2f} to {hi:+.2f})", va="center", fontsize=6.4)
    a.text(3.55, y, ph, va="center", fontsize=6.4, ha="right")
a.text(1.75, 3.62, "Mean (95% CI)", fontsize=6.4, fontweight="bold", va="center")
a.text(3.55, 3.62, "Holm p", fontsize=6.4, fontweight="bold", va="center", ha="right")
a.axvline(0, color="#7f7f7f", lw=0.8)
a.set_yticks(range(len(abl))); a.set_yticklabels([t for t, _, _ in abl[::-1]], fontsize=6.8)
a.set_xlabel("Change in Dice vs final model (pp), 108 cores")
a.set_xlim(-2.0, 1.5); a.set_ylim(-0.5, 3.9); a.grid(axis="x"); a.set_axisbelow(True)
a.spines["right"].set_visible(False)
a.set_title("(b) One-change ablations, folds 0, 2, 4", fontsize=8)
save(fig, "Figure5")

T = json.load(open(os.path.join(TABLES, "summary.json")))
KEY = {"FrozenSE-UNI": "uni", "FrozenSE-Phikon": "phikon", "FrozenSE-SAM": "sam", "U-Net (ImageNet)": "unet_pre",
       "U-Net (scratch)": "unet_scr", "nnU-Net (2D)": "nnunet", "MobileSAM (zero-shot)": "zero"}
SUM = {KEY[r["model"]]: {"dice": [r["dice_pct"] / 100, r["dice_ci_low"] / 100, r["dice_ci_high"] / 100]}
       for r in T["main"] if r["model"] in KEY}
for k in SUM:
    SUM[k]["fold"] = T["fold_dice"][k]
PR = {f'{KEY[r["model_a"]]}_vs_{KEY[r["model_b"]]}': {"mean_diff": r["mean_diff_pp"] / 100,
      "ci": [r["ci_low_pp"] / 100, r["ci_high_pp"] / 100]} for r in T["paired"]
      if r["model_a"] in KEY and r["model_b"] in KEY}
C.update({"uni": "#A4243B", "phikon": "#D98E04", "nnunet": "#5C5C5C", "sam": C["final"], "zero": C["zeroshot"]})
NAME = {"uni": "FrozenSE-UNI", "phikon": "FrozenSE-Phikon", "sam": "FrozenSE-SAM", "unet_pre": "U-Net (ImageNet)",
        "unet_scr": "U-Net (scratch)", "nnunet": "nnU-Net (2D)", "zero": "MobileSAM (zero-shot)"}
MK = {"uni": "o", "phikon": "s", "sam": "D", "unet_pre": "^", "unet_scr": "v", "nnunet": "X", "zero": "P"}

# ------------------------------------------------------------------ Figure 4: main results
order = ["uni", "phikon", "unet_pre", "unet_scr", "sam", "nnunet", "zero"]
fig = plt.figure(figsize=(7.2, 3.05))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.3, 1.0], wspace=0.5)
a = fig.add_subplot(gs[0, 0])
for i, k in enumerate(order):
    m, lo, hi = [v * 100 for v in SUM[k]["dice"]]; y = len(order) - 1 - i
    a.plot([lo, hi], [y, y], color=C[k], lw=2, solid_capstyle="round")
    a.plot(m, y, MK[k], ms=6.5, color=C[k], mec="white", mew=1.1, zorder=3)
    a.text(hi + 0.8, y, f"{m:.1f}", va="center", fontsize=7, fontweight="bold" if k == "uni" else "normal")
a.set_yticks(range(len(order))); a.set_yticklabels([NAME[k].replace(" (", "\n(") for k in order][::-1], fontsize=6.6)
a.set_xlim(46, 80); a.set_ylim(-0.6, len(order) - 0.4); a.set_xlabel("Dice (%), 193 cores")
a.set_title("(a) Pooled Dice, 95% CI", fontsize=8); a.grid(axis="x"); a.set_axisbelow(True)
a = fig.add_subplot(gs[0, 1]); xs = np.arange(6); handles = []; shown = ["uni", "phikon", "sam", "unet_pre", "nnunet"]
for k in shown:
    v = np.array(SUM[k]["fold"]) * 100
    h, = a.plot(xs, v, "-", marker=MK[k], color=C[k], lw=2.0 if k == "uni" else 1.3, ms=4.6, mec="white", mew=0.8,
                label=NAME[k], zorder=4 if k == "uni" else 2)
    handles.append(h)
a.annotate("slide 7:\n+19.6 pp", xy=(5, SUM["uni"]["fold"][5] * 100), xytext=(3.25, 38.5), fontsize=6.4, color=C["uni"],
           arrowprops=dict(arrowstyle="-", color=C["uni"], lw=0.7))
a.set_xticks(xs); a.set_xticklabels(["1", "2", "3", "5", "6", "7"]); a.set_xlabel("Held-out test slide"); a.set_ylabel("Dice (%)")
a.set_ylim(30, 85); a.grid(axis="y"); a.set_axisbelow(True); a.set_title("(b) Dice by held-out slide", fontsize=8)
a = fig.add_subplot(gs[0, 2])
A, B = per_core(load_cv("pfm_uni")), per_core(load_cv(RUN["unet_pre"])); ids = sorted(A)
x = np.array([B[i]["dice"] for i in ids]) * 100; y = np.array([A[i]["dice"] for i in ids]) * 100
s7 = np.array([i.startswith("slide007") for i in ids])
a.plot([0, 100], [0, 100], color="#9b9b9b", lw=0.8, ls="--", zorder=1)
a.scatter(x[~s7], y[~s7], s=9, color=C["uni"], alpha=0.5, edgecolors="none", zorder=2, label="Other slides")
a.scatter(x[s7], y[s7], s=13, facecolors="none", edgecolors=INK, linewidths=0.7, zorder=3, label="Slide 7")
pf = PR["uni_vs_unet_pre"]
a.text(97, 4, f"mean difference +{pf['mean_diff']*100:.1f} pp\n95% CI {pf['ci'][0]*100:.1f} to {pf['ci'][1]*100:.1f}", fontsize=6.2,
       va="bottom", ha="right")
a.set_xlim(0, 100); a.set_ylim(0, 100); a.set_aspect("equal")
a.set_xlabel("U-Net (ImageNet), Dice %"); a.set_ylabel("FrozenSE-UNI, Dice %")
a.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, fontsize=6.2, handletextpad=0.2, borderpad=0.2)
a.set_title("(c) Paired per-core Dice", fontsize=8); a.grid(True); a.set_axisbelow(True)
fig.legend(handles=handles, labels=[NAME[k] for k in shown], loc="lower center", ncol=5, frameon=False, fontsize=7,
           bbox_to_anchor=(0.5, -0.12), handlelength=1.8, columnspacing=1.2)
save(fig, "Figure4")

# ------------------------------------------------------------------ Figure 6: five annotated cores
rows = list(csv.DictReader(open(os.path.join(RES, "lowdata_by_draw.csv"))))
d5 = {"sam": [float(r["frozense_dice"]) * 100 for r in rows], "unet_pre": [float(r["unet_dice"]) * 100 for r in rows]}
d35 = {"sam": SUM["sam"]["dice"][0] * 100, "unet_pre": SUM["unet_pre"]["dice"][0] * 100}
def fold5(tag):
    v = []
    for dr in (42, 1042, 2042):
        v.append(json.load(open(os.path.join(RES, f"lowdata5_{tag}_d{dr}", "cv_result.json")))["fold_dice"])
    return np.mean(v, axis=0) * 100
f5 = {"sam": fold5("frozense"), "unet_pre": fold5("unet")}
fig = plt.figure(figsize=(7.2, 2.6)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.3)
a = fig.add_subplot(gs[0, 0])
for k, off in (("sam", -0.07), ("unet_pre", 0.07)):
    m5 = np.mean(d5[k])
    a.plot([0 + off, 1 + off], [d35[k], m5], "-", color=C[k], lw=1.6)
    a.plot(0 + off, d35[k], MK[k], ms=6.5, color=C[k], mec="white", mew=1.1, label=NAME[k])
    a.plot(1 + off, m5, MK[k], ms=6.5, color=C[k], mec="white", mew=1.1)
    a.scatter([1 + off + 0.09] * 3, d5[k], s=10, color=C[k], alpha=0.45, edgecolors="none")
    a.text(0 + off - 0.07, d35[k], f"{d35[k]:.1f}", ha="right", va="center", fontsize=6.8)
    a.text(1 + off + 0.17, m5, f"{m5:.1f}  ({m5 / d35[k] * 100:.1f}% kept)", ha="left", va="center", fontsize=6.8)
a.set_xticks([0, 1]); a.set_xticklabels(["35 cores", "5 cores\n(3 draws)"]); a.set_xlabel("Annotated training cores"); a.set_xlim(-0.45, 2.15)
a.set_ylim(60, 70); a.set_ylabel("Test Dice (%), 193 cores"); a.grid(axis="y"); a.set_axisbelow(True)
a.legend(loc="lower left", frameon=False, fontsize=6.6); a.set_title("(a) Seven-fold cut in annotation", fontsize=8)
a = fig.add_subplot(gs[0, 1]); xs = np.arange(6); w = 0.36
v35 = np.array(SUM["sam"]["fold"]) * 100
a.bar(xs - w / 2, v35, w, color=C["sam"], edgecolor="white", linewidth=1, label="FrozenSE-SAM, 35 cores")
a.bar(xs + w / 2, f5["sam"], w, color="#E9B48F", edgecolor="white", linewidth=1, label="FrozenSE-SAM, 5 cores (mean of 3 draws)")
for x0, u, v in zip(xs, v35, f5["sam"]):
    a.text(x0 - w / 2, u + 0.8, f"{u:.0f}", ha="center", fontsize=6.3); a.text(x0 + w / 2, v + 0.8, f"{v:.0f}", ha="center", fontsize=6.3)
a.set_xticks(xs); a.set_xticklabels(["1", "2", "3", "5", "6", "7"]); a.set_xlabel("Held-out test slide"); a.set_ylabel("Dice (%)")
a.set_ylim(0, 100); a.grid(axis="y"); a.set_axisbelow(True); a.legend(loc="upper right", frameon=False, fontsize=6.4, ncol=1)
a.set_title("(b) Dice by held-out slide", fontsize=8)
save(fig, "Figure6")

# ------------------------------------------------------------------ Figure 7: cost
eff = {"sam": (4.59, 7.19, 112.1, 76.1), "phikon": (16.23, 87.85, 131.7, 47.1), "uni": (27.37, 305.52, 316.4, 97.9),
       "unet_pre": (24.44, 24.44, 38.7, 41.1), "unet_scr": (24.44, 24.44, 39.2, 43.8), "nnunet": (46.33, 46.33, None, 170.7)}
fig = plt.figure(figsize=(7.2, 2.55)); gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.2)
a = fig.add_subplot(gs[0, 0])
lab = {"sam": (1.5, 0, "left"), "phikon": (-1.5, 0, "right"), "uni": (-1.5, 0, "right"), "unet_pre": (0, -0.75, "center"),
       "unet_scr": (0, -0.75, "center"), "nnunet": (-1.5, 0, "right")}
for k in ("sam", "phikon", "uni", "unet_pre", "unet_scr", "nnunet"):
    dv = SUM[k]["dice"][0] * 100
    a.plot(eff[k][0], dv, MK[k], ms=7, color=C[k], mec="white", mew=1.1)
    dx, dy, ha = lab[k]
    if k == "unet_scr": continue
    t = "U-Nets" if k == "unet_pre" else NAME[k]
    a.text(eff[k][0] + (1.5 if k == "unet_pre" else dx), dv + (-0.35 if k == "unet_pre" else dy), t, fontsize=6.4, va="center", ha=("left" if k == "unet_pre" else ha))
a.set_xlabel("Trainable parameters (millions)"); a.set_ylabel("Dice (%), 193 cores"); a.set_xlim(0, 50); a.set_ylim(61.5, 73)
a.grid(True); a.set_axisbelow(True); a.set_title("(a) Accuracy vs trainable size", fontsize=8)
for j, (idx, t, unit, ks, xmax) in enumerate(((2, "(b) Inference latency", "ms per 512 × 512 patch", ["sam", "phikon", "uni", "unet_pre"], 380),
                                              (3, "(c) Training time", "minutes per fold", ["sam", "phikon", "uni", "unet_pre", "nnunet"], 205))):
    a = fig.add_subplot(gs[0, 1 + j]); ys = np.arange(len(ks))[::-1]; vals = [eff[k][idx] for k in ks]
    a.barh(ys, vals, height=0.6, color=[C[k] for k in ks], edgecolor="white", linewidth=1)
    for y0, v in zip(ys, vals): a.text(v + xmax * 0.02, y0, f"{v:.0f}", va="center", fontsize=6.6)
    a.set_yticks(ys); a.set_yticklabels([NAME[k].replace(" (2D)", "") for k in ks], fontsize=6.4)
    a.set_xlim(0, xmax); a.set_xlabel(unit, fontsize=7); a.grid(axis="x"); a.set_axisbelow(True); a.set_title(t, fontsize=8)
fig.subplots_adjust(wspace=0.2); axs = fig.axes
p1, p2 = axs[1].get_position(), axs[2].get_position()
axs[1].set_position([p1.x0 + 0.075, p1.y0, p1.width - 0.045, p1.height])
axs[2].set_position([p2.x0 + 0.105, p2.y0, p2.width - 0.045, p2.height])
save(fig, "Figure7")
print("figures written to", OUT)
