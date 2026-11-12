"""End-to-end smoke test of the REAL pipeline on CPU.

Runs every model architecture, every ablation and every loss through the
actual training loop, then cross-validation with resume, the results
tables and the efficiency measurement -- on a miniature copy of the real
folder layout (imagesTr / imagesVal / Generated_Labels_Train / labelsVal /
annotations/Map*). Nothing is mocked except the data, which is synthetic.

Its purpose is to make anything that would crash on the first GPU run crash
here instead, before free-tier quota is spent on it.

Run:  python3 tests/e2e_cpu_smoke.py path/to/mobile_sam.pt
"""

import os
import shutil
import sys
import tempfile
import time
import traceback
import warnings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
from PIL import Image                                           # noqa: E402

from frozense.config import (                                   # noqa: E402
    ExperimentConfig, SplitConfig, DataConfig, replace_nested)
from frozense.data import (                                     # noqa: E402
    discover_cores, make_split, n_folds, PatchDataset, audit_label_values)
from frozense.models import build_model, count_parameters       # noqa: E402
from frozense.engine import fit, measure_efficiency             # noqa: E402
from frozense.crossval import (                                 # noqa: E402
    run_cross_validation, cv_summary_table, cv_comparison_table,
    format_cv_table, format_comparison, FOLD_RESULT)
from frozense.experiments import build_ablations, build_loss_ablations  # noqa: E402

TEST_ROOT = os.environ.get("FROZENSE_TEST_DIR",
                           os.path.join(tempfile.gettempdir(), "frozense_tests"))
ROOT = os.path.join(TEST_ROOT, "e2e")
SLIDES = ["slide001", "slide002", "slide003", "slide005", "slide006", "slide007"]
SIDE = 576
RESULTS: list[tuple[str, bool, str]] = []


def record(label, ok, detail=""):
    RESULTS.append((label, ok, detail))
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""),
          flush=True)


def build_dataset():
    shutil.rmtree(ROOT, ignore_errors=True)
    dirs = {k: f"{ROOT}/data/{k}" for k in
            ("imagesTr", "imagesVal", "Generated_Labels_Train", "labelsVal")}
    for d in dirs.values():
        os.makedirs(d)
    rng = np.random.default_rng(11)
    for s, slide in enumerate(SLIDES):
        for c, (img_dir, lbl_dir) in enumerate([("imagesTr", "Generated_Labels_Train"),
                                                ("imagesVal", "labelsVal")]):
            cid = f"{slide}_core{c + 1:03d}"
            # Pink-purple H&E-ish texture with a darker "tumour" region.
            img = rng.normal((200, 140, 180), 18, size=(SIDE, SIDE, 3))
            lbl = np.zeros((SIDE, SIDE), np.uint8)
            y0, x0 = 60 + 20 * s, 80 + 30 * c
            lbl[y0:y0 + 300, x0:x0 + 260] = 4 if (s + c) % 2 else 3
            lbl[:, :40] = 1                                 # benign tissue
            img[lbl >= 3] -= 50
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(
                f"{dirs[img_dir]}/{cid}.jpg", quality=92)
            Image.fromarray(lbl).save(f"{dirs[lbl_dir]}/{cid}_classimg_nonconvex.png")
    return dirs


def tiny(cfg, **kw):
    """Shrink a config to CPU size without changing any code path."""
    over = {"train.epochs": 1, "train.batch_size": 2, "train.num_workers": 0,
            "train.early_stopping_patience": 1,
            "data.patches_per_core_train": 2, "data.patches_per_core_eval": 2,
            "split.n_train_cores": 3}
    over.update(kw)
    return replace_nested(cfg, **over)


def step(title):
    print(f"\n=== {title} ===", flush=True)
    return time.time()


def main():
    ckpt = sys.argv[1]
    torch.manual_seed(0)
    dirs = build_dataset()

    t = step("discovery, in place across both source folders")
    base = ExperimentConfig(
        name="e2e_proposed",
        output_root=f"{ROOT}/runs",
        split=SplitConfig(scheme="leave_one_group_out", seed=42),
        data=DataConfig(images_dir=(dirs["imagesTr"], dirs["imagesVal"]),
                        labels_dir=(dirs["Generated_Labels_Train"], dirs["labelsVal"])),
    )
    base = replace_nested(base, **{"model.sam_checkpoint": ckpt})
    base = tiny(base)
    cores = discover_cores(base.data, base.split)
    record("12 cores over 6 slides found without copying",
           len(cores) == 12 and n_folds(cores) == 6, f"{len(cores)} cores")
    audit_label_values(cores, base.data)
    record("label audit passes (benign grade excluded)", True)
    split = make_split(cores, base.split)
    record("fold 0 split: 3 train cores, 1 val slide, 1 test slide",
           len(split.train) == 3 and len(split.val_groups) == 1)

    t = step("dataset: shapes and dtypes the model will see")
    ds = PatchDataset(split.train, base.data, base.aug, train=True, seed=42, verbose=True)
    item = ds[0]
    record("image 3x512x512 float32", tuple(item["image"].shape) == (3, 512, 512)
           and item["image"].dtype == torch.float32, str(tuple(item["image"].shape)))
    record("mask 1x512x512 float32 (native resolution, not 1024)",
           tuple(item["mask"].shape) == (1, 512, 512) and item["mask"].dtype == torch.float32,
           str(tuple(item["mask"].shape)))
    from torch.utils.data import DataLoader
    batch = next(iter(DataLoader(ds, batch_size=2, num_workers=2)))
    record("forked DataLoader workers read preloaded patches",
           tuple(batch["image"].shape) == (2, 3, 512, 512))

    # ------------------------------------------------------------ variants
    variants = [("proposed (real decoder size)", base)]
    small = {"model.decoder_channels": (24, 12, 6)}
    for exp in build_ablations(base):
        variants.append((exp.cfg.name, tiny(exp.cfg, **small)))
    for exp in build_loss_ablations(base):
        variants.append((exp.cfg.name, tiny(exp.cfg, **small)))
    variants.append(("baseline_unet_scratch",
                     tiny(replace_nested(base, **{"model.arch": "unet_scratch"}))))
    variants.append(("baseline_unet_pretrained",
                     tiny(replace_nested(base, **{"model.arch": "unet_pretrained"}))))

    params = {}
    t = step(f"one training epoch for each of {len(variants)} variants (fold 0)")
    for label, cfg in variants:
        t0 = time.time()
        try:
            sp = make_split(cores, cfg.split)
            tr = PatchDataset(sp.train, cfg.data, cfg.aug, train=True, seed=cfg.train.seed)
            va = PatchDataset(sp.val, cfg.data, cfg.aug, train=False, seed=cfg.train.seed)
            model = build_model(cfg.model, target_size=cfg.data.patch_size)
            with torch.no_grad():
                out, sdf = model.eval()(torch.zeros(1, 3, 512, 512))
            assert tuple(out.shape) == (1, 1, 512, 512), f"logits {tuple(out.shape)}"
            if cfg.model.aux_sdf_head:
                assert sdf is not None and tuple(sdf.shape) == (1, 1, 512, 512)
            model, _, h = fit(model, tr, va, cfg, device=torch.device("cpu"), verbose=False)
            params[label] = count_parameters(model)
            ok = h["epochs_run"] == 1 and np.isfinite(h["best_val_score"])
            record(f"{label}", ok,
                   f"val Dice {h['best_val_score']:.3f}, trainable "
                   f"{params[label]['trainable']:,}, {time.time() - t0:.0f}s")
        except Exception as exc:
            net = "pretrained" in label and any(s in str(exc).lower() for s in
                                                ("connect", "url", "http", "resolve", "hub", "403"))
            if net:
                record(f"{label}", True, "SKIPPED here: ImageNet weights need internet "
                                         "(available on Kaggle/Colab); same code path as scratch")
            else:
                record(f"{label}", False, f"{type(exc).__name__}: {exc}")
                traceback.print_exc()

    t = step("ablation switches actually change the model")
    p = lambda k: params.get(k, {}).get("trainable", -1)
    total = lambda k: params.get(k, {}).get("total", -1)
    ref = small_ref = None
    try:
        small_ref = tiny(base, **small)
        m = build_model(small_ref.model, 512)
        small_ref = count_parameters(m)
    except Exception as exc:
        record("reference small model builds", False, str(exc))
    if small_ref:
        record("SE off has fewer trainable parameters", p("ablation_se_off") < small_ref["trainable"],
               f"{p('ablation_se_off'):,} vs {small_ref['trainable']:,}")
        record("skip connections add decoder parameters (switch is wired)",
               p("ablation_skip_on") > small_ref["trainable"],
               f"{p('ablation_skip_on'):,} vs {small_ref['trainable']:,}")
        record("unfrozen encoder trains the whole encoder",
               p("ablation_encoder_unfrozen") >= small_ref["trainable"] + 5_700_000,
               f"{p('ablation_encoder_unfrozen'):,}")
        record("frozen encoder count excludes TinyViT's unused classifier head",
               small_ref["total"] - small_ref["trainable"] == 6_065_532 - 321_640,
               f"{small_ref['total'] - small_ref['trainable']:,} frozen")
        record("partial unfreeze sits between frozen and unfrozen",
               small_ref["trainable"] < p("ablation_encoder_partial") < p("ablation_encoder_unfrozen"),
               f"{p('ablation_encoder_partial'):,}")
        record("aux SDF adds a head", p("ablation_aux_sdf") > small_ref["trainable"])
        record("partial unfreeze trains about half the encoder",
               0.45 < (p("ablation_encoder_partial") - small_ref["trainable"]) / 5_743_892 < 0.6,
               f"{(p('ablation_encoder_partial') - small_ref['trainable']) / 5_743_892:.0%}")

    t = step("cross-validation, 2 folds, then resume")
    cv_cfg = tiny(base, **small)
    t0 = time.time()
    cv = run_cross_validation(cv_cfg, cores, folds=(0, 1), device=torch.device("cpu"),
                              verbose=False)
    record("2 folds trained, tested and saved",
           cv is not None and len(cv.fold_results) == 2
           and all(os.path.exists(f"{cv_cfg.run_dir}/fold{f}/{FOLD_RESULT}") for f in (0, 1)),
           f"{time.time() - t0:.0f}s")
    t0 = time.time()
    again = run_cross_validation(cv_cfg, cores, folds=(0, 1), device=torch.device("cpu"),
                                 verbose=False)
    record("re-run loads both folds instead of retraining",
           again is not None and time.time() - t0 < 5
           and again.pooled.mean("dice") == cv.pooled.mean("dice"),
           f"{time.time() - t0:.1f}s")
    record("efficiency measured and kept", bool(cv.efficiency.get("trainable")),
           f"latency {cv.efficiency.get('latency_ms_mean', 0):.0f} ms on CPU")

    # epochs=0 AFTER tiny(): tiny() sets epochs=1, which would (correctly)
    # be refused for a model with nothing to train.
    zs_cfg = replace_nested(tiny(base), name="e2e_mobilesam_native",
                            **{"model.arch": "mobilesam_native", "train.epochs": 0})
    zs = run_cross_validation(zs_cfg, cores, folds=(0, 1), device=torch.device("cpu"),
                              verbose=False)
    record("zero-shot MobileSAM baseline runs through the same driver",
           zs is not None and len(zs.pooled.cores) == len(cv.pooled.cores),
           f"Dice {zs.pooled.mean('dice'):.3f}" if zs else "")

    try:
        run_cross_validation(replace_nested(zs_cfg, name="e2e_bad", **{"train.epochs": 1}),
                             cores, folds=(0,), device=torch.device("cpu"), verbose=False)
        record("epochs>0 on a model with nothing to train is refused clearly", False)
    except ValueError as exc:
        record("epochs>0 on a model with nothing to train is refused clearly",
               "no trainable parameters" in str(exc))

    t = step("results tables")
    rows = cv_summary_table([cv, zs], cv_cfg)
    print(format_cv_table(rows))
    comp = cv_comparison_table([cv, zs], reference=cv_cfg.name)
    print(format_comparison(comp))
    record("main table and paired test produced", len(rows) == 2 and len(comp) == 1)

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)} of {len(RESULTS)} checks passed")
    if failed:
        print("FAILED:")
        for label, _, detail in failed:
            print(f"  - {label}: {detail}")
        sys.exit(1)
    shutil.rmtree(ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
