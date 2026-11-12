"""Resumable cross-validation: save, reload, refuse to mix.

Exercises the persistence layer and the resume path of run_cross_validation
without a GPU: when every requested fold already has a saved result, the
driver must load them, never train, never touch the test set again, and
pool them exactly as if they had just been computed.

Run:  python3 -I tests/test_resume.py
"""

import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np                                              # noqa: E402

from frozense.config import ExperimentConfig, replace_nested    # noqa: E402
from frozense.data import Core, make_split, n_folds             # noqa: E402
from frozense.metrics import CoreResult, EvaluationResult, PatchResult  # noqa: E402
from frozense.crossval import (                                 # noqa: E402
    FOLD_RESULT, config_fingerprint, save_fold_result, load_fold_result,
    run_cross_validation, pool_folds,
)

TEST_ROOT = os.environ.get("FROZENSE_TEST_DIR",
                           os.path.join(tempfile.gettempdir(), "frozense_tests"))
ROOT = os.path.join(TEST_ROOT, "resume_test")
REAL = {"slide001": 36, "slide002": 29, "slide003": 28,
        "slide005": 31, "slide006": 44, "slide007": 25}


def cores():
    return [Core(f"{s}_core{i:03d}", f"/i/{s}_core{i:03d}.jpg",
                 f"/l/{s}_core{i:03d}.png", s)
            for s, n in REAL.items() for i in range(1, n + 1)]


def fake_result(ids, rng):
    cr, pr = [], []
    for k, cid in enumerate(ids):
        d = float(rng.uniform(0.3, 0.9))
        undefined = k % 9 == 0
        cr.append(CoreResult(cid, 20, d, d / (2 - d),
                             float("nan") if undefined else 0.1,
                             float("nan") if undefined else 0.03,
                             0 if undefined else 20, {"both_present": 20}))
        pr.append(PatchResult(cid, d, d / (2 - d),
                              float("nan") if undefined else 0.1,
                              float("nan") if undefined else 0.03,
                              "gt_not_pred_empty" if undefined else "both_present"))
    return EvaluationResult("proposed", pr, cr, 0.00025, "report_separately",
                            {"both_present": 20 * len(ids)})


def check(label, ok, detail=""):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        raise AssertionError(label + " " + detail)


def main():
    shutil.rmtree(ROOT, ignore_errors=True)
    cs = cores()
    base = ExperimentConfig(name="proposed", output_root=ROOT)
    fp = config_fingerprint(base)

    print("fingerprint")
    moved = replace_nested(base, output_root="/kaggle/working/frozense_runs",
                           **{"data.images_dir": ("/kaggle/input/x/imagesTr",),
                              "model.sam_checkpoint": "/kaggle/input/w/mobile_sam.pt",
                              "train.num_workers": 4, "split.fold": 3})
    check("unchanged by where files live (Colab vs Kaggle paths, workers, fold)",
          config_fingerprint(moved) == fp)
    for key, val in [("loss.tversky_alpha", 0.7), ("train.lr", 3e-4),
                     ("model.use_se", False), ("split.n_train_cores", 20)]:
        check(f"changed by a real setting: {key}",
              config_fingerprint(replace_nested(base, **{key: val})) != fp)

    print("\nround trip")
    rng = np.random.default_rng(0)
    res = fake_result([c.core_id for c in cs[:30]], rng)
    path = f"{ROOT}/x/{FOLD_RESULT}"
    save_fold_result(path, fingerprint=fp, fold=0, model_name="proposed",
                     manifest={"m": 1}, history={"epochs_run": 40},
                     efficiency={"trainable": 123}, result=res)
    raw = json.load(open(path))
    check("written as standard JSON (NaN stored as null)",
          "NaN" not in open(path).read() and raw["result"]["cores"][0]["hd95_mm"] is None)
    back = load_fold_result(path, fp)["result"]
    check("every core survives the round trip", len(back.cores) == len(res.cores))
    check("Dice identical after reload", back.mean("dice") == res.mean("dice"))
    check("undefined HD95 comes back as NaN, still excluded from the mean",
          np.isnan(back.cores[0].hd95_mm) and back.mean("hd95_mm") == res.mean("hd95_mm"))
    check("patch-level records survive too", len(back.patches) == len(res.patches))
    check("no temporary file left behind", not os.path.exists(path + ".tmp"))
    try:
        load_fold_result(path, "0" * 16)
        check("mismatched fingerprint is refused", False)
    except RuntimeError as exc:
        check("mismatched fingerprint is refused", "different configuration" in str(exc))

    print("\nresume: all six folds saved")
    expected = []
    for fold in range(n_folds(cs)):
        cfg = replace_nested(base, name=f"proposed/fold{fold}", **{"split.fold": fold})
        sp = make_split(cs, cfg.split)
        r = fake_result([c.core_id for c in sp.test], rng)
        expected.append(r)
        save_fold_result(os.path.join(cfg.run_dir, FOLD_RESULT), fingerprint=fp,
                         fold=fold, model_name="proposed", manifest=sp.manifest(),
                         history={"epochs_run": 50 + fold},
                         efficiency={"trainable": 999} if fold == 0 else {},
                         result=r)
        # A fold that was saved also had its test set claimed once.
        json.dump({"accesses": [f"fold {fold}"]},
                  open(os.path.join(cfg.run_dir, "test_access.json"), "w"))

    torch_before = "torch" in sys.modules
    cv = run_cross_validation(base, cs, verbose=False, train_missing=True)
    check("all six folds loaded, nothing trained", cv is not None and len(cv.fold_results) == 6)
    check("PyTorch never imported to do it",
          torch_before or "torch" not in sys.modules)
    check("pooled over all 193 cores", len(cv.pooled.cores) == 193)
    check("pooled Dice equals pooling the saved results directly",
          cv.pooled.mean("dice") == pool_folds(expected, "proposed").mean("dice"))
    check("efficiency restored from the fold that measured it",
          cv.efficiency.get("trainable") == 999)
    check("loading did not count as a second test-set read",
          all(len(json.load(open(f"{ROOT}/proposed/fold{f}/test_access.json"))["accesses"]) == 1
              for f in range(6)))

    print("\nreport-only mode with a fold missing")
    os.remove(f"{ROOT}/proposed/fold4/{FOLD_RESULT}")
    out = run_cross_validation(base, cs, verbose=True, train_missing=False)
    check("returns None rather than training", out is None)

    print("\na changed setting is caught, not mixed in")
    changed = replace_nested(base, **{"loss.tversky_alpha": 0.7})
    try:
        run_cross_validation(changed, cs, folds=(0,), verbose=False, train_missing=False)
        check("old results under a changed config raise", False)
    except RuntimeError as exc:
        check("old results under a changed config raise", "different configuration" in str(exc))

    shutil.rmtree(ROOT, ignore_errors=True)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
