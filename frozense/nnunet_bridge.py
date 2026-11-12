"""
nnU-Net v2 (2D) on exactly the same folds and patches as every other model (v1.6).

    python -m frozense.nnunet_bridge --job JOB.json --worker I --n-workers N

Raw and preprocessed data live in scratch space and are rebuilt per fold;
a fold whose result.json exists is skipped, so sessions resume cleanly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from .config import experiment_config_from_dict, replace_nested
from .data import Core, PatchDataset, make_split

MODEL_NAME = "nnunet_2d"


def _write_png(path, arr):
    from PIL import Image
    Image.fromarray(arr).save(path)


def export_fold(base, cores, fold: int, raw_root: str, dataset_id: int, verbose=True):
    """Write fold `fold` as nnU-Net raw dataset `dataset_id`. Returns (name, test_index)."""
    cfg = replace_nested(base, **{"split.fold": fold})
    split = make_split(cores, cfg.split)
    name = f"Dataset{dataset_id:03d}_GleasonFold{fold}"
    root = os.path.join(raw_root, name)
    shutil.rmtree(root, ignore_errors=True)
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        os.makedirs(os.path.join(root, sub))

    tr = PatchDataset(split.train, cfg.data, None, train=True, seed=cfg.train.seed, verbose=verbose)
    va = PatchDataset(split.val, cfg.data, None, train=False, seed=cfg.train.seed, verbose=verbose)
    te = PatchDataset(split.test, cfg.data, None, train=False, seed=cfg.train.seed, verbose=verbose)

    train_cases, val_cases = [], []
    for prefix, ds, bucket in (("tr", tr, train_cases), ("va", va, val_cases)):
        for i in range(len(ds)):
            img, msk = ds.raw_patch(i)
            case = f"{prefix}_{i:05d}"
            _write_png(os.path.join(root, "imagesTr", f"{case}_0000.png"), img)
            _write_png(os.path.join(root, "labelsTr", f"{case}.png"), (msk > 0).astype(np.uint8))
            bucket.append(case)
    test_index = []
    for i in range(len(te)):
        img, msk = te.raw_patch(i)
        case = f"te_{i:05d}"
        _write_png(os.path.join(root, "imagesTs", f"{case}_0000.png"), img)
        _write_png(os.path.join(root, "labelsTs", f"{case}.png"), (msk > 0).astype(np.uint8))
        ci, _ = te.index[i]
        test_index.append({"case": case, "core_id": te.cores[ci].core_id})

    with open(os.path.join(root, "dataset.json"), "w") as fh:
        json.dump({"channel_names": {"0": "R", "1": "G", "2": "B"},
                   "labels": {"background": 0, "tumour": 1},
                   "numTraining": len(train_cases) + len(val_cases),
                   "file_ending": ".png"}, fh, indent=1)
    with open(os.path.join(root, "frozense_split.json"), "w") as fh:
        json.dump({"train": train_cases, "val": val_cases, "test": test_index,
                   "manifest": split.manifest()}, fh, default=str)
    return name, train_cases, val_cases, test_index, split.manifest()


def _sh(cmd: list[str], env: dict, log) -> None:
    log.write("$ " + " ".join(cmd) + "\n"); log.flush()
    p = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        log.flush()
        with open(log.name, errors="replace") as fh:
            tail = "".join(fh.readlines()[-40:])
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n"
                           f"----- last lines of nnU-Net's output -----\n{tail}")


def score(base, pred_dir: str, raw_dir: str, test_index: list[dict]):
    """Score nnU-Net's test predictions with the code that scored every other model."""
    from PIL import Image
    from .metrics import MetricAccumulator
    spacing = base.data.pixel_spacing_mm
    diag = float(np.hypot(base.data.patch_size, base.data.patch_size) * spacing)
    acc = MetricAccumulator(MODEL_NAME, spacing_mm=spacing,
                            empty_mask_policy=base.eval.empty_mask_policy,
                            patch_diagonal_mm=diag)
    for item in test_index:
        pred = np.array(Image.open(os.path.join(pred_dir, item["case"] + ".png"))).astype(np.float32)
        gt = np.array(Image.open(os.path.join(raw_dir, "labelsTs", item["case"] + ".png"))).astype(np.float32)
        if pred.ndim == 3:
            pred = pred[..., 0]
        acc.add_batch((pred > 0).astype(np.float32)[None, None], gt[None, None], [item["core_id"]])
    return acc.result()


def run_fold(job: dict, fold: int, gpu_env: dict) -> bool:
    base = experiment_config_from_dict(job["base"])
    cores = [Core(**c) for c in job["cores"]]
    out_dir = os.path.join(job["output_root"], MODEL_NAME, f"fold{fold}")
    if os.path.exists(os.path.join(out_dir, "result.json")):
        print(f"fold {fold}: already done", flush=True)
        return True
    os.makedirs(out_dir, exist_ok=True)
    did = job["dataset_base_id"] + fold
    env = dict(gpu_env, nnUNet_raw=job["raw"], nnUNet_preprocessed=job["preprocessed"],
               nnUNet_results=job["results"], nnUNet_n_proc_DA=str(job.get("n_proc_da", 2)),
               nnUNet_compile="f")          # torch.compile off: Triton on T4 + new torch is fragile
    trainer = f"nnUNetTrainer_{job['epochs']}epochs"
    t0 = time.time()
    with open(os.path.join(out_dir, "nnunet_log.txt"), "a") as log:
        name, train_cases, val_cases, test_index, manifest = export_fold(
            base, cores, fold, job["raw"], did, verbose=False)
        print(f"fold {fold}: exported {len(train_cases)} train, {len(val_cases)} val, "
              f"{len(test_index)} test patches", flush=True)
        _sh(["nnUNetv2_plan_and_preprocess", "-d", str(did), "-c", "2d",
             "-np", str(job.get("n_proc_pre", 2))], env, log)
        with open(os.path.join(job["preprocessed"], name, "splits_final.json"), "w") as fh:
            json.dump([{"train": train_cases, "val": val_cases}], fh)
        _sh(["nnUNetv2_train", str(did), "2d", "0", "-tr", trainer], env, log)
        pred = os.path.join(job["scratch"], f"pred_fold{fold}")
        shutil.rmtree(pred, ignore_errors=True)
        _sh(["nnUNetv2_predict", "-i", os.path.join(job["raw"], name, "imagesTs"), "-o", pred,
             "-d", str(did), "-c", "2d", "-f", "0", "-tr", trainer, "--disable_tta"], env, log)
        result = score(base, pred, os.path.join(job["raw"], name), test_index)

    import glob
    trained = os.path.join(job["results"], name)
    for hit in (glob.glob(os.path.join(trained, "**", "fold_0", "progress.png"), recursive=True)
                + glob.glob(os.path.join(trained, "**", "fold_0", "training_log*.txt"), recursive=True)):
        shutil.copy(hit, out_dir)
    plans = glob.glob(os.path.join(trained, "**", "plans.json"), recursive=True)
    with open(os.path.join(out_dir, "result.json"), "w") as fh:
        json.dump({"model_name": MODEL_NAME, "fold": fold, "trainer": trainer,
                   "tta": False, "minutes": (time.time() - t0) / 60,
                   "manifest": manifest,
                   "plans": json.load(open(plans[0])) if plans else None,
                   "result": result.to_dict()}, fh, default=str)
    shutil.rmtree(trained, ignore_errors=True)                     # checkpoints: large, not needed
    shutil.rmtree(os.path.join(job["preprocessed"], name), ignore_errors=True)
    shutil.rmtree(os.path.join(job["raw"], name), ignore_errors=True)
    shutil.rmtree(pred, ignore_errors=True)
    print(f"fold {fold}: test Dice {result.mean('dice'):.4f} "
          f"({(time.time() - t0) / 60:.0f} min)", flush=True)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    a = ap.parse_args(argv)
    job = json.load(open(a.job))
    folds = [f for i, f in enumerate(job["folds"]) if i % a.n_workers == a.worker]
    per_fold = job.get("fold_hours_estimate", 3.0) * 3600
    for f in folds:
        if job.get("deadline") and time.time() + per_fold > job["deadline"]:
            print(f"fold {f}: not started, would pass the session limit; next session", flush=True)
            continue
        try:
            run_fold(job, f, dict(os.environ))
        except Exception as e:                                     # keep going with the next fold
            print(f"fold {f}: FAILED: {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
