"""
Leave-one-slide-out cross-validation driver and aggregation.

WHY CROSS-VALIDATION RATHER THAN A SINGLE SPLIT
-----------------------------------------------
The labelled Gleason-2019 data in this project is 193 cores spread over six
glass slides (36, 29, 28, 31, 44, 25).  A single three-way group split would
therefore rest the entire result on which one or two slides happened to land
in the test partition -- with six groups that is a coin toss, not an estimate.

Holding out one slide per fold gives:

WHAT IT COSTS
-------------
Six training runs per model instead of one.  Run `folds=(0,)` first as a
smoke test, look at the number, then commit to the full six for anything that
goes in a table.  Do not report a single fold as the headline result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import time

import numpy as np

from .config import ExperimentConfig, replace_nested
from .data import (
    Core, PatchDataset, Split, make_split, n_folds as count_folds,
)
from .metrics import (
    CoreResult, EvaluationResult, holm_bonferroni, paired_wilcoxon,
)


# --------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------
def pool_folds(results: list[EvaluationResult],
               model_name: str | None = None) -> EvaluationResult:
    """Merge per-fold results into one EvaluationResult over all cores.

    Valid precisely because leave-one-group-out tests every core exactly
    once: the pooled core list has no duplicates, so the bootstrap and the
    paired tests in metrics.py operate on 193 independent cores.

    Raises if a core appears in two folds, which would mean the folds were
    not disjoint and any pooled statistic would be wrong.
    """
    if not results:
        raise ValueError("pool_folds() got no results")

    cores: list[CoreResult] = []
    patches = []
    seen: dict[str, int] = {}
    totals: dict[str, int] = {}

    for fold_idx, res in enumerate(results):
        for c in res.cores:
            if c.core_id in seen:
                raise AssertionError(
                    f"Core {c.core_id} was tested in fold {seen[c.core_id]} and "
                    f"again in fold {fold_idx}. Folds are not disjoint, so "
                    "pooled statistics would double-count it. Check the split "
                    "manifests before reporting anything."
                )
            seen[c.core_id] = fold_idx
            cores.append(c)
        patches.extend(res.patches)
        for k, v in res.case_counts.items():
            totals[k] = totals.get(k, 0) + v

    return EvaluationResult(
        model_name=model_name or results[0].model_name,
        patches=patches,
        cores=sorted(cores, key=lambda c: c.core_id),
        spacing_mm=results[0].spacing_mm,
        empty_mask_policy=results[0].empty_mask_policy,
        case_counts=totals,
    )


# --------------------------------------------------------------------------
@dataclass
class CVResult:
    """One model, evaluated across all folds."""

    model_name: str
    fold_results: list[EvaluationResult]
    fold_manifests: list[dict]
    fold_histories: list[dict]
    pooled: EvaluationResult
    seconds: float = 0.0
    efficiency: dict = field(default_factory=dict)

    # ---- across-fold statistics ----
    def fold_means(self, metric: str = "dice") -> list[float]:
        return [r.mean(metric) for r in self.fold_results]

    def fold_mean(self, metric: str = "dice") -> float:
        vals = [v for v in self.fold_means(metric) if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def fold_std(self, metric: str = "dice") -> float:
        vals = [v for v in self.fold_means(metric) if np.isfinite(v)]
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")

    def summary_row(self, cfg: ExperimentConfig) -> dict:
        """One table row: pooled point estimate, CI, and across-fold spread.

        Two spreads are reported and they answer different questions:
          `*_core_sd` is variation between cores (how uneven the model is);
          `*_fold_sd` is variation between held-out slides (how much the
          result depends on which staining batch you test on).
        The second is the one that should temper any claim of robustness.
        """
        row = self.pooled.summary_row(
            n_boot=cfg.eval.n_bootstrap,
            level=cfg.eval.ci_level,
            seed=cfg.eval.bootstrap_seed,
        )
        row["model"] = self.model_name
        row["n_folds"] = len(self.fold_results)
        for metric, label in [("dice", "Dice"), ("iou", "IoU"),
                              ("hd95_mm", "HD95_mm"), ("asd_mm", "ASD_mm")]:
            row[f"{label}_core_sd"] = row.pop(f"{label}_sd", self.pooled.std(metric))
            row[f"{label}_fold_mean"] = self.fold_mean(metric)
            row[f"{label}_fold_sd"] = self.fold_std(metric)
        row["per_fold_dice"] = [round(v, 4) for v in self.fold_means("dice")]
        row["held_out_slides"] = [m["groups"]["test"] for m in self.fold_manifests]
        row["train_cores_per_fold"] = [m["n_cores"]["train"] for m in self.fold_manifests]
        row["seconds"] = round(self.seconds, 1)
        if self.efficiency:
            row.update({f"eff_{k}": v for k, v in self.efficiency.items()})
        return row

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "n_folds": len(self.fold_results),
            "fold_dice": self.fold_means("dice"),
            "pooled_dice": self.pooled.mean("dice"),
            "pooled_n_cores": len(self.pooled.cores),
            "manifests": self.fold_manifests,
            "histories": self.fold_histories,
            "efficiency": self.efficiency,
            "per_core": [
                {"core_id": c.core_id, "dice": c.dice, "iou": c.iou,
                 "hd95_mm": c.hd95_mm, "asd_mm": c.asd_mm,
                 "n_patches": c.n_patches,
                 "n_defined_distance": c.n_defined_distance}
                for c in self.pooled.cores
            ],
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return path


# --------------------------------------------------------------------------
# Per-fold persistence
# --------------------------------------------------------------------------
FOLD_RESULT = "fold_result.json"

# Fields that say WHERE things are rather than WHAT was run. Excluding them
# lets a result produced on Kaggle be resumed on Colab, or the reverse,
# while any change to the experiment itself still breaks the match.
# Options added after results existed, with their original-behaviour default.
_ADDED_FIELDS = ((("train", "decoder_lr_mult"), 1.0),)

_LOCATION_FIELDS = {
    ("name",), ("output_root",),
    ("split", "fold"),
    ("data", "images_dir"), ("data", "labels_dir"),
    ("model", "sam_checkpoint"),
    ("train", "num_workers"),
}


def config_fingerprint(cfg: ExperimentConfig) -> str:
    """Short hash of everything that determines a fold's result."""
    import dataclasses
    import hashlib

    d = dataclasses.asdict(cfg)
    for path in _LOCATION_FIELDS:
        node = d
        for key in path[:-1]:
            node = node.get(key, {})
        node.pop(path[-1], None)
    for path, default in _ADDED_FIELDS:
        node = d
        for key in path[:-1]:
            node = node.get(key, {})
        if node.get(path[-1], default) == default:
            node.pop(path[-1], None)
    payload = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def save_fold_result(path: str, *, fingerprint: str, fold: int, model_name: str,
                     manifest: dict, history: dict, efficiency: dict,
                     result: EvaluationResult) -> str:
    """Write atomically, so a session killed mid-write leaves no torn file."""
    payload = {
        "fingerprint": fingerprint, "fold": fold, "model_name": model_name,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": manifest, "history": history, "efficiency": efficiency,
        "result": result.to_dict(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, default=str)
    os.replace(tmp, path)
    return path


def load_fold_result(path: str, fingerprint: str) -> dict | None:
    """Return the saved fold, None if absent, or raise if it is from a
    DIFFERENT configuration -- silently mixing those would corrupt results."""
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        payload = json.load(fh)
    if payload.get("fingerprint") != fingerprint:
        raise RuntimeError(
            f"{path} was produced by a different configuration "
            f"(saved {payload.get('fingerprint')}, current {fingerprint}).\n"
            "Pooling it with new folds would mix two experiments. If you "
            "changed a setting on purpose, give the experiment a new name; "
            "if not, restore the setting."
        )
    payload["result"] = EvaluationResult.from_dict(payload["result"])
    return payload


# --------------------------------------------------------------------------
def run_cross_validation(
    base_cfg: ExperimentConfig,
    cores: list[Core],
    folds: tuple[int, ...] | None = None,
    device=None,
    measure_eff: bool = True,
    verbose: bool = True,
    train_missing: bool = True,
    deadline: float | None = None,
    fold_seconds_estimate: float = 5400.0,
    save_summary: bool = True,
) -> CVResult | None:
    """Train and test one model configuration across folds -- resumably.

    Each fold gets its own run directory and its own TestSetGuard, so the
    test partition of each fold is read exactly once.  Nothing from the test
    partition reaches training or checkpoint selection.

    RESUMING. A fold's result is saved the moment it exists. On a re-run, a
    fold with a saved result is LOADED, not retrained and not re-tested --
    so a run cut off by a session limit continues where it stopped, and
    loading a saved result is not a second read of the test set. Saved
    results carry a fingerprint of the configuration; a mismatch raises
    rather than mixing two experiments.

    `train_missing=False` is report-only mode: nothing trains, and the
    function returns None if any requested fold has no saved result, so a
    report can never quietly start hours of training.

    `deadline` (a time.time() value) is the latest moment a fold may FINISH.
    A fold is only started if it is expected to finish in time, using the
    longest fold seen so far in this call, or `fold_seconds_estimate` before
    any has run. On Kaggle, a session killed at its 12-hour limit may never
    save its output, which would lose every fold it completed; stopping
    early instead ends the session cleanly with all finished folds saved,
    and the next session picks up the rest. Returns None if it stopped
    early, exactly as if those folds were missing.

    `folds=None` uses every available fold.  Pass `folds=(0,)` for a smoke
    test, but do not report a single fold as a result.
    """
    import gc

    available = count_folds(cores)
    fold_ids = tuple(range(available)) if folds is None else tuple(folds)
    fingerprint = config_fingerprint(base_cfg)

    fold_results: list[EvaluationResult] = []
    manifests: list[dict] = []
    histories: list[dict] = []
    efficiency: dict = {}
    missing: list[int] = []
    fold_durations: list[float] = []
    stopped_for_time = False
    n_loaded = 0
    t_start = time.time()

    torch = None
    for fold in fold_ids:
        cfg = replace_nested(
            base_cfg,
            name=f"{base_cfg.name}/fold{fold}",
            **{"split.fold": fold},
        )
        result_path = os.path.join(cfg.run_dir, FOLD_RESULT)

        saved = load_fold_result(result_path, fingerprint)
        if saved is not None:
            fold_results.append(saved["result"])
            manifests.append(saved["manifest"])
            histories.append(saved["history"])
            efficiency = efficiency or saved.get("efficiency") or {}
            n_loaded += 1
            if verbose:
                print(f"[{base_cfg.name}] fold {fold}: saved result loaded "
                      f"(test Dice {saved['result'].mean('dice'):.4f}, "
                      f"from {saved['saved_at']})")
            continue

        if not train_missing:
            missing.append(fold)
            continue

        if deadline is not None:
            needed = max(fold_durations) if fold_durations else fold_seconds_estimate
            if time.time() + needed > deadline:
                missing.append(fold)
                stopped_for_time = True
                continue

        # ---- this fold has to be computed ----
        fold_t0 = time.time()
        if torch is None:
            import torch
            from .engine import TestSetGuard, evaluate, fit, measure_efficiency
            from .losses import build_loss
            from .models import build_model, count_parameters
            device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        split: Split = make_split(cores, cfg.split)   # asserts disjointness
        manifest = split.manifest()

        if verbose:
            print(f"\n[{base_cfg.name}] fold {fold}: "
                  f"test {manifest['groups']['test']} "
                  f"({manifest['n_cores']['test']} cores), "
                  f"val {manifest['groups']['val']} "
                  f"({manifest['n_cores']['val']}), "
                  f"train {manifest['n_cores']['train']} cores "
                  f"from {manifest['n_groups']['train']} slides")

        os.makedirs(cfg.run_dir, exist_ok=True)
        split.save_manifest(os.path.join(cfg.run_dir, "split_manifest.json"))
        cfg.save()

        model = build_model(cfg.model, target_size=cfg.data.patch_size)

        trainable = any(p.requires_grad for p in model.parameters())
        if cfg.train.epochs > 0 and not trainable:
            raise ValueError(
                f"{base_cfg.name}: train.epochs={cfg.train.epochs} but the model "
                "has no trainable parameters. Set train.epochs=0 for a "
                "zero-shot baseline.")
        if cfg.train.epochs > 0:
            train_ds = PatchDataset(split.train, cfg.data, cfg.aug, train=True,
                                    seed=cfg.train.seed, verbose=verbose)
            val_ds = PatchDataset(split.val, cfg.data, cfg.aug, train=False,
                                  seed=cfg.train.seed, verbose=verbose)
            model, _, history = fit(model, train_ds, val_ds, cfg,
                                    device=device, verbose=verbose)
            # Free the training patches before the test set is loaded, so
            # peak memory is train + val, never train + val + test.
            del train_ds, val_ds
            gc.collect()
        else:
            # Zero-shot baseline: no training, so no checkpoint selection and
            # no validation pass. Recorded explicitly rather than inferred
            # from an empty history.
            model = model.to(device)
            history = {"zero_shot": True, "epochs_run": 0,
                       "best_val_score": None, "best_epoch": None,
                       "params": count_parameters(model)}

        test_ds = PatchDataset(split.test, cfg.data, cfg.aug, train=False,
                               seed=cfg.train.seed, verbose=verbose)

        # ---- the single permitted test-set read for this fold ----
        guard = TestSetGuard(cfg.run_dir)
        guard.claim(f"cross-validation fold {fold}, model {base_cfg.name}")

        criterion = build_loss(cfg.loss)
        result, _ = evaluate(model, _loader(test_ds, cfg), criterion, device,
                             cfg, model_name=base_cfg.name, compute_loss=False)

        if measure_eff and not efficiency:
            efficiency = measure_efficiency(model, cfg, device=device)

        # Saved IMMEDIATELY: from here on, this fold survives a disconnect.
        save_fold_result(result_path, fingerprint=fingerprint, fold=fold,
                         model_name=base_cfg.name, manifest=manifest,
                         history=history, efficiency=efficiency, result=result)

        fold_results.append(result)
        manifests.append(manifest)
        histories.append(history)

        fold_durations.append(time.time() - fold_t0)
        if verbose:
            print(f"  fold {fold} test Dice {result.mean('dice'):.4f} "
                  f"over {len(result.cores)} cores  [saved, "
                  f"{fold_durations[-1] / 60:.0f} min]", flush=True)

        del model, test_ds
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if missing:
        if verbose:
            if stopped_for_time:
                print(f"[{base_cfg.name}] stopped before folds {missing} to stay "
                      "inside the session time limit; every finished fold is "
                      "saved. Run this stage again to continue.", flush=True)
            else:
                print(f"[{base_cfg.name}] report-only: folds {missing} have no saved "
                      f"result yet ({len(fold_ids) - len(missing)} of {len(fold_ids)} "
                      "available). Run the training stage for this experiment first.")
        return None

    pooled = pool_folds(fold_results, model_name=base_cfg.name)
    cv = CVResult(
        model_name=base_cfg.name,
        fold_results=fold_results,
        fold_manifests=manifests,
        fold_histories=histories,
        pooled=pooled,
        seconds=time.time() - t_start,
        efficiency=efficiency,
    )

    if verbose:
        print(f"\n[{base_cfg.name}] pooled over {len(pooled.cores)} cores: "
              f"Dice {pooled.mean('dice'):.4f}  "
              f"(across-fold {cv.fold_mean('dice'):.4f} "
              f"+/- {cv.fold_std('dice'):.4f})"
              + (f"  [{n_loaded} of {len(fold_ids)} folds loaded from disk]"
                 if n_loaded else ""))

    # A subset of folds gets its own summary file, so that (for example) the
    # ablation reference loading folds 0/2/4 never overwrites the six-fold
    # summary of the same model.
    summary = ("cv_result.json" if set(fold_ids) == set(range(available)) else
               f"cv_result_folds{''.join(str(f) for f in sorted(fold_ids))}.json")
    if save_summary:
        cv.save(os.path.join(base_cfg.output_root, base_cfg.name, summary))
    return cv


# --------------------------------------------------------------------------
# Learning curve
# --------------------------------------------------------------------------
def check_curve_sizes(base_cfg: ExperimentConfig, cores, sizes, folds) -> None:
    """Raise if any learning-curve size exceeds any fold's training pool.

    Called before anything trains (and by the multi-GPU planner before any
    worker starts), so an impossible size fails in seconds rather than hours
    into a session.
    """
    for fold in folds:
        pool = make_split(cores, replace_nested(
            base_cfg, **{"split.fold": fold, "split.n_train_cores": None}).split).train_pool_size
        too_big = [n for n in sizes if n > pool]
        if too_big:
            raise ValueError(
                f"Learning-curve sizes {too_big} exceed fold {fold}'s training pool "
                f"of {pool} cores. Use sizes no larger than the smallest pool.")


def run_learning_curve(
    base_cfg: ExperimentConfig,
    cores: list[Core],
    sizes: tuple[int, ...] = (5, 10, 20, 35),
    folds: tuple[int, ...] = (0, 2, 4),
    device=None,
    train_missing: bool = True,
    deadline: float | None = None,
    run_seconds_estimate: float = 3600.0,
    verbose: bool = True,
) -> list[dict]:
    """Validation Dice as a function of the number of training cores.

    Measured on VALIDATION: the budgets are a design question, and each
    fold's test partition is already spent on the main comparison.

    Resumable like run_cross_validation: each run is saved as it finishes
    and loaded on re-runs. A size equal to the main experiment's training
    budget, on a fold the main comparison has already run, IS that training
    run -- same configuration, same fold -- so its saved history is reused
    instead of training it twice.

    Returns one row per (size, fold) that has a result; anything not yet
    run is listed in a printed note.
    """
    import gc

    rows: list[dict] = []
    missing: list[tuple[int, int]] = []
    durations: list[float] = []
    torch = None

    check_curve_sizes(base_cfg, cores, sizes, folds)

    for n in sizes:
        cfg_n = replace_nested(base_cfg, **{"split.n_train_cores": n})
        fp = config_fingerprint(cfg_n)
        for fold in folds:
            cfg = replace_nested(cfg_n, name=f"learning_curve/{base_cfg.name}/{n}cores/fold{fold}",
                                 **{"split.fold": fold})
            sp = make_split(cores, cfg.split)
            row = {"n_train_cores": n, "fold": fold,
                   "held_out_slide": sp.test_groups[0] if sp.test_groups else None}
            path = os.path.join(cfg.run_dir, "history.json")
            main_path = os.path.join(base_cfg.output_root, base_cfg.name,
                                     f"fold{fold}", FOLD_RESULT)

            history, source = None, None
            if n == base_cfg.split.n_train_cores and os.path.exists(main_path):
                history = load_fold_result(main_path, fp)["history"]
                source = "main experiment"
            elif os.path.exists(path):
                with open(path) as fh:
                    saved = json.load(fh)
                if saved.get("fingerprint") != fp:
                    raise RuntimeError(
                        f"{path} was produced by a different configuration; "
                        "give the experiment a new name or restore the setting.")
                history, source = saved["history"], "saved"

            if history is None:
                if not train_missing:
                    missing.append((n, fold))
                    continue
                if deadline is not None:
                    needed = max(durations) if durations else run_seconds_estimate
                    if time.time() + needed > deadline:
                        missing.append((n, fold))
                        continue
                if torch is None:
                    import torch
                    from .engine import fit
                    from .models import build_model
                    device = device or torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu")

                t0 = time.time()
                if verbose:
                    print(f"\n[learning curve] {n} training cores, fold {fold}", flush=True)
                tr = PatchDataset(sp.train, cfg.data, cfg.aug, train=True,
                                  seed=cfg.train.seed, verbose=verbose)
                va = PatchDataset(sp.val, cfg.data, cfg.aug, train=False,
                                  seed=cfg.train.seed, verbose=verbose)
                model = build_model(cfg.model, target_size=cfg.data.patch_size)
                model, _, history = fit(model, tr, va, cfg, device=device, verbose=verbose)
                os.makedirs(cfg.run_dir, exist_ok=True)
                with open(path + ".tmp", "w") as fh:
                    json.dump({"fingerprint": fp, "n_train_cores": n, "fold": fold,
                               "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "history": history}, fh, default=str)
                os.replace(path + ".tmp", path)
                source = "trained"
                durations.append(time.time() - t0)
                del model, tr, va
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            row.update({"val_dice": history.get("best_val_score"),
                        "epochs_run": history.get("epochs_run"), "source": source})
            rows.append(row)
            if verbose:
                print(f"  {n:>2} cores, fold {fold}: val Dice "
                      f"{row['val_dice']:.4f}  ({source})", flush=True)

    if missing and verbose:
        print(f"[learning curve] not yet run: {missing}. Run the "
              "'data_efficiency' stage (again) to continue.", flush=True)
    return rows


def _loader(dataset, cfg: ExperimentConfig):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False,
                      num_workers=cfg.train.num_workers, pin_memory=True)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def cv_summary_table(cvs: list[CVResult], cfg: ExperimentConfig) -> list[dict]:
    """One summary row per model."""
    return [cv.summary_row(cfg) for cv in cvs]


def cv_comparison_table(cvs: list[CVResult], reference: str,
                        metric: str = "dice") -> list[dict]:
    """Paired tests against `reference`, over the pooled per-core scores.

    Pairing is by core_id, and every model sees the same 193 cores on the
    same fixed evaluation patches, so the pairing is genuine rather than a
    comparison of two independent samples.  Holm-Bonferroni is applied across
    the family of comparisons.
    """
    ref = next((c for c in cvs if c.model_name == reference), None)
    if ref is None:
        raise ValueError(
            f"No model named {reference!r}; have {[c.model_name for c in cvs]}"
        )

    rows = [paired_wilcoxon(ref.pooled, c.pooled, metric)
            for c in cvs if c.model_name != reference]
    if rows:
        adjusted = holm_bonferroni([r["p_value"] for r in rows])
        for r, p in zip(rows, adjusted):
            r["p_holm"] = p
            r["significant_at_0.05"] = bool(np.isfinite(p) and p < 0.05)
    return rows


def format_cv_table(rows: list[dict]) -> str:
    """Fixed-width rendering of the main results table."""
    head = (f"{'model':<28}{'Dice':>8}{'95% CI':>18}{'core SD':>9}"
            f"{'fold SD':>9}{'HD95 mm':>9}{'cores':>7}")
    lines = [head, "-" * len(head)]
    for r in rows:
        ci = f"[{r.get('Dice_ci_low', float('nan')):.3f}, {r.get('Dice_ci_high', float('nan')):.3f}]"
        lines.append(
            f"{str(r['model'])[:27]:<28}"
            f"{r.get('Dice', float('nan')):>8.4f}"
            f"{ci:>18}"
            f"{r.get('Dice_core_sd', float('nan')):>9.4f}"
            f"{r.get('Dice_fold_sd', float('nan')):>9.4f}"
            f"{r.get('HD95_mm', float('nan')):>9.4f}"
            f"{r.get('n_cores', 0):>7}"
        )
    return "\n".join(lines)


def format_comparison(rows: list[dict]) -> str:
    head = (f"{'comparison':<44}{'n':>5}{'median diff':>13}"
            f"{'p':>10}{'p (Holm)':>10}")
    lines = [head, "-" * len(head)]
    for r in rows:
        label = f"{r['model_a']} vs {r['model_b']}"
        lines.append(
            f"{label[:43]:<44}"
            f"{r.get('n_pairs', 0):>5}"
            f"{r.get('median_difference', float('nan')):>13.4f}"
            f"{r.get('p_value', float('nan')):>10.4g}"
            f"{r.get('p_holm', float('nan')):>10.4g}"
        )
    return "\n".join(lines)
