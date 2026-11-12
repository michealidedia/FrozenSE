"""
Run one stage's training on several GPUs at once, one process per GPU.

    python -m frozense.runner --job JOB.json --worker I --n-workers N

The notebook writes JOB.json (the exact configuration, the core list, the
stage and the session deadline) and starts one worker per GPU, each with
CUDA_VISIBLE_DEVICES set to its own card. Every worker builds the SAME
ordered list of work units -- one (experiment, fold) pair each -- and the
same assignment of units to workers, so no coordination is needed: each
simply runs its own share. Results land in the usual per-fold files, so the
notebook then assembles the tables exactly as if it had trained everything
itself.

WHY. Kaggle's only working GPU option is "T4 x2" (its P100 fails with the
current PyTorch build), and the pipeline itself uses one GPU. Whatever the
quota counts -- session hours or GPU hours -- running a fold on each card
is never worse than leaving one idle, and is up to twice the experiments per
quota hour.

ASSIGNMENT. The notebook computes the plan ONCE, before any worker starts,
and writes each worker's share into JOB.json. Workers that each derived a
plan from what was already saved on disk could disagree -- if one finished a
unit before another started, the work would be split differently and units
run twice or skipped. Units are ordered most-important-first (for 'baselines', the
proposed model's folds come before the baselines', so a session cut short
leaves complete models rather than every model half done). They are dealt
to workers greedily by estimated cost, so the two cards finish at about the
same time, and each worker runs its share in priority order. Units already
saved cost nothing and are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .config import experiment_config_from_dict, replace_nested
from .crossval import (FOLD_RESULT, check_curve_sizes, config_fingerprint,
                       run_cross_validation, run_learning_curve)
from .data import Core, n_folds
from .experiments import (build_ablations, build_baselines, build_improvements,
                          build_loss_ablations)

# Relative training cost, used only to balance the workers.
COST = {
    "mobilesam_native": 0.05,            # zero-shot: no training
    "unet_pretrained": 0.7, "unet_scratch": 0.7,
    "ablation_encoder_unfrozen": 2.0,    # backward through the whole encoder
    "ablation_encoder_partial": 1.6,
    "improved_skip_partial": 1.7,
}


def _cost(cfg) -> float:
    return COST.get(cfg.name, COST.get(cfg.model.arch, 1.0))


def work_units(job: dict, base, cores) -> list[dict]:
    """The stage's work, as an ordered list of one-fold units."""
    stage = job["stage"]
    units = []
    if stage in ("baselines", "improve"):
        exps = build_baselines(base) if stage == "baselines" else build_improvements(base)
        for exp in exps:                                # model-major
            for fold in range(n_folds(cores)):
                units.append({"kind": "cv", "cfg": exp.cfg, "fold": fold,
                              "cost": _cost(exp.cfg), "measure_eff": True})
    elif stage == "ablations":
        folds = tuple(job["ablation_folds"])
        ref_fp = config_fingerprint(base)
        for fold in folds:                               # the reference first
            units.append({"kind": "cv", "cfg": base, "fold": fold,
                          "cost": 1.0, "measure_eff": False})
        exps = {e.cfg.name: e for e in build_ablations(base) + build_loss_ablations(base)}
        order = [n for n in job["ablation_priority"] if n in exps]
        order += [n for n in exps if n not in order]
        for name in order:
            cfg = exps[name].cfg
            if config_fingerprint(cfg) == ref_fp:
                continue                                 # identical to the reference
            for fold in folds:
                units.append({"kind": "cv", "cfg": cfg, "fold": fold,
                              "cost": _cost(cfg), "measure_eff": False})
    elif stage == "extra":
        # v1.6: whole models listed by the notebook (low-data draws, pathology
        # encoders), all six folds each, model-major so a cut-short session
        # leaves complete models.
        for m in job["extra_models"]:
            cfg = experiment_config_from_dict(m["cfg"])
            for fold in range(n_folds(cores)):
                units.append({"kind": "cv", "cfg": cfg, "fold": fold,
                              "cost": float(m.get("cost", 1.0)),
                              "measure_eff": bool(m.get("measure_eff", False))})
    elif stage == "data_efficiency":
        # Refuse impossible sizes here, in the notebook's planning step, so
        # the error appears once instead of from every worker.
        models = [experiment_config_from_dict(d) for d in job.get("curve_models", [])] or [base]
        full = 1225 + 213          # train + validation work per epoch at 35 cores
        for m in models:
            check_curve_sizes(m, cores, job["curve_sizes"], job["curve_folds"])
        for n in job["curve_sizes"]:               # smallest budgets first, both models
            for m in models:
                for fold in job["curve_folds"]:
                    units.append({"kind": "curve", "n": n, "fold": fold, "cfg": m,
                                  "cost": (n * 35 + 213) / full * _cost(m)})
    else:
        raise ValueError(f"stage {stage!r} has no GPU work")
    return units


def _done(unit: dict, base) -> bool:
    if unit["kind"] == "cv":
        cfg = unit["cfg"]
        return os.path.exists(os.path.join(cfg.output_root, cfg.name,
                                           f"fold{unit['fold']}", FOLD_RESULT))
    n, fold, cfg = unit["n"], unit["fold"], unit["cfg"]
    if n == cfg.split.n_train_cores and os.path.exists(os.path.join(
            cfg.output_root, cfg.name, f"fold{fold}", FOLD_RESULT)):
        return True
    return os.path.exists(os.path.join(cfg.output_root, "learning_curve", cfg.name,
                                       f"{n}cores", f"fold{fold}", "history.json"))


def unit_id(u: dict) -> str:
    if u["kind"] == "cv":
        return f"cv|{u['cfg'].name}|{u['fold']}"
    return f"curve|{u['cfg'].name}|{u['n']}|{u['fold']}"


def plan(job: dict, base, cores, n_workers: int) -> list[list[str]]:
    """Each worker's share, as unit ids, computed once by the notebook."""
    return [[unit_id(u) for u in share]
            for share in assign(work_units(job, base, cores), base, n_workers)]


def assign(units: list[dict], base, n_workers: int) -> list[list[dict]]:
    """Deal unfinished units, in order, to the least-loaded worker."""
    load = [0.0] * n_workers
    shares: list[list[dict]] = [[] for _ in range(n_workers)]
    for u in units:
        if _done(u, base):
            continue
        w = min(range(n_workers), key=lambda i: (load[i], i))
        shares[w].append(u)
        load[w] += u["cost"]
    return shares


def run(job: dict, worker: int, n_workers: int) -> int:
    import torch

    base = experiment_config_from_dict(job["base"])
    cores = [Core(**c) for c in job["cores"]]
    deadline = job.get("deadline")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    by_id = {unit_id(u): u for u in work_units(job, base, cores)}
    mine = [by_id[i] for i in job["shares"][worker]]
    tag = f"[worker {worker}]"
    print(f"{tag} {len(mine)} unit(s) on {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
          flush=True)

    per_cost: list[float] = []      # seconds per unit of cost, observed
    stopped = 0
    failed = 0
    for k, u in enumerate(mine):
        estimate = (max(per_cost) if per_cost else 5400.0) * max(u["cost"], 0.05)
        t0 = time.time()
        try:
            finished = _run_unit(u, k, mine, tag, cores, device, deadline, estimate)
        except Exception:                  # one broken experiment must not stop the rest
            import traceback
            traceback.print_exc()
            print(f"{tag} unit {unit_id(u)} FAILED; moving on to the next unit", flush=True)
            failed += 1
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        if finished:
            per_cost.append((time.time() - t0) / max(u["cost"], 0.05))
        else:
            stopped += 1

    print(f"\n{tag} done: {len(mine) - stopped - failed} unit(s) finished"
          + (f", {stopped} left for the next session (time limit)" if stopped else "")
          + (f", {failed} FAILED (traceback above)" if failed else ""),
          flush=True)
    return 1 if failed else 0


def _run_unit(u, k, mine, tag, cores, device, deadline, estimate) -> bool:
        if u["kind"] == "cv":
            label = f"{u['cfg'].name} fold {u['fold']}"
            print(f"\n{tag} ({k + 1}/{len(mine)}) {label}", flush=True)
            out = run_cross_validation(
                u["cfg"], cores, folds=(u["fold"],), device=device,
                measure_eff=u["measure_eff"], verbose=True, deadline=deadline,
                fold_seconds_estimate=estimate, save_summary=False)
            finished = out is not None
        else:
            label = f"learning curve {u['cfg'].name} {u['n']} cores fold {u['fold']}"
            print(f"\n{tag} ({k + 1}/{len(mine)}) {label}", flush=True)
            rows = run_learning_curve(
                u["cfg"], cores, sizes=(u["n"],), folds=(u["fold"],), device=device,
                deadline=deadline, run_seconds_estimate=estimate, verbose=True)
            finished = bool(rows)
        return finished


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--job", required=True)
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--n-workers", type=int, required=True)
    args = ap.parse_args(argv)
    with open(args.job) as fh:
        job = json.load(fh)
    return run(job, args.worker, args.n_workers)


if __name__ == "__main__":
    sys.exit(main())
