"""
Training and evaluation loops.

THE TEST-SET GUARD
------------------
`TestSetGuard` enforces in code that the test partition is read exactly once,
after all training and model selection is complete. Any second access raises,
and the guard's state is written into the run manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .losses import build_loss, SegmentationWithAuxSDF
from .metrics import MetricAccumulator, EvaluationResult
from .models import count_parameters


# --------------------------------------------------------------------------
class TestSetGuard:
    """Raises if the test set is evaluated more than once."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.path = os.path.join(run_dir, "test_access.json")
        self.accesses: list[str] = []
        if os.path.exists(self.path):
            with open(self.path) as fh:
                self.accesses = json.load(fh).get("accesses", [])

    def claim(self, reason: str) -> None:
        if self.accesses:
            raise RuntimeError(
                "The test set has already been evaluated for this run:\n"
                + "\n".join(f"  - {a}" for a in self.accesses)
                + "\n\nEvaluating it again would reintroduce selection bias. "
                "Start a new run directory, or "
                "delete test_access.json only if you are certain this is a "
                "fresh, legitimate evaluation."
            )
        stamp = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {reason}"
        self.accesses.append(stamp)
        os.makedirs(self.run_dir, exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump({"accesses": self.accesses}, fh, indent=2)


# --------------------------------------------------------------------------
@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_dice: float
    val_loss: float
    lr: float
    seconds: float


def _forward(model: nn.Module, images: torch.Tensor):
    out = model(images)
    if isinstance(out, tuple):
        return out
    return out, None


def train_one_epoch(model, loader, criterion, optimizer, device,
                    scaler=None, aux_sdf: bool = False) -> float:
    model.train()
    # A frozen encoder must stay in eval mode so its BatchNorm/dropout
    # statistics are not disturbed by the training loop.
    #
    # `getattr(model, "cfg", None)` rather than `model.cfg`: the U-Net
    # baselines are segmentation_models_pytorch modules, which DO expose
    # `.encoder` but have no `.cfg`, so touching `model.cfg` directly raised
    # AttributeError on every pretrained-baseline epoch.
    model_cfg = getattr(model, "cfg", None)
    if hasattr(model, "encoder") and model_cfg is not None:
        if getattr(model_cfg, "freeze_encoder", False):
            model.encoder.eval()
        else:
            # A trainable SAM encoder still keeps its BatchNorm statistics
            # from pretraining -- standard when fine-tuning with batches this
            # small, it keeps frozen-vs-unfrozen a comparison of WEIGHTS
            # only, and it makes activation checkpointing exact (see
            # models._Checkpointed).
            from .models import freeze_batchnorm_statistics
            freeze_batchnorm_statistics(model.encoder)

    total, n = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits, sdf = _forward(model, images)
            if aux_sdf and isinstance(criterion, SegmentationWithAuxSDF):
                loss = criterion(logits.float(), masks, sdf.float() if sdf is not None else None)
            else:
                loss = criterion(logits.float(), masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total += loss.item() * images.size(0)
        n += images.size(0)

    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg: ExperimentConfig,
             model_name: str, compute_loss: bool = True) -> tuple[EvaluationResult, float]:
    """Run the model over a loader and return metrics plus mean loss."""
    model.eval()
    spacing = cfg.data.pixel_spacing_mm
    diag = float(np.hypot(cfg.data.patch_size, cfg.data.patch_size) * spacing)

    acc = MetricAccumulator(
        model_name=model_name,
        spacing_mm=spacing,
        empty_mask_policy=cfg.eval.empty_mask_policy,
        patch_diagonal_mm=diag,
    )

    total, n = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits, _ = _forward(model, images)
        logits = logits.float()

        if compute_loss and criterion is not None:
            try:
                total += criterion(logits, masks).item() * images.size(0)
                n += images.size(0)
            except TypeError:
                pass

        probs = torch.sigmoid(logits).cpu().numpy()
        acc.add_batch(probs, masks.cpu().numpy(), batch["core_id"],
                      threshold=cfg.train.threshold)

    return acc.result(), (total / n if n else float("nan"))


# --------------------------------------------------------------------------
def fit(model, train_ds, val_ds, cfg: ExperimentConfig,
        device: torch.device | None = None, verbose: bool = True):
    """Train with early stopping on the VALIDATION split only.

    Returns (model with best weights restored, list of EpochLog, history dict).
    The test set is never touched here.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=True,
    )

    base_criterion = build_loss(cfg.loss)
    criterion = (SegmentationWithAuxSDF(base_criterion, cfg.model.aux_sdf_weight)
                 if cfg.model.aux_sdf_head else base_criterion)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError(
            f"{cfg.name}: the model has no trainable parameters, so there is "
            "nothing to fit. A zero-shot model (such as mobilesam_native) "
            "must run with train.epochs=0.")
    mult = float(getattr(cfg.train, "decoder_lr_mult", 1.0))
    encoder = getattr(model, "encoder", None)
    if mult == 1.0:
        groups = [{"params": params, "lr": cfg.train.lr}]
    elif encoder is not None and cfg.model.arch != "unet_scratch":
        enc_ids = {id(p) for p in encoder.parameters()}
        groups = [g for g in (
            {"params": [p for p in params if id(p) in enc_ids], "lr": cfg.train.lr},
            {"params": [p for p in params if id(p) not in enc_ids],
             "lr": cfg.train.lr * mult},
        ) if g["params"]]
    else:                      # nothing pretrained: every weight is new
        groups = [{"params": params, "lr": cfg.train.lr * mult}]
    if cfg.train.optimizer == "adamw":
        optimizer = torch.optim.AdamW(groups, lr=cfg.train.lr,
                                      weight_decay=cfg.train.weight_decay)
    elif cfg.train.optimizer == "adam":
        optimizer = torch.optim.Adam(groups, lr=cfg.train.lr,
                                     weight_decay=cfg.train.weight_decay)
    else:
        optimizer = torch.optim.SGD(groups, lr=cfg.train.lr, momentum=0.9,
                                    weight_decay=cfg.train.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=cfg.train.selection_mode,
        factor=cfg.train.scheduler_factor, patience=cfg.train.scheduler_patience,
        min_lr=cfg.train.min_lr,
    )
    scaler = torch.amp.GradScaler(device.type) if (cfg.train.amp and device.type == "cuda") else None

    best_score = -np.inf if cfg.train.selection_mode == "max" else np.inf
    best_state = None
    best_epoch = -1
    patience_left = cfg.train.early_stopping_patience
    logs: list[EpochLog] = []

    for epoch in range(cfg.train.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     device, scaler, cfg.model.aux_sdf_head)
        val_result, val_loss = evaluate(model, val_loader, base_criterion, device,
                                        cfg, cfg.name + "_val")
        score = val_result.mean(cfg.train.selection_metric)
        elapsed = time.time() - t0

        logs.append(EpochLog(epoch, train_loss, score, val_loss,
                             optimizer.param_groups[0]["lr"], elapsed))
        scheduler.step(score)

        improved = (score > best_score if cfg.train.selection_mode == "max"
                    else score < best_score)
        if improved:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.train.early_stopping_patience
        else:
            patience_left -= 1

        # Every epoch, flushed: unattended runs (Kaggle's Save & Run All) can
        # treat a long-silent cell as dead, and the line doubles as a progress
        # record in the saved output.
        if verbose:
            print(f"  epoch {epoch:3d}  train_loss {train_loss:.4f}  "
                  f"val_dice {score:.4f}  lr {max(g['lr'] for g in optimizer.param_groups):.2e}  "
                  f"{elapsed:4.0f}s{'  *' if improved else ''}", flush=True)

        if patience_left <= 0:
            if verbose:
                print(f"  early stopping at epoch {epoch} "
                      f"(best {best_score:.4f} @ epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = {
        "best_val_score": float(best_score),
        "best_epoch": int(best_epoch),
        "epochs_run": len(logs),
        "params": count_parameters(model),
        "epoch_logs": [vars(l) for l in logs],
    }
    return model, logs, history


# --------------------------------------------------------------------------
@torch.no_grad()
def measure_efficiency(model, cfg: ExperimentConfig,
                       device: torch.device | None = None,
                       n_warmup: int = 5, n_runs: int = 30) -> dict:
    """Trainable/total parameters, peak GPU memory, and inference latency.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Models take native patches and resize internally where they need to,
    # so latency is measured from the same input the data pipeline delivers.
    size = cfg.data.patch_size
    x = torch.randn(1, 3, size, size, device=device)

    for _ in range(n_warmup):
        model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    out = count_parameters(model)
    out.update({
        "latency_ms_mean": float(np.mean(times) * 1000),
        "latency_ms_std": float(np.std(times) * 1000),
        "throughput_patches_per_s": float(1.0 / np.mean(times)),
        "device": str(device),
    })
    if device.type == "cuda":
        out["peak_memory_mb"] = float(torch.cuda.max_memory_allocated() / 1024**2)
    return out
