"""
FrozenSE: a parameter-efficient adaptation framework for frozen foundation
encoders in prostate cancer histopathology segmentation.

Modules
    config       dataclass configuration of split, data, model, loss, training
    data         core index, leave-one-slide-out split, patch sampling, label audit
    consensus    per-pathologist maps: coverage, agreement, consensus reconstruction
    models       FrozenSE-SAM, the U-Net baselines, zero-shot MobileSAM, build_model()
    pathology    FrozenSE-Phikon and FrozenSE-UNI (the same decoder on pathology ViTs)
    losses       Tversky + focal objective, focal Tversky, Dice, auxiliary SDF loss
    engine       training loop, evaluation, TestSetGuard, efficiency measurement
    metrics      Dice, IoU, HD95, ASD with explicit empty-mask handling; core-level
                 aggregation, bootstrap CIs, paired Wilcoxon tests
    crossval     cross-validation with per-fold saving and resume; pooling of folds
    runner       one worker per GPU, used by the notebooks
    nnunet_bridge  nnU-Net v2 on exactly the same folds and patches

Quick start: see README.md and docs/REPRODUCE.md.
"""

__version__ = "1.6.2"

from .config import (
    ExperimentConfig, SplitConfig, DataConfig, AugConfig,
    ModelConfig, LossConfig, TrainConfig, EvalConfig, replace_nested,
)

__all__ = [
    "ExperimentConfig", "SplitConfig", "DataConfig", "AugConfig",
    "ModelConfig", "LossConfig", "TrainConfig", "EvalConfig", "replace_nested",
]
