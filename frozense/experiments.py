"""The experiment matrix.

ORDER MATTERS.  Everything here trains on the TRAIN split and selects on the
VALIDATION split.  The test split is evaluated once, at the very end, by
`run_final_test()` in the notebook.  Do not evaluate on test while iterating.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ExperimentConfig, replace_nested


@dataclass
class Experiment:
    cfg: ExperimentConfig
    purpose: str

    def __repr__(self) -> str:
        return f"Experiment({self.cfg.name})"


# --------------------------------------------------------------------------
# 1. Main comparison: the proposed model against FAIR baselines
# --------------------------------------------------------------------------
def build_baselines(base: ExperimentConfig) -> list[Experiment]:
    """Baselines for the main comparison.
    """
    return [
        Experiment(
            replace_nested(base, name="proposed_frozense_sam",
                           **{"model.arch": "frozense_sam"}),
            "FrozenSE-SAM, fully frozen encoder, no skip connections.",
        ),
        Experiment(
            replace_nested(base, name="baseline_unet_pretrained",
                           **{"model.arch": "unet_pretrained"}),
            "U-Net with an ImageNet-pretrained encoder: the fair CNN baseline.",
        ),
        Experiment(
            replace_nested(base, name="baseline_unet_scratch",
                           **{"model.arch": "unet_scratch"}),
            "U-Net from scratch, to measure the effect of pretraining.",
        ),
        Experiment(
            replace_nested(base, name="baseline_mobilesam_native",
                           **{"model.arch": "mobilesam_native", "train.epochs": 0}),
            "MobileSAM with its own prompt-based mask decoder, grid-prompted. "
            "Zero-shot: no training, so epochs=0.",
        ),
    ]


# --------------------------------------------------------------------------
# 2. Component ablations
# --------------------------------------------------------------------------
def build_ablations(base: ExperimentConfig) -> list[Experiment]:
    """One switch at a time.
    """
    return [
        Experiment(
            replace_nested(base, name="ablation_se_off", **{"model.use_se": False}),
            "SE blocks removed.",
        ),
        Experiment(
            replace_nested(base, name="ablation_skip_on",
                           **{"model.use_skip_connections": True}),
            "Skip connections added.",
        ),
        Experiment(
            replace_nested(base, name="ablation_encoder_unfrozen",
                           **{"model.freeze_encoder": False}),
            "Fully trainable encoder: does freezing help, or merely cost less? "
            "The encoder's BatchNorm statistics stay as pretrained (standard at "
            "batch size 6), so this differs from the proposed model in trainable "
            "weights only; activations are checkpointed to fit a 16 GB GPU.",
        ),
        Experiment(
            replace_nested(base, name="ablation_encoder_partial",
                           **{"model.unfreeze_last_n_blocks": 1}),
            "Partial fine-tuning: the last TinyViT stage and the neck, 52% of "
            "the encoder's parameters. (The last TWO stages plus neck would be "
            "85%, too close to full fine-tuning to be an informative middle "
            "point.)",
        ),
        Experiment(
            replace_nested(base, name="ablation_aux_sdf",
                           **{"model.aux_sdf_head": True}),
            "Auxiliary signed-distance head added beside the segmentation head, as in nnSAM.",
        ),
    ]


def build_loss_ablations(base: ExperimentConfig) -> list[Experiment]:
    """Loss-function ablations."""
    specs = [
        ("loss_tversky_plus_focal", "tversky_plus_focal",
         "Tversky plus focal term: the objective used by FrozenSE."),
        ("loss_focal_tversky", "focal_tversky",
         "Focal Tversky loss as defined by Abraham & Khan (2019)."),
        ("loss_tversky_only", "tversky", "Tversky alone, no focal term."),
        ("loss_dice", "dice", "Dice baseline."),
        ("loss_bce", "bce", "BCE baseline."),
    ]
    return [
        Experiment(
            replace_nested(base, name=name, **{"loss.name": loss}),
            purpose,
        )
        for name, loss, purpose in specs
    ]


def sweep_tversky_weights(base: ExperimentConfig) -> list[Experiment]:
    """Test both directions of the precision/recall trade-off.
    """
    pairs = [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]
    out = []
    for a, b in pairs:
        favours = "recall" if a > b else ("precision" if a < b else "balanced")
        out.append(Experiment(
            replace_nested(base, name=f"tversky_a{a}_b{b}",
                           **{"loss.tversky_alpha": a, "loss.tversky_beta": b}),
            f"alpha={a}, beta={b} (favours {favours}).",
        ))
    return out


# --------------------------------------------------------------------------
# 3. Data efficiency
# --------------------------------------------------------------------------
def build_data_efficiency(base: ExperimentConfig,
                          sizes: tuple[int, ...] = (5, 10, 20, 35),
                          seeds: tuple[int, ...] = (42, 43, 44)) -> list[Experiment]:
    """Learning curve over annotation budget, with repeated splits.
    """
    out = []
    for n in sizes:
        for seed in seeds:
            out.append(Experiment(
                replace_nested(
                    base, name=f"datasize_{n}cores_seed{seed}",
                    **{"split.n_train_cores": n, "split.seed": seed, "train.seed": seed},
                ),
                f"{n} training cores, split seed {seed}.",
            ))
    return out


def build_seed_repeats(base: ExperimentConfig,
                       seeds: tuple[int, ...] = (42, 43, 44, 45, 46)) -> list[Experiment]:
    """The proposed model across seeds, for variance estimates.
    """
    return [
        Experiment(
            replace_nested(base, name=f"proposed_seed{seed}",
                           **{"split.seed": seed, "train.seed": seed}),
            f"Proposed model, seed {seed}.",
        )
        for seed in seeds
    ]


# --------------------------------------------------------------------------
# 4. Robustness
# --------------------------------------------------------------------------
def build_robustness(base: ExperimentConfig) -> list[Experiment]:
    """Stain normalisation as a domain-shift proxy."""
    return [
        Experiment(
            replace_nested(base, name="robust_stain_norm",
                           **{"data.stain_normalise": True}),
            "With stain normalisation, as a domain-shift proxy.",
        ),
    ]


# --------------------------------------------------------------------------
# 5. The improved model, and the U-Nets retrained under the same protocol
# --------------------------------------------------------------------------
#: Applied to EVERY model in this group, U-Nets included, so a gain over them
#: comes from the model rather than from a better optimiser setting.
IMPROVED_PROTOCOL = {
    "train.decoder_lr_mult": 10.0,   # pretrained weights 5e-5, new weights 5e-4
    "train.scheduler_patience": 4,   # was 20, which never fired before early stopping
}


def build_improvements(base: ExperimentConfig) -> list[Experiment]:
    """Two design changes to the proposed model, the winner chosen on
    VALIDATION Dice, plus both U-Nets retrained under the same protocol.
    Ordered most important first."""
    r = IMPROVED_PROTOCOL
    return [
        Experiment(
            replace_nested(base, name="improved_skip",
                           **{"model.use_skip_connections": True}, **r),
            "Skip connections from the encoder's two high-resolution stages; "
            "encoder still frozen.",
        ),
        Experiment(
            replace_nested(base, name="baseline_unet_pretrained_r2",
                           **{"model.arch": "unet_pretrained"}, **r),
            "Pretrained U-Net under the same protocol as the improved model.",
        ),
        Experiment(
            replace_nested(base, name="improved_skip_partial",
                           **{"model.use_skip_connections": True,
                              "model.unfreeze_last_n_blocks": 1}, **r),
            "Skip connections plus the last TinyViT stage and neck fine-tuned.",
        ),
        Experiment(
            replace_nested(base, name="baseline_unet_scratch_r2",
                           **{"model.arch": "unet_scratch"}, **r),
            "U-Net from scratch under the same protocol.",
        ),
    ]


# --------------------------------------------------------------------------
def build_all(base: ExperimentConfig, include: tuple[str, ...] = ("baselines", "ablations")) -> list[Experiment]:
    """Assemble the requested experiment groups.

    Suggested order of execution, cheapest and most decisive first:

        1. "baselines"       - establishes whether the central claim survives
        2. "ablations"       - attributes the gain to specific components
        3. "loss"            - isolates the objective
        4. "seeds"           - variance for the statistics
        5. "data_efficiency" - the learning curve (most expensive: 12 runs)
        6. "robustness"

    Run group 1 first and look at the numbers before committing compute to
    the rest; if the proposed model does not beat a pretrained U-Net.
    """
    groups = {
        "baselines": build_baselines,
        "ablations": build_ablations,
        "loss": build_loss_ablations,
        "tversky_sweep": sweep_tversky_weights,
        "seeds": build_seed_repeats,
        "data_efficiency": build_data_efficiency,
        "robustness": build_robustness,
        "improve": build_improvements,
    }
    out: list[Experiment] = []
    for key in include:
        if key not in groups:
            raise ValueError(f"Unknown group {key!r}; options: {sorted(groups)}")
        out.extend(groups[key](base))
    return out

