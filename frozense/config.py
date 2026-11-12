"""
Configuration for all FrozenSE experiments.

Every setting lives in a single frozen dataclass, so each run records exactly
the configuration that produced it. Train, validation and test partitions are
disjoint and grouped by slide, and the test partition is read once. The
handling of empty masks is an explicit, recorded policy (see metrics.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Sequence
import json
import os


# --------------------------------------------------------------------------
# Loss names.  These are deliberately explicit so that a config can never
# again claim one objective while optimising another.
# --------------------------------------------------------------------------
LossName = Literal[
    "tversky",            # 1 - TI(alpha, beta)
    "focal_tversky",      # (1 - TI(alpha, beta)) ** gamma   <- Abraham & Khan 2019
    "tversky_plus_focal",
    "dice",
    "bce",
    "bce_dice",
]

EmptyMaskPolicy = Literal[
    "report_separately",  # recommended: exclude from distance means but COUNT and report
    "worst_case",         # substitute the patch diagonal as the distance
    "drop_silently",      # drop undefined values (not recommended)
]


SplitScheme = Literal[
    "leave_one_group_out",  # recommended for Gleason-2019: 6 slides -> 6 folds
    "holdout",              # single fixed three-way partition
]


@dataclass(frozen=True)
class SplitConfig:
    """Group-aware partition of the TMA cores.

    `group_regex` extracts the grouping key from the core filename.  The
    Gleason-2019 cores in this project are named `slide001_core004.jpg`, so
    the default pattern captures the glass slide.  If the pattern does not
    match, the splitter REFUSES to fall back silently -- it raises, unless
    `allow_core_level_fallback` is set, in which case it warns loudly and
    records the fallback in the run manifest.

    LEAVE-ONE-GROUP-OUT
    -------------------
    This dataset has only six slides, so a single fixed partition would rest
    the entire result on which two slides happened to land in test.  Under
    `scheme="leave_one_group_out"` each fold holds out one slide for test and
    the next for validation, training on the remainder.  Over all folds every
    core is tested exactly once, which gives both a across-fold standard
    deviation and a pooled core-level distribution for confidence intervals
.
    """

    group_regex: str = r"^(?P<group>slide\d+)"
    allow_core_level_fallback: bool = False

    scheme: SplitScheme = "leave_one_group_out"

    # leave_one_group_out only: which fold to build.  Groups are taken in
    # sorted order, so fold 0 always tests the lowest-numbered slide and the
    # mapping from fold index to slide is stable and reportable.
    fold: int = 0

    # holdout only.  Fractions apply to GROUPS, not cores, so that no group
    # straddles a split.
    test_frac: float = 0.40
    val_frac: float = 0.20
    # remainder becomes the training pool

    # Number of training cores actually used.  `None` = use the whole pool.
    # test; the data-efficiency study sweeps it.
    n_train_cores: int | None = 35

    seed: int = 42

    def __post_init__(self) -> None:
        if self.fold < 0:
            raise ValueError(f"fold must be >= 0, got {self.fold}")
        if self.scheme == "holdout":
            if not 0.0 < self.test_frac < 1.0:
                raise ValueError(f"test_frac must be in (0,1), got {self.test_frac}")
            if not 0.0 < self.val_frac < 1.0:
                raise ValueError(f"val_frac must be in (0,1), got {self.val_frac}")
            if self.test_frac + self.val_frac >= 1.0:
                raise ValueError(
                    "test_frac + val_frac must leave a non-empty training pool; "
                    f"got {self.test_frac} + {self.val_frac}"
                )


@dataclass(frozen=True)
class DataConfig:
    """Patch extraction and preprocessing."""

    # Where the cores live.  Either a single pooled folder, or a SEQUENCE of
    # folders read in place -- so an imagesTr / imagesVal layout
    # needs no copying:
    #
    #   images_dir=("D:/.../imagesTr", "D:/.../imagesVal")
    #   labels_dir=("D:/.../Generated_Labels_Train", "D:/.../labelsVal")
    #
    # Either way the SPLITTER decides the partition, not the directory
    # layout.  Cores are
    # deduplicated by filename, so a core appearing in two source folders is
    # still counted once.
    images_dir: str | Sequence[str] = "/content/data/images"
    labels_dir: str | Sequence[str] = "/content/data/labels"

    image_extensions: Sequence[str] = (".jpg", ".jpeg", ".png")
    label_extensions: Sequence[str] = (".png",)

    # Gleason-2019 label files carry a suffix the image files do not:
    #   image  slide001_core004.jpg
    #   label  slide001_core004_classimg_nonconvex.png
    # Candidates are tried in order, so "" keeps plain stem-matched labels
    # working too.  A core with no match raises rather than being dropped.
    label_suffixes: Sequence[str] = ("_classimg_nonconvex", "")

    # Patches and masks stay at this native size end to end. MobileSAM's
    # 1024 x 1024 input is produced inside the SAM-based models (see
    # models.SAM_INPUT_SIZE), never in the data pipeline.
    patch_size: int = 512

    patches_per_core_train: int = 35
    patches_per_core_eval: int = 20

    # Label values counted as tumour foreground.  `None` means "any non-zero
    # value", which is correct for already-binarised masks and WRONG for raw
    # Gleason grade maps, where low values encode benign tissue.
    #
    # This project's two label sources may not agree: Generated_Labels_Train
    # was produced locally while labelsVal may have shipped with the dataset.
    # Run `data.audit_label_values()` (notebook step 2b) before training -- if
    # the sources carry different value sets, binarising both with one rule
    # silently empties half the masks.
    tumour_grades: Sequence[int] | None = (3, 4, 5)

    # Foreground-biased sampling for training patches only.
    min_foreground_frac_train: float = 0.05
    max_sampling_tries: int = 20

    # Evaluation patches are sampled WITHOUT a foreground requirement, which
    # is why the empty-mask policy below matters.
    eval_requires_foreground: bool = False

    pixel_spacing_um: float = 0.25    # 40x Leica SCN400

    # SAM/MobileSAM inherit ImageNet normalisation statistics.
    norm_mean: Sequence[float] = (0.485, 0.456, 0.406)
    norm_std: Sequence[float] = (0.229, 0.224, 0.225)

    # Stain normalisation: off by default, so held-out slides test stain
    # robustness.
    stain_normalise: bool = False

    @property
    def pixel_spacing_mm(self) -> float:
        return self.pixel_spacing_um / 1000.0


@dataclass(frozen=True)
class AugConfig:
    """Augmentation pipeline.
    """

    hflip_p: float = 0.5
    vflip_p: float = 0.5
    rotate_p: float = 0.75
    rotate_limit: tuple[int, int] = (-60, 60)

    affine_p: float = 0.8
    affine_scale: tuple[float, float] = (0.7, 1.3)
    affine_translate_pct: tuple[float, float] = (-0.18, 0.18)
    affine_rotate: tuple[int, int] = (-30, 30)
    affine_shear: tuple[int, int] = (-25, 25)

    elastic: bool = True
    elastic_p: float = 0.6
    elastic_alpha: float = 120.0
    elastic_sigma: float = 120.0 * 0.09

    grid_distortion: bool = True
    grid_distortion_p: float = 0.5
    grid_distortion_steps: int = 5
    grid_distortion_limit: float = 0.3

    color_jitter_p: float = 0.85
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.4
    hue: float = 0.15

    blur_p: float = 0.4
    blur_limit: tuple[int, int] = (3, 7)

    cutout: bool = True
    cutout_p: float = 0.5
    cutout_max_holes: int = 8
    cutout_max_h: int = 64
    cutout_max_w: int = 64

    def as_table_rows(self) -> list[tuple[str, str, str]]:
        """        """
        return [
            ("Horizontal flip", f"{self.hflip_p}", "-"),
            ("Vertical flip", f"{self.vflip_p}", "-"),
            ("Rotation", f"{self.rotate_p}", f"{self.rotate_limit[0]} to {self.rotate_limit[1]} deg"),
            ("Affine", f"{self.affine_p}",
             f"scale {self.affine_scale}, translate {self.affine_translate_pct}, "
             f"rotate {self.affine_rotate} deg, shear {self.affine_shear} deg"),
            ("Elastic", f"{self.elastic_p}" if self.elastic else "off",
             f"alpha={self.elastic_alpha}, sigma={self.elastic_sigma:.2f}"),
            ("Grid distortion", f"{self.grid_distortion_p}" if self.grid_distortion else "off",
             f"steps={self.grid_distortion_steps}, limit={self.grid_distortion_limit}"),
            ("Colour jitter", f"{self.color_jitter_p}",
             f"brightness/contrast/saturation {self.brightness}, hue {self.hue}"),
            ("Gaussian blur", f"{self.blur_p}", f"kernel {self.blur_limit}"),
            ("Coarse dropout", f"{self.cutout_p}" if self.cutout else "off",
             f"{self.cutout_max_holes} holes of {self.cutout_max_h}x{self.cutout_max_w} px "
             f"(image and mask)"),
        ]


@dataclass(frozen=True)
class ModelConfig:
    """Architecture."""

    arch: Literal[
        "frozense_sam",      # proposed: frozen MobileSAM encoder + SE decoder
        "unet_pretrained",   # baseline: ImageNet-pretrained ResNet-34 encoder
        "unet_scratch",
        "mobilesam_native",  # frozen encoder + SAM's own prompt-based decoder
        "mobilesam_plain",   # frozen encoder + decoder WITHOUT SE
        "pfm_phikon",        # v1.6: Phikon pathology ViT + the same decoder
        "pfm_uni",           # v1.6: UNI pathology ViT + the same decoder (gated)
    ] = "frozense_sam"

    decoder_channels: tuple[int, ...] = (192, 96, 48)
    convs_per_stage: int = 2

    use_se: bool = True            # ablation switch
    se_reduction: int = 16
    use_skip_connections: bool = False  # ablation switch

    freeze_encoder: bool = True
    # Partial fine-tuning ablation: unfreeze the last N encoder blocks.
    unfreeze_last_n_blocks: int = 0

    out_channels: int = 1

    # Auxiliary signed-distance head, added beside the segmentation head
    # (as in nnSAM).
    aux_sdf_head: bool = False
    aux_sdf_weight: float = 0.5

    # Encoder weights
    sam_checkpoint: str = "/content/weights/mobile_sam.pt"
    pretrained_encoder_name: str = "resnet34"   # for unet_pretrained
    pretrained_encoder_weights: str | None = "imagenet"


@dataclass(frozen=True)
class LossConfig:
    name: LossName = "tversky_plus_focal"

    # Tversky: alpha weights FALSE NEGATIVES, beta weights FALSE POSITIVES.
    # alpha=0.4, beta=0.6 penalises false positives more (favours precision).
    # `sweep_tversky_weights()` in experiments.py tests both directions.
    tversky_alpha: float = 0.4
    tversky_beta: float = 0.6

    focal_gamma: float = 2.0
    focal_alpha: float = 0.20
    focal_weight: float = 1.0

    smooth: float = 1e-6


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 300
    batch_size: int = 6
    lr: float = 5e-5
    weight_decay: float = 1e-5
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    # Pretrained weights train at `lr`; newly initialised ones (decoders,
    # heads, a scratch encoder) at `lr * decoder_lr_mult`. 1.0 = one rate.
    decoder_lr_mult: float = 1.0

    scheduler_patience: int = 20
    scheduler_factor: float = 0.2
    min_lr: float = 1e-7

    early_stopping_patience: int = 40
    # Selection metric is computed on the VALIDATION split only.
    selection_metric: str = "dice"
    selection_mode: Literal["max", "min"] = "max"

    amp: bool = True
    num_workers: int = 2
    seed: int = 42

    threshold: float = 0.5   # sigmoid threshold


@dataclass(frozen=True)
class EvalConfig:
    empty_mask_policy: EmptyMaskPolicy = "report_separately"

    # Aggregate per CORE before averaging.  Patch-level aggregation treats
    # correlated patches as independent and inflates precision
    #.
    aggregate_by_core: bool = True

    n_bootstrap: int = 2000
    ci_level: float = 0.95
    bootstrap_seed: int = 0

    # Report distances in BOTH pixels and mm.
    report_pixels_and_mm: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "frozense_sam_baseline"
    split: SplitConfig = field(default_factory=SplitConfig)
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    output_root: str = "/content/drive/MyDrive/frozense_runs"

    @property
    def run_dir(self) -> str:
        return os.path.join(self.output_root, self.name)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def save(self) -> str:
        os.makedirs(self.run_dir, exist_ok=True)
        path = os.path.join(self.run_dir, "config.json")
        with open(path, "w") as fh:
            fh.write(self.to_json())
        return path


def replace_nested(cfg: ExperimentConfig, **overrides) -> ExperimentConfig:
    """Return a copy of `cfg` with dotted-path overrides applied.

    Example
    -------
    >>> replace_nested(cfg, name="se_off", **{"model.use_se": False})
    """
    import dataclasses

    sub_updates: dict[str, dict] = {}
    top_updates: dict[str, object] = {}

    for key, value in overrides.items():
        if "." in key:
            head, tail = key.split(".", 1)
            sub_updates.setdefault(head, {})[tail] = value
        else:
            top_updates[key] = value

    for head, updates in sub_updates.items():
        current = getattr(cfg, head)
        top_updates[head] = dataclasses.replace(current, **updates)

    return dataclasses.replace(cfg, **top_updates)


def _tuplify(value):
    """JSON turns tuples into lists; turn them back, recursively."""
    if isinstance(value, list):
        return tuple(_tuplify(v) for v in value)
    if isinstance(value, dict):
        return {k: _tuplify(v) for k, v in value.items()}
    return value


def experiment_config_from_dict(d: dict) -> ExperimentConfig:
    """Rebuild an ExperimentConfig from `asdict()` / `to_json()` output.

    Used to hand the notebook's exact configuration to worker processes
    (one per GPU). The round trip preserves the configuration fingerprint,
    so results computed by a worker are interchangeable with results
    computed in the notebook itself.
    """
    sections = {"split": SplitConfig, "data": DataConfig, "aug": AugConfig,
                "model": ModelConfig, "loss": LossConfig, "train": TrainConfig,
                "eval": EvalConfig}
    kwargs = {k: cls(**_tuplify(d[k])) for k, cls in sections.items() if k in d}
    for k in ("name", "output_root"):
        if k in d:
            kwargs[k] = d[k]
    return ExperimentConfig(**kwargs)
