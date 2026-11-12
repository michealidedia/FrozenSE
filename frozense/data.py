"""
Dataset, patch extraction and the three-way, slide-grouped split.

`make_split()` builds disjoint training, validation and test partitions with
no slide in two partitions. `Split.manifest()` records every core in every
partition so the split can be verified from the saved results.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Sequence
import glob
import hashlib
import json
import os
import random
import re
import warnings

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # allows import on a machine without torch, for tests
    torch = None
    Dataset = object

from .config import DataConfig, AugConfig, SplitConfig


Image.MAX_IMAGE_PIXELS = None  # TMA cores are ~5000x5000; Pillow's bomb guard trips


def derive_seed(*parts: object) -> int:
    """Deterministic 64-bit seed from arbitrary parts.

    Python's built-in `hash()` is SALTED for str and bytes unless
    PYTHONHASHSEED is fixed, so `random.Random(hash((seed, i, "train")))`
    produces DIFFERENT streams in different processes.  That would make
    training-patch selection and training-core subsampling irreproducible
    between Colab sessions, and would break the guarantee that every model is
    evaluated on identical patches.  blake2b is stable across processes,
    versions and platforms.
    """
    payload = "|".join(repr(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _as_dirs(value) -> list[str]:
    """Accept a single directory or several.

    Letting `images_dir` and `labels_dir` hold a list means the original
    imagesTr / imagesVal layout can be read in place, with no copying into a
    pooled folder. That matters on a machine where the data sits on an
    external drive with little room to duplicate it, and it removes a step
    in Colab too.
    """
    if isinstance(value, (str, os.PathLike)):
        return [os.fspath(value)]
    return [os.fspath(v) for v in value]


# --------------------------------------------------------------------------
# Core discovery and grouping
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Core:
    """One TMA core: an image path, its consensus mask, and its group id."""

    core_id: str
    image_path: str
    label_path: str
    group: str

    def __repr__(self) -> str:  # keeps manifests readable
        return f"Core({self.core_id}, group={self.group})"


def discover_cores(cfg: DataConfig, split_cfg: SplitConfig) -> list[Core]:
    """Find every image with a matching label, and assign each a group id.
    """
    image_dirs = _as_dirs(cfg.images_dir)
    label_dirs = _as_dirs(cfg.labels_dir)

    images: list[str] = []
    for d in image_dirs:
        for ext in cfg.image_extensions:
            images += glob.glob(os.path.join(d, f"*{ext}"))
            images += glob.glob(os.path.join(d, f"*{ext.upper()}"))
    # Deduplicate by BASENAME, not path: the same core must not enter the
    # pool twice because it appears in two source directories.
    seen_names: dict[str, str] = {}
    for path in sorted(set(images)):
        seen_names.setdefault(os.path.basename(path), path)
    images = sorted(seen_names.values())

    if not images:
        raise FileNotFoundError(
            f"No images found in {image_dirs}. Check the path, and that the "
            "extensions in DataConfig.image_extensions match your files."
        )

    pattern = re.compile(split_cfg.group_regex)
    cores: list[Core] = []
    missing_labels: list[str] = []
    ungrouped: list[str] = []

    for img in images:
        stem = os.path.splitext(os.path.basename(img))[0]

        # Gleason-2019 labels carry a suffix the images do not, e.g.
        # slide001_core004.jpg -> slide001_core004_classimg_nonconvex.png
        label = None
        for d in label_dirs:
            for suffix in cfg.label_suffixes:
                for ext in cfg.label_extensions:
                    candidate = os.path.join(d, stem + suffix + ext)
                    if os.path.exists(candidate):
                        label = candidate
                        break
                if label is not None:
                    break
            if label is not None:
                break
        if label is None:
            missing_labels.append(stem)
            continue

        match = pattern.match(stem)
        if match is None:
            ungrouped.append(stem)
            group = stem          # provisional; validated below
        else:
            group = match.group("group")

        cores.append(Core(core_id=stem, image_path=img, label_path=label, group=group))

    if missing_labels:
        tried = ", ".join(
            repr(s + e) for s in cfg.label_suffixes for e in cfg.label_extensions
        )
        raise FileNotFoundError(
            f"{len(missing_labels)} of {len(images)} image(s) have no matching "
            f"label, e.g. {missing_labels[:5]}.\n"
            f"Looked in {label_dirs} for the stem plus: {tried}\n\n"
            "If your labels use a different suffix, set "
            "DataConfig(label_suffixes=(...)). Otherwise every core must be "
            "accounted for: either supply the labels or move these files out "
            "of the pool and record the exclusion criterion "
            ". Note that imagesTs has no labels at all and must "
            "not be staged into the pool."
        )

    if ungrouped:
        message = (
            f"{len(ungrouped)} core(s) did not match group_regex "
            f"{split_cfg.group_regex!r}, e.g. {ungrouped[:5]}. "
            "Without a grouping key the split cannot be guaranteed free of "
            "patient-level overlap."
        )
        if not split_cfg.allow_core_level_fallback:
            raise ValueError(
                message
                + "\n\nEither fix group_regex to match your filenames, or set "
                "SplitConfig(allow_core_level_fallback=True) to split at core "
                "level and report that limitation."
            )
        warnings.warn(message + "\n\nFalling back to CORE-level splitting.", stacklevel=2)

    return cores


# --------------------------------------------------------------------------
# The three-way, group-aware split
# --------------------------------------------------------------------------
@dataclass
class Split:
    train: list[Core]
    val: list[Core]
    test: list[Core]
    grouping_used: str          # "group" or "core-level-fallback"
    train_pool_size: int        # cores available before subsampling to n_train_cores
    scheme: str = "holdout"
    fold: int = 0
    test_groups: list[str] = field(default_factory=list)
    val_groups: list[str] = field(default_factory=list)
    train_groups: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.assert_disjoint()

    def assert_disjoint(self) -> None:
        """Fail loudly if any group straddles two partitions."""
        g_train = {c.group for c in self.train}
        g_val = {c.group for c in self.val}
        g_test = {c.group for c in self.test}

        for a_name, a, b_name, b in [
            ("train", g_train, "val", g_val),
            ("train", g_train, "test", g_test),
            ("val", g_val, "test", g_test),
        ]:
            overlap = a & b
            if overlap:
                raise AssertionError(
                    f"LEAKAGE: {len(overlap)} group(s) appear in both {a_name} "
                    f"and {b_name}: {sorted(overlap)[:10]}"
                )

        ids = [c.core_id for c in self.train + self.val + self.test]
        if len(ids) != len(set(ids)):
            raise AssertionError("LEAKAGE: a core appears in more than one partition.")

    def manifest(self) -> dict:
        """Summary of the partition: groups, core counts and checks."""
        return {
            "scheme": self.scheme,
            "fold": self.fold,
            "groups": {
                "train": self.train_groups,
                "val": self.val_groups,
                "test": self.test_groups,
            },
            "grouping_used": self.grouping_used,
            "n_cores": {
                "train": len(self.train),
                "val": len(self.val),
                "test": len(self.test),
                "total": len(self.train) + len(self.val) + len(self.test),
            },
            "n_groups": {
                "train": len({c.group for c in self.train}),
                "val": len({c.group for c in self.val}),
                "test": len({c.group for c in self.test}),
            },
            "train_pool_size": self.train_pool_size,
            "group_overlap_check": "passed - no group appears in two partitions",
            "cores": {
                "train": sorted(c.core_id for c in self.train),
                "val": sorted(c.core_id for c in self.val),
                "test": sorted(c.core_id for c in self.test),
            },
        }

    def save_manifest(self, path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.manifest(), fh, indent=2)
        return path


def n_folds(cores: Sequence[Core]) -> int:
    """Number of leave-one-group-out folds available, i.e. the group count."""
    return len({c.group for c in cores})


def _assign_groups(groups: list[str], cfg: SplitConfig) -> tuple[set[str], set[str], set[str]]:
    """Return (test_groups, val_groups, train_groups) for this configuration."""
    n = len(groups)

    if cfg.scheme == "leave_one_group_out":
        if n < 3:
            raise ValueError(
                f"leave_one_group_out needs at least 3 groups (one each for "
                f"train, val and test); found {n}: {groups}. Use "
                'SplitConfig(scheme="holdout", allow_core_level_fallback=True) '
                "and disclose the limitation."
            )
        if cfg.fold >= n:
            raise ValueError(
                f"fold={cfg.fold} but only {n} groups exist, so folds run "
                f"0..{n - 1}. Call n_folds(cores) to size the loop."
            )
        # Groups in SORTED order: fold index -> slide is stable and reportable.
        ordered = sorted(groups)
        test = {ordered[cfg.fold]}
        val = {ordered[(cfg.fold + 1) % n]}
        train = set(ordered) - test - val
        return test, val, train

    # holdout: shuffle once with the seed, then allocate test, val, remainder.
    rng = random.Random(cfg.seed)
    shuffled = sorted(groups)
    rng.shuffle(shuffled)

    n_test = max(1, round(n * cfg.test_frac))
    n_val = max(1, round(n * cfg.val_frac))
    if n_test + n_val >= n:
        raise ValueError(
            f"Split leaves no training groups: {n} groups, "
            f"{n_test} test + {n_val} val requested."
        )
    return (set(shuffled[:n_test]),
            set(shuffled[n_test:n_test + n_val]),
            set(shuffled[n_test + n_val:]))


def make_split(cores: Sequence[Core], cfg: SplitConfig) -> Split:
    """Partition cores into train/val/test at the GROUP level.

    Under `scheme="leave_one_group_out"` (the default) fold `k` holds out
    group `k` for test and group `k+1` for validation, training on the rest.
    Over all folds every core is tested exactly once.  Under
    `scheme="holdout"` groups are shuffled with a fixed seed and allocated
    test-first, then val, with the remainder forming the training pool.

    Every core is accounted for: train + val + test + unused_pool == total.
    """
    groups = sorted({c.group for c in cores})
    test_groups, val_groups, train_groups = _assign_groups(groups, cfg)

    by_group: dict[str, list[Core]] = {}
    for c in cores:
        by_group.setdefault(c.group, []).append(c)

    def collect(gs: set[str]) -> list[Core]:
        out: list[Core] = []
        for g in sorted(gs):
            out.extend(sorted(by_group[g], key=lambda c: c.core_id))
        return out

    train_pool = collect(train_groups)
    val = collect(val_groups)
    test = collect(test_groups)

    pool_size = len(train_pool)
    if cfg.n_train_cores is not None:
        if cfg.n_train_cores > pool_size:
            raise ValueError(
                f"n_train_cores={cfg.n_train_cores} exceeds the training pool "
                f"of {pool_size} cores (groups {sorted(train_groups)}). "
                "Reduce it, or use fewer held-out groups."
            )
        # Subsample cores reproducibly.  The seed is mixed with the fold so
        # that different folds do not draw the same positions in their pools.
        pool_rng = random.Random(derive_seed(cfg.seed, cfg.fold, "train_pool"))
        train = sorted(pool_rng.sample(train_pool, cfg.n_train_cores),
                       key=lambda c: c.core_id)
    else:
        train = train_pool

    grouping = "core-level-fallback" if len(groups) == len(cores) else "group"

    return Split(train=train, val=val, test=test,
                 grouping_used=grouping, train_pool_size=pool_size,
                 scheme=cfg.scheme, fold=cfg.fold,
                 test_groups=sorted(test_groups), val_groups=sorted(val_groups),
                 train_groups=sorted(train_groups))


# --------------------------------------------------------------------------
# Label construction
# --------------------------------------------------------------------------
def binarise_mask(mask: np.ndarray, tumour_grades: Sequence[int] | None) -> np.ndarray:
    """Map Gleason grade labels to a binary tumour mask.

    `tumour_grades=None` treats any non-zero value as tumour, which is right
    for masks already binarised upstream and WRONG for raw grade maps where
    low values encode benign tissue.  `audit_label_values()` below is how you
    find out which you have.
    """
    if tumour_grades is None:
        return (mask > 0).astype(np.float32)
    return np.isin(mask, list(tumour_grades)).astype(np.float32)


# --------------------------------------------------------------------------
# Label value audit
# --------------------------------------------------------------------------
def audit_label_values(
    cores: Sequence[Core],
    cfg: DataConfig,
    source_of: dict[str, str] | None = None,
    sample_per_source: int = 12,
    seed: int = 0,
) -> dict:
    """Report the distinct values present in the label PNGs, by source folder.

    WHY THIS EXISTS
    ---------------
    This project has two label sources with different provenance:
    `Generated_Labels_Train` was produced locally by majority vote, while
    `labelsVal` may have shipped with the dataset.  If one is already binary
    (values {0, 1}) and the other holds raw Gleason grades (values including
    3, 4, 5), then binarising both with `tumour_grades=(3, 4, 5)` yields
    ALL-ZERO masks for every core from the binary source -- silently, with no
    error, and with a plausible-looking loss curve, because an all-background
    target is trivially satisfiable.

    Pass `source_of` mapping core_id -> source label ("train_labels",
    "val_labels") to get a per-source breakdown; otherwise everything is
    pooled under "all".

    Returns a dict with, per source, the union of observed values and the
    mean foreground fraction under the current `cfg.tumour_grades`.  Raises if
    any sampled core binarises to an empty mask, because that is always a
    configuration error rather than a property of the data.
    """
    rng = random.Random(seed)
    by_source: dict[str, list[Core]] = {}
    for c in cores:
        key = (source_of or {}).get(c.core_id, "all")
        by_source.setdefault(key, []).append(c)

    report: dict[str, dict] = {}
    empty: list[tuple[str, str]] = []

    for source, members in sorted(by_source.items()):
        picks = rng.sample(members, min(sample_per_source, len(members)))
        values: set[int] = set()
        fg_fracs: list[float] = []

        for core in picks:
            raw = np.array(Image.open(core.label_path))
            if raw.ndim == 3:
                raw = raw[..., 0]
            values.update(int(v) for v in np.unique(raw))
            frac = float(binarise_mask(raw, cfg.tumour_grades).mean())
            fg_fracs.append(frac)
            if frac == 0.0:
                empty.append((source, core.core_id))

        report[source] = {
            "n_cores": len(members),
            "n_sampled": len(picks),
            "observed_values": sorted(values),
            "foreground_fraction_mean": round(float(np.mean(fg_fracs)), 5),
            "foreground_fraction_min": round(float(np.min(fg_fracs)), 5),
            "foreground_fraction_max": round(float(np.max(fg_fracs)), 5),
        }

    value_sets = {s: tuple(r["observed_values"]) for s, r in report.items()}
    report["_sources_agree"] = len(set(value_sets.values())) <= 1
    report["_tumour_grades_used"] = (
        "any non-zero" if cfg.tumour_grades is None else list(cfg.tumour_grades)
    )

    # A core with no Gleason 3-5 pixels is a benign core -- Gleason2019 has
    # some -- and is kept (Section 7 accounts for empty masks). Only a source
    # where EVERY sampled core is empty points to a wrong tumour_grades.
    report["_benign_cores_sampled"] = empty
    all_empty = [s for s in by_source
                 if report[s]["n_sampled"] and report[s]["foreground_fraction_max"] == 0.0]
    if all_empty:
        observed = {s: r["observed_values"] for s, r in report.items()
                    if isinstance(r, dict) and "observed_values" in r}
        raise ValueError(
            f"Every sampled core from {all_empty} binarises to an EMPTY mask under "
            f"tumour_grades={report['_tumour_grades_used']}.\n\n"
            f"Observed label values by source: {observed}\n\n"
            "This is a configuration error, not a data property. Set "
            "DataConfig(tumour_grades=...) to the values that actually encode "
            "tumour in these files, or tumour_grades=None if the masks are "
            "already binary."
        )

    return report


def format_label_audit(report: dict) -> str:
    """Human-readable rendering of `audit_label_values()`."""
    lines = [f"tumour_grades = {report['_tumour_grades_used']}", ""]
    for source, r in report.items():
        if source.startswith("_"):
            continue
        lines.append(f"{source}:")
        lines.append(f"  cores            {r['n_cores']} ({r['n_sampled']} sampled)")
        lines.append(f"  label values     {r['observed_values']}")
        lines.append(f"  foreground frac  mean {r['foreground_fraction_mean']:.4f}  "
                     f"range {r['foreground_fraction_min']:.4f}"
                     f"-{r['foreground_fraction_max']:.4f}")
    benign = report.get("_benign_cores_sampled", [])
    if benign:
        lines += ["", f"{len(benign)} sampled core(s) contain no tumour pixels (benign "
                  f"cores, kept; counted in Section 7), e.g. {[c for _, c in benign[:3]]}"]
    if not report["_sources_agree"]:
        lines += [
            "",
            "WARNING: the label sources do NOT share the same value set.",
            "One may be binary and the other a raw grade map. Reconcile them",
            "before training and document how it was constructed.",
        ]
    return "\n".join(lines)


def group_composition(cores: Sequence[Core],
                      source_of: dict[str, str] | None = None) -> dict:
    """Cores per group, optionally broken down by original source folder.
    """
    out: dict[str, dict] = {}
    for c in sorted(cores, key=lambda c: c.core_id):
        entry = out.setdefault(c.group, {"n_cores": 0, "sources": {}})
        entry["n_cores"] += 1
        if source_of:
            src = source_of.get(c.core_id, "unknown")
            entry["sources"][src] = entry["sources"].get(src, 0) + 1

    shared = sorted(g for g, e in out.items() if len(e["sources"]) > 1)
    return {
        "per_group": out,
        "n_groups": len(out),
        "n_cores": len(cores),
        "groups_spanning_multiple_sources": shared,
    }


# --------------------------------------------------------------------------
# Patch dataset
# --------------------------------------------------------------------------
class PatchDataset(Dataset):
    """Patches sampled from a set of cores.

    Training patches use foreground-biased sampling; evaluation patches do
    not (which is why empty ground-truth masks occur at evaluation time and
    must be handled explicitly -- see metrics.py).

    PRELOADING
    ----------
    Because the positions are fixed, every patch can be cut once, up front.
    `preload=True` (the default) decodes each core exactly once -- several at
    a time, since image decoding releases the GIL -- crops its patches, and
    discards the full image.  For 35 training cores at 35 patches each this
    holds about 1.3 GB, instead of a whole-core cache.

    The alternative it replaces cached 8 cores and cleared the cache when
    full.  With shuffled batches drawing from all 35 training cores, most
    samples missed the cache and re-decoded a ~5000 x 5000 JPEG plus its
    label PNG, so the GPU spent most of every epoch waiting for the CPU --
    billed GPU time on any platform.  `preload=False` keeps that lazy path
    for machines too small to hold the patches, and yields identical patches.
    """

    def __init__(
        self,
        cores: Sequence[Core],
        data_cfg: DataConfig,
        aug_cfg: AugConfig | None = None,
        train: bool = False,
        seed: int = 42,
        preload: bool = True,
        preload_threads: int = 4,
        verbose: bool = False,
    ) -> None:
        self.cores = list(cores)
        self.cfg = data_cfg
        self.train = train
        self.seed = seed
        self.transform = build_transform(aug_cfg, data_cfg, train=train) if aug_cfg else None

        n_per_core = (data_cfg.patches_per_core_train if train
                      else data_cfg.patches_per_core_eval)
        self.n_per_core = n_per_core

        # (core_index, patch_index) pairs
        self.index: list[tuple[int, int]] = [
            (ci, pi) for ci in range(len(self.cores)) for pi in range(n_per_core)
        ]

        # Lazy path only: a small cache of decoded cores.
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._cache_limit = 8

        self._patches: list[tuple[np.ndarray, np.ndarray]] | None = None
        if preload:
            self._preload(max(1, int(preload_threads)), verbose)

    def __len__(self) -> int:
        return len(self.index)

    # ---------------------------------------------------------------- cores
    def _decode_core(self, ci: int) -> tuple[np.ndarray, np.ndarray]:
        """Full core as (uint8 H x W x 3 image, uint8 H x W mask of 0/1).

        The mask is uint8 rather than float32: identical values, a quarter of
        the memory. It becomes float only per patch, at the very end.
        """
        core = self.cores[ci]
        img = np.array(Image.open(core.image_path).convert("RGB"))
        lbl_raw = np.array(Image.open(core.label_path))
        if lbl_raw.ndim == 3:
            lbl_raw = lbl_raw[..., 0]
        lbl = (binarise_mask(lbl_raw, self.cfg.tumour_grades) > 0.5).astype(np.uint8)
        del lbl_raw

        if lbl.shape != img.shape[:2]:
            lbl = np.array(Image.fromarray(lbl).resize(
                (img.shape[1], img.shape[0]), Image.NEAREST))
        return img, lbl

    def _load_core(self, ci: int) -> tuple[np.ndarray, np.ndarray]:
        """Lazy path: decode with a small cache."""
        if ci in self._cache:
            return self._cache[ci]
        img, lbl = self._decode_core(ci)
        if len(self._cache) >= self._cache_limit:
            self._cache.clear()
        self._cache[ci] = (img, lbl)
        return img, lbl

    # -------------------------------------------------------------- patches
    def _coords(self, ci: int, pi: int, h: int, w: int,
                lbl: np.ndarray | None) -> tuple[int, int]:
        """Top-left corner of patch `pi` of core `ci`.

        Keyed on core_id rather than list position, so a core always gets the
        same patches no matter which split it lands in or how the cores happen
        to be ordered. That is what makes the model comparison paired, and
        the paired Wilcoxon tests in metrics.py valid.
        """
        ps = self.cfg.patch_size
        max_y, max_x = max(0, h - ps), max(0, w - ps)
        core_id = self.cores[ci].core_id

        if not self.train:
            rng = random.Random(derive_seed(self.seed, core_id, pi, "eval"))
            return rng.randint(0, max_y), rng.randint(0, max_x)

        rng = random.Random(derive_seed(self.seed, core_id, pi, "train"))
        best = (rng.randint(0, max_y), rng.randint(0, max_x))
        for _ in range(self.cfg.max_sampling_tries):
            y, x = rng.randint(0, max_y), rng.randint(0, max_x)
            frac = lbl[y:y + ps, x:x + ps].mean()
            if frac >= self.cfg.min_foreground_frac_train:
                return y, x
            best = (y, x)
        return best

    def _sample_coords(self, ci: int, pi: int, h: int, w: int) -> tuple[int, int]:
        """Backwards-compatible wrapper used by the lazy path."""
        lbl = self._load_core(ci)[1] if self.train else None
        return self._coords(ci, pi, h, w, lbl)

    def _crop(self, img: np.ndarray, lbl: np.ndarray, y: int, x: int
              ) -> tuple[np.ndarray, np.ndarray]:
        ps = self.cfg.patch_size
        img_patch = img[y:y + ps, x:x + ps]
        lbl_patch = lbl[y:y + ps, x:x + ps]
        # Pad if the core was smaller than one patch
        if img_patch.shape[0] < ps or img_patch.shape[1] < ps:
            pad_y = ps - img_patch.shape[0]
            pad_x = ps - img_patch.shape[1]
            img_patch = np.pad(img_patch, ((0, pad_y), (0, pad_x), (0, 0)))
            lbl_patch = np.pad(lbl_patch, ((0, pad_y), (0, pad_x)))
        # COPY: a slice is a view that would keep the whole core alive.
        return np.ascontiguousarray(img_patch), np.ascontiguousarray(lbl_patch)

    def _patches_for_core(self, ci: int) -> list[tuple[np.ndarray, np.ndarray]]:
        img, lbl = self._decode_core(ci)
        h, w = img.shape[:2]
        out = []
        for pi in range(self.n_per_core):
            y, x = self._coords(ci, pi, h, w, lbl)
            out.append(self._crop(img, lbl, y, x))
        return out

    def _preload(self, threads: int, verbose: bool) -> None:
        from concurrent.futures import ThreadPoolExecutor

        t0 = __import__("time").time()
        with ThreadPoolExecutor(max_workers=threads) as pool:
            per_core = list(pool.map(self._patches_for_core, range(len(self.cores))))
        self._patches = [p for core_patches in per_core for p in core_patches]

        if verbose:
            mb = sum(a.nbytes + b.nbytes for a, b in self._patches) / 1024 ** 2
            kind = "train" if self.train else "eval"
            print(f"  preloaded {len(self._patches)} {kind} patches from "
                  f"{len(self.cores)} cores: {mb:,.0f} MB in "
                  f"{__import__('time').time() - t0:.0f} s")

    def raw_patch(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Patch `i` before augmentation: (uint8 image, uint8 mask)."""
        if self._patches is not None:
            return self._patches[i]
        ci, pi = self.index[i]
        img, lbl = self._load_core(ci)
        y, x = self._sample_coords(ci, pi, img.shape[0], img.shape[1])
        return self._crop(img, lbl, y, x)

    def __getitem__(self, i: int):
        ci, _ = self.index[i]
        core = self.cores[ci]
        img_patch, lbl_patch = self.raw_patch(i)

        if self.transform is not None:
            augmented = self.transform(image=img_patch, mask=lbl_patch)
            img_patch, lbl_patch = augmented["image"], augmented["mask"]

        if torch is not None:
            if not isinstance(img_patch, torch.Tensor):
                img_patch = torch.from_numpy(
                    np.ascontiguousarray(img_patch.transpose(2, 0, 1))
                ).float() / 255.0
            if not isinstance(lbl_patch, torch.Tensor):
                lbl_patch = torch.from_numpy(np.ascontiguousarray(lbl_patch))
            # Masks are stored as uint8; the losses and metrics need float.
            lbl_patch = lbl_patch.float()
            if lbl_patch.ndim == 2:
                lbl_patch = lbl_patch.unsqueeze(0)
        elif lbl_patch.ndim == 2:
            lbl_patch = lbl_patch[None]

        return {
            "image": img_patch,
            "mask": lbl_patch,
            "core_id": core.core_id,
            "group": core.group,
        }


def _coarse_dropout(A, aug: AugConfig):
    """CoarseDropout with EXACTLY `cutout_max_holes` holes of the given size.

    Albumentations 2.x renamed every argument here. Given the old names it
    only WARNS and then ignores them, silently running its own defaults while
    the augmentation table reports the configured values -- so the argument
    names are chosen from the installed version's signature, never assumed.

    Under the old API, min_holes/min_height/min_width defaulted to their max_
    counterparts, i.e. a fixed count of fixed-size holes. The same fixed
    behaviour is requested explicitly under both APIs.
    """
    import inspect

    n, h, w = aug.cutout_max_holes, aug.cutout_max_h, aug.cutout_max_w
    params = inspect.signature(A.CoarseDropout.__init__).parameters
    if "num_holes_range" in params:
        kw = dict(num_holes_range=(n, n), hole_height_range=(h, h),
                  hole_width_range=(w, w))
        kw.update(dict(fill=0, fill_mask=0) if "fill" in params
                  else dict(fill_value=0, mask_fill_value=0))
    else:
        kw = dict(max_holes=n, min_holes=n, max_height=h, min_height=h,
                  max_width=w, min_width=w, fill_value=0, mask_fill_value=0)
    return A.CoarseDropout(p=aug.cutout_p, **kw)


def build_transform(aug: AugConfig | None, data: DataConfig, train: bool):
    """Albumentations pipeline.

    Validation and test share an identical, augmentation-free pipeline so
    that nothing stochastic influences model selection or reporting.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # No Resize here. Albumentations resizes the MASK along with the image,
    # so resizing patches to MobileSAM's 1024 input here would hand the loss a
    # 1024 x 1024 mask against a 512 x 512 prediction -- a crash on the first
    # batch -- and would put HD95/ASD on the wrong pixel grid. Patches stay at
    # native resolution; SAM-based models upsample internally (models.py).

    if not train or aug is None:
        return A.Compose([
            A.Normalize(mean=data.norm_mean, std=data.norm_std, max_pixel_value=255.0),
            ToTensorV2(),
        ])

    ops = [
        A.HorizontalFlip(p=aug.hflip_p),
        A.VerticalFlip(p=aug.vflip_p),
        A.Rotate(limit=aug.rotate_limit, p=aug.rotate_p, border_mode=0),
        A.Affine(
            scale=aug.affine_scale,
            translate_percent={"x": aug.affine_translate_pct, "y": aug.affine_translate_pct},
            rotate=aug.affine_rotate,
            shear=aug.affine_shear,
            p=aug.affine_p,
        ),
    ]
    if aug.elastic:
        ops.append(A.ElasticTransform(alpha=aug.elastic_alpha, sigma=aug.elastic_sigma,
                                      p=aug.elastic_p, border_mode=0))
    if aug.grid_distortion:
        ops.append(A.GridDistortion(num_steps=aug.grid_distortion_steps,
                                    distort_limit=aug.grid_distortion_limit,
                                    p=aug.grid_distortion_p, border_mode=0))
    ops += [
        A.ColorJitter(brightness=aug.brightness, contrast=aug.contrast,
                      saturation=aug.saturation, hue=aug.hue, p=aug.color_jitter_p),
        A.GaussianBlur(blur_limit=aug.blur_limit, p=aug.blur_p),
    ]
    if aug.cutout:
        ops.append(_coarse_dropout(A, aug))
    ops += [
        A.Normalize(mean=data.norm_mean, std=data.norm_std, max_pixel_value=255.0),
        ToTensorV2(),
    ]
    return A.Compose(ops)
