"""
Per-pathologist annotation maps: coverage, agreement, and consensus labels.

The `annotations/` tree holds one folder per annotating pathologist:

    annotations/
      Map1/  slide001_core003_classimg_nonconvex.png  ...
      Map2/  ...

so all of it is recoverable and checkable rather than asserted.

3. `build_consensus()`  -- regenerate the consensus mask under an explicit,
   named vote order and tie policy.  `compare_to_existing()` then checks the
   regenerated mask against the label PNGs the project already trained on.
   If they match, the construction is documented and verified; if they do
   not, that needs explaining before submission, not after.

VOTE ORDER
----------
`binarise_then_vote`  each annotator's grade map is collapsed to a binary
                      tumour mask, then pixels are voted on.
`vote_then_binarise`  the modal GRADE is taken per pixel across annotators,
                      then that consensus grade map is collapsed.

They differ wherever annotators agree that tissue is tumour but disagree on
grade, and wherever the modal grade is a tie.  Reporting both, and showing
the downstream result is insensitive to the choice, is a much stronger answer
than picking one and hoping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence
import glob
import os

import numpy as np
from PIL import Image

from .config import DataConfig
from .data import binarise_mask

Image.MAX_IMAGE_PIXELS = None

VoteOrder = Literal["binarise_then_vote", "vote_then_binarise"]
TiePolicy = Literal["benign", "tumour"]


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
@dataclass
class AnnotationSet:
    """Every per-pathologist map found, indexed by core and annotator."""

    by_core: dict[str, dict[str, str]]   # core_id -> annotator -> path
    annotators: list[str]
    root: str

    def core_ids(self, min_annotators: int = 1) -> list[str]:
        return sorted(c for c, m in self.by_core.items() if len(m) >= min_annotators)

    def n_annotators(self, core_id: str) -> int:
        return len(self.by_core.get(core_id, {}))


def discover_annotations(
    annotations_dir: str,
    label_suffixes: Sequence[str] = ("_classimg_nonconvex", ""),
    pattern: str = "*.png",
) -> AnnotationSet:
    """Index `annotations/<annotator>/<core><suffix>.png`.

    Annotator names are taken from the immediate subdirectory names, so the
    structure is read rather than assumed -- it works for `Map1..Map6`,
    `pathologist_A`, or anything else.
    """
    if not os.path.isdir(annotations_dir):
        raise FileNotFoundError(
            f"{annotations_dir} is not a directory. Expected one subfolder per "
            "annotating pathologist, each holding that reader's maps."
        )

    subdirs = sorted(
        d for d in os.listdir(annotations_dir)
        if os.path.isdir(os.path.join(annotations_dir, d))
    )
    if not subdirs:
        raise FileNotFoundError(
            f"No subdirectories in {annotations_dir}. If the maps sit directly "
            "in this folder with no per-annotator split, they cannot be used "
            "for agreement analysis -- there is only one reader's worth."
        )

    # Longest suffix first, so "_classimg_nonconvex" wins over "".
    ordered = sorted(label_suffixes, key=len, reverse=True)

    by_core: dict[str, dict[str, str]] = {}
    for annotator in subdirs:
        for path in sorted(glob.glob(os.path.join(annotations_dir, annotator, pattern))):
            stem = os.path.splitext(os.path.basename(path))[0]
            core_id = stem
            for suffix in ordered:
                if suffix and stem.endswith(suffix):
                    core_id = stem[: -len(suffix)]
                    break
            by_core.setdefault(core_id, {})[annotator] = path

    return AnnotationSet(by_core=by_core, annotators=subdirs, root=annotations_dir)


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------
def coverage_report(ann: AnnotationSet,
                    labelled_core_ids: Sequence[str] | None = None) -> dict:
    """Who annotated what, and how it lines up with the cores being trained on."""
    per_annotator = {a: 0 for a in ann.annotators}
    for readers in ann.by_core.values():
        for a in readers:
            per_annotator[a] += 1

    histogram: dict[int, int] = {}
    for readers in ann.by_core.values():
        histogram[len(readers)] = histogram.get(len(readers), 0) + 1

    out: dict = {
        "root": ann.root,
        "n_annotators": len(ann.annotators),
        "annotators": ann.annotators,
        "cores_per_annotator": per_annotator,
        "n_distinct_cores": len(ann.by_core),
        "cores_by_annotator_count": dict(sorted(histogram.items())),
        "n_cores_with_2plus": len(ann.core_ids(2)),
        "n_cores_with_all": len(ann.core_ids(len(ann.annotators))),
    }

    if labelled_core_ids is not None:
        labelled = set(labelled_core_ids)
        annotated = set(ann.by_core)
        out["n_labelled_cores"] = len(labelled)
        out["labelled_without_annotations"] = sorted(labelled - annotated)
        out["annotated_beyond_labelled"] = sorted(annotated - labelled)
        out["n_labelled_without_annotations"] = len(out["labelled_without_annotations"])
        out["n_annotated_beyond_labelled"] = len(out["annotated_beyond_labelled"])
    return out


def format_coverage(report: dict) -> str:
    lines = [
        f"{report['n_annotators']} annotators over {report['n_distinct_cores']} distinct cores",
        "",
        "cores annotated by each reader:",
    ]
    for a, n in report["cores_per_annotator"].items():
        lines.append(f"  {a:<16}{n:>5}")
    total = report["n_distinct_cores"]
    partial = [a for a, n in report["cores_per_annotator"].items()
               if total and n < 0.9 * total]
    if partial:
        lines.append("")
        lines.append("  readers who did NOT annotate most cores: "
                     + ", ".join(partial))

    lines += ["", "cores by number of annotators:"]
    for k, n in report["cores_by_annotator_count"].items():
        lines.append(f"  {k} annotator(s): {n:>5} cores")
    lines.append(f"\n{report['n_cores_with_2plus']} cores have 2+ annotators "
                 "(agreement is computable only on these)")

    if len(report["cores_by_annotator_count"]) > 1:
        lines += [
            "",
            "READER COUNT VARIES ACROSS CORES. Three consequences:",
            "  1. The majority threshold is not one number -- 3 of 4 on some",
            "     cores, 4 of 6 on others. consensus_manifest() records which",
            "     rule each core got.",
            "  2. Any claim of the form 'consensus of N pathologists' is wrong",
            "     for most cores and must be replaced by the distribution.",
            "  3. Cores with an even reader count can tie; odd ones cannot, so",
            "     the tie policy applies unevenly. Report how many cores it",
            "     actually touched.",
        ]

    if "n_labelled_cores" in report:
        lines += [
            "",
            f"cores currently being trained on: {report['n_labelled_cores']}",
            f"  of those, with no annotation maps: {report['n_labelled_without_annotations']}",
            f"  annotated cores NOT in the training pool: {report['n_annotated_beyond_labelled']}",
        ]
        if report["n_annotated_beyond_labelled"]:
            lines.append("  -> these may be usable data the project has not drawn on;")
            lines.append("     check whether the matching images exist (imagesTs).")
    return "\n".join(lines)


def readers_by_group(ann: AnnotationSet,
                     core_ids: Sequence[str] | None = None,
                     group_regex: str = r"^(?P<group>slide\d+)") -> dict:
    """Cross-tabulate reader count against slide.

    WHY THIS MATTERS HERE
    ---------------------
    The experiment design holds out one slide per fold.  If reader coverage
    is uneven across slides -- if, say, one pathologist read only slide001 --
    then label quality varies by slide too, and a fold's test score is partly
    a measure of how well-annotated its held-out slide was rather than how
    well the model generalises.  That would contaminate the across-slide
    standard deviation, which is reported as evidence of robustness.
    """
    import re

    pattern = re.compile(group_regex)
    ids = list(core_ids) if core_ids is not None else list(ann.by_core)

    per_group: dict[str, list[int]] = {}
    reader_group: dict[str, dict[str, int]] = {}
    for core_id in ids:
        readers = ann.by_core.get(core_id, {})
        if not readers:
            continue
        match = pattern.match(core_id)
        group = match.group("group") if match else core_id
        per_group.setdefault(group, []).append(len(readers))
        for r in readers:
            reader_group.setdefault(r, {})
            reader_group[r][group] = reader_group[r].get(group, 0) + 1

    stats = {}
    for group, counts in sorted(per_group.items()):
        arr = np.array(counts, dtype=float)
        stats[group] = {
            "n_cores": len(counts),
            "mean_readers": float(arr.mean()),
            "min_readers": int(arr.min()),
            "max_readers": int(arr.max()),
        }

    means = np.array([s["mean_readers"] for s in stats.values()], dtype=float)
    spread = float(means.max() - means.min()) if means.size else 0.0

    # Two distinct patterns make label quality track the fold, and they need
    # separating because they differ in severity.
    #
    # ABSENT: a reader who did not touch some slide at all. This is the
    # serious one -- that reader's judgement is structurally missing from
    # those folds.
    absent = sorted(r for r, g in reader_group.items() if len(g) < len(stats))

    # UNEVEN: a reader present everywhere but covering very different
    # PROPORTIONS of each slide. Milder, but it is what actually moves the
    # mean reader count between slides, so it belongs in the report.
    proportional: dict[str, dict] = {}
    for reader, groups in reader_group.items():
        fracs = {g: groups.get(g, 0) / stats[g]["n_cores"] for g in stats}
        vals = np.array(list(fracs.values()), dtype=float)
        proportional[reader] = {
            "by_group": {g: round(f, 3) for g, f in sorted(fracs.items())},
            "min": float(vals.min()),
            "max": float(vals.max()),
            "spread": float(vals.max() - vals.min()),
        }
    uneven = sorted(r for r, p in proportional.items() if p["spread"] > 0.25)

    # Bands rather than a yes/no: 0.63 of a reader is not 2.0 of a reader,
    # and collapsing both to "unbalanced" would overstate the first.
    if spread < 0.30 and not absent:
        severity = "even"
    elif spread < 1.00 and not absent:
        severity = "mild"
    else:
        severity = "material"

    return {
        "per_group": stats,
        "reader_by_group": {r: dict(sorted(g.items()))
                            for r, g in sorted(reader_group.items())},
        "reader_proportional_coverage": proportional,
        "mean_reader_spread": spread,
        "readers_missing_from_some_slides": absent,
        "readers_with_uneven_proportional_coverage": uneven,
        "severity": severity,
        "balanced": severity == "even",
    }


def format_readers_by_group(report: dict) -> str:
    lines = [f"{'slide':<12}{'cores':>7}{'mean readers':>15}{'min':>6}{'max':>6}",
             "-" * 46]
    for group, s in report["per_group"].items():
        lines.append(f"{group:<12}{s['n_cores']:>7}{s['mean_readers']:>15.2f}"
                     f"{s['min_readers']:>6}{s['max_readers']:>6}")

    lines += ["", "cores read per slide, by pathologist:"]
    for reader, groups in report["reader_by_group"].items():
        cells = "  ".join(f"{g}={n}" for g, n in groups.items())
        lines.append(f"  {reader:<8}{cells}")

    uneven = report["readers_with_uneven_proportional_coverage"]
    if uneven:
        lines += ["", "readers whose share of each slide varies widely:"]
        for r in uneven:
            p = report["reader_proportional_coverage"][r]
            lines.append(f"  {r:<8}{p['min']:.0%} to {p['max']:.0%} of a slide's cores")

    spread = report["mean_reader_spread"]
    severity = report["severity"]
    lines += ["", f"spread in mean readers across slides: {spread:.2f} "
                  f"({severity})"]

    if severity == "even":
        lines += [
            "",
            "Coverage is even across slides. State that in the methods: it",
            "rules out the possibility that per-fold differences reflect",
            "label quality rather than generalisation, and it costs one",
            "sentence.",
        ]
    elif severity == "mild":
        lines += [
            "",
            f"Mild imbalance. Every pathologist contributed to every slide, so",
            "no fold is missing a reader's judgement entirely -- but mean",
            f"reader count still varies by {spread:.2f} across slides, enough",
            "that it should not go unmentioned.",
            "",
            "  1. Report mean readers per fold beside each fold's score, so a",
            "     reader can see whether the two move together.",
            "  2. Step 5 tests the correlation directly. A weak one lets you",
            "     say the confound was checked and not observed, which is a",
            "     stronger sentence than silence.",
            "  3. Optional: uniform_reader_subset() finds the largest set of",
            "     pathologists who read nearly every core, giving one fixed",
            "     majority rule across the whole dataset. Report it as a",
            "     sensitivity analysis rather than the primary result.",
        ]
    else:
        lines += [
            "",
            "COVERAGE IS MATERIALLY UNEVEN ACROSS SLIDES. Under",
            "leave-one-slide-out this is a confound: a fold's test score",
            "partly reflects how well its held-out slide was annotated, so",
            "the across-slide SD is not a clean robustness figure.",
        ]
        if report["readers_missing_from_some_slides"]:
            lines.append("  readers absent from some slides entirely: "
                         + ", ".join(report["readers_missing_from_some_slides"]))
        lines += [
            "  Options, in order of preference:",
            "    1. Report mean readers per fold alongside each fold's score.",
            "    2. Check whether fold Dice correlates with mean reader count",
            "       across folds; if it does not, the confound is present in",
            "       principle but not in effect, and that is worth stating.",
            "    3. Restrict to a uniform reader subset via",
            "       uniform_reader_subset(), or to cores meeting a minimum",
            "       reader count via min_annotators, disclosing what drops.",
            "  Do not simply report the SD without mentioning this.",
        ]
    return "\n".join(lines)


def uniform_reader_subset(ann: AnnotationSet,
                          core_ids: Sequence[str] | None = None) -> list[dict]:
    """Trade-off curve: how many cores survive if you demand k fixed readers.

    With reader counts varying from three to six, the majority threshold
    changes core by core, which is awkward to describe and slightly awkward
    to defend.  An alternative is to pick the k pathologists with the widest
    coverage and keep only cores all k of them read: one fixed rule
    (e.g. 3 of 4) across the whole dataset, at the cost of a few cores.

    Returns one row per k, each naming the readers and the number of cores
    retained, so the trade-off is visible rather than guessed.  Feed the
    chosen set to `build_consensus(..., annotators=[...])`.

    This is a sensitivity analysis, not a replacement for the primary
    labels: report it alongside, and say the conclusions do not change.
    """
    ids = list(core_ids) if core_ids is not None else list(ann.by_core)
    coverage = {a: 0 for a in ann.annotators}
    for core_id in ids:
        for a in ann.by_core.get(core_id, {}):
            coverage[a] += 1

    ranked = sorted(ann.annotators, key=lambda a: -coverage[a])
    rows = []
    for k in range(2, len(ranked) + 1):
        chosen = ranked[:k]
        kept = [c for c in ids
                if all(a in ann.by_core.get(c, {}) for a in chosen)]
        rows.append({
            "k": k,
            "readers": chosen,
            "n_cores_retained": len(kept),
            "fraction_retained": len(kept) / len(ids) if ids else 0.0,
            "threshold": f"{k // 2 + 1} of {k}",
            "can_tie": k % 2 == 0,
        })
    return rows


def format_uniform_subset(rows: list[dict], total: int) -> str:
    head = f"{'k':>3}  {'readers':<34}{'cores kept':>12}{'rule':>10}{'ties?':>7}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['k']:>3}  {','.join(r['readers'])[:33]:<34}"
            f"{r['n_cores_retained']:>7} ({r['fraction_retained']:.0%})"
            f"{r['threshold']:>10}{'yes' if r['can_tie'] else 'no':>7}"
        )
    odd = [r for r in rows if not r["can_tie"] and r["fraction_retained"] > 0.9]
    lines.append("")
    if odd:
        best = max(odd, key=lambda r: r["n_cores_retained"])
        lines += [
            f"Best odd-sized option: {best['k']} readers "
            f"({', '.join(best['readers'])}),",
            f"keeping {best['n_cores_retained']} of {total} cores under a fixed "
            f"{best['threshold']} rule with no ties possible.",
            "An odd subset removes the tie question entirely, which is one",
            "fewer thing to explain.",
        ]
    else:
        lines += [
            "No odd-sized subset retains most cores, so ties cannot be",
            "designed away here -- report the tie policy and how many cores",
            "it touched instead.",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _load_map(path: str, shape: tuple[int, int] | None = None,
              step: int = 1) -> np.ndarray:
    """Load one grade map as uint8, optionally subsampled on the way in.

    `step` subsamples BEFORE anything else touches the array. A TMA core is
    around 5000x5000, so a six-reader stack is 150 MB as uint8 and 600 MB
    the moment something promotes it to float32. On a machine with under a
    gigabyte free that is the difference between running and thrashing, so
    subsampling at load rather than afterwards is not a micro-optimisation.
    """
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    if shape is not None and arr.shape != shape:
        arr = np.array(
            Image.fromarray(arr).resize((shape[1], shape[0]), Image.NEAREST)
        )
    if step > 1:
        arr = arr[::step, ::step]
    return np.ascontiguousarray(arr, dtype=np.uint8)


def load_core_maps(ann: AnnotationSet, core_id: str,
                   annotators: Sequence[str] | None = None,
                   step: int = 1,
                   ) -> tuple[list[str], np.ndarray]:
    """Return (annotator names, stacked grade maps of shape (R, H, W)).

    Maps are resampled to the first reader's shape with nearest-neighbour if
    they disagree, which happens in practice and would otherwise crash the
    stack.  Grade values are never interpolated.

    `step` subsamples every map as it loads. Pass it for agreement
    statistics on a memory-constrained machine; leave it at 1 whenever the
    mask itself is the output.
    """
    readers = ann.by_core.get(core_id)
    if not readers:
        raise KeyError(f"No annotation maps for core {core_id!r}")
    names = sorted(readers if annotators is None
                   else [a for a in annotators if a in readers])
    if not names:
        raise KeyError(f"None of {annotators} annotated {core_id!r}")

    # The reference shape is read at FULL resolution, so a reader whose map
    # differs in size is resampled to match before subsampling, not after.
    first_full = np.array(Image.open(readers[names[0]]))
    if first_full.ndim == 3:
        first_full = first_full[..., 0]
    shape = first_full.shape
    first = first_full[::step, ::step] if step > 1 else first_full
    del first_full

    stack = [np.ascontiguousarray(first, dtype=np.uint8)]
    for n in names[1:]:
        stack.append(_load_map(readers[n], shape, step))
    return names, np.stack(stack, axis=0)


# --------------------------------------------------------------------------
# Agreement
# --------------------------------------------------------------------------
def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    total = float(a.sum() + b.sum())
    if total == 0.0:
        return 1.0          # both readers called the whole core benign
    return 2.0 * inter / total


def _cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two raters on a binary pixel labelling."""
    n = a.size
    po = float((a == b).sum()) / n
    pa1, pb1 = float(a.sum()) / n, float(b.sum()) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def _fleiss_kappa_binary(votes: np.ndarray, n_raters: int) -> float:
    """Fleiss' kappa from a per-pixel count of 'tumour' votes.

    `votes` holds, for each pixel, how many of `n_raters` called it tumour.
    """
    if n_raters < 2:
        return float("nan")
    k = votes.astype(np.float64).ravel()
    n = float(n_raters)
    # Per-pixel agreement: (sum_j n_ij^2 - n) / (n(n-1))
    p_i = (k ** 2 + (n - k) ** 2 - n) / (n * (n - 1))
    p_bar = float(p_i.mean())
    p1 = float(k.sum()) / (k.size * n)
    p_e = p1 ** 2 + (1 - p1) ** 2
    if p_e >= 1.0:
        return 1.0 if p_bar >= 1.0 else 0.0
    return (p_bar - p_e) / (1 - p_e)


@dataclass
class AgreementResult:
    core_ids: list[str]
    pairwise_dice: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    pairwise_kappa: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    fleiss: list[float] = field(default_factory=list)
    per_core_fleiss: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict:
        def stat(vals):
            v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
            if v.size == 0:
                return {"mean": float("nan"), "sd": float("nan"), "n": 0}
            return {"mean": float(v.mean()),
                    "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
                    "n": int(v.size)}

        return {
            "n_cores": len(self.core_ids),
            "pairwise_dice": {f"{a} vs {b}": stat(v)
                              for (a, b), v in sorted(self.pairwise_dice.items())},
            "pairwise_kappa": {f"{a} vs {b}": stat(v)
                               for (a, b), v in sorted(self.pairwise_kappa.items())},
            "mean_pairwise_dice": stat(
                [x for v in self.pairwise_dice.values() for x in v]),
            "mean_pairwise_kappa": stat(
                [x for v in self.pairwise_kappa.values() for x in v]),
            "fleiss_kappa": stat(self.fleiss),
        }


def agreement_report(ann: AnnotationSet, cfg: DataConfig,
                     core_ids: Sequence[str] | None = None,
                     max_cores: int | None = None,
                     downsample: int = 4,
                     progress=None) -> AgreementResult:
    """Pairwise Dice/kappa and Fleiss' kappa over cores with 2+ annotators.
    """
    ids = list(core_ids) if core_ids is not None else ann.core_ids(2)
    if max_cores is not None:
        ids = ids[:max_cores]

    out = AgreementResult(core_ids=ids)
    step = max(1, int(downsample))

    for core_id in ids:
        # Subsample on load, not after binarising: binarise_mask returns
        # float32, so doing it the other way round would briefly hold a
        # six-reader stack at 600 MB.
        names, stack = load_core_maps(ann, core_id, step=step)
        if len(names) < 2:
            del stack
            continue
        binary = np.stack([
            binarise_mask(stack[i], cfg.tumour_grades).astype(bool)
            for i in range(len(names))
        ], axis=0)
        del stack

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                key = (names[i], names[j])
                out.pairwise_dice.setdefault(key, []).append(
                    _dice(binary[i], binary[j]))
                out.pairwise_kappa.setdefault(key, []).append(
                    _cohen_kappa(binary[i], binary[j]))

        kappa = _fleiss_kappa_binary(binary.sum(axis=0), len(names))
        out.fleiss.append(kappa)
        out.per_core_fleiss[core_id] = kappa
        del binary

        if progress is not None:
            progress(core_id, len(out.fleiss), len(ids))

    return out


def format_agreement(result: AgreementResult) -> str:
    s = result.summary()
    lines = [
        f"inter-annotator agreement over {s['n_cores']} cores with 2+ readers",
        "",
        f"  mean pairwise Dice   {s['mean_pairwise_dice']['mean']:.4f} "
        f"(SD {s['mean_pairwise_dice']['sd']:.4f})",
        f"  mean pairwise kappa  {s['mean_pairwise_kappa']['mean']:.4f} "
        f"(SD {s['mean_pairwise_kappa']['sd']:.4f})",
        f"  Fleiss' kappa        {s['fleiss_kappa']['mean']:.4f} "
        f"(SD {s['fleiss_kappa']['sd']:.4f})",
        "",
        "by reader pair:",
    ]
    for pair, st in s["pairwise_dice"].items():
        kap = s["pairwise_kappa"].get(pair, {}).get("mean", float("nan"))
        lines.append(f"  {pair:<24} Dice {st['mean']:.4f}  kappa {kap:.4f}  "
                     f"({st['n']} cores)")
    lines += [
        "",
        "The mean pairwise Dice is a practical ceiling: a model scoring near",
        "it is at the limit of what these labels can distinguish, and the gap",
        "to 1.0 is observer variability rather than model error.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Consensus construction
# --------------------------------------------------------------------------
@dataclass
class ConsensusResult:
    core_id: str
    mask: np.ndarray             # float32, 0.0 / 1.0
    n_annotators: int
    annotators: list[str]
    vote_order: str
    tie_policy: str
    n_tie_pixels: int
    tie_fraction: float
    foreground_fraction: float
    threshold: str = ""          # e.g. "3 of 4" -- varies with reader count

    def summary(self) -> dict:
        return {
            "core_id": self.core_id,
            "n_annotators": self.n_annotators,
            "annotators": self.annotators,
            "threshold": self.threshold,
            "vote_order": self.vote_order,
            "tie_policy": self.tie_policy,
            "n_tie_pixels": self.n_tie_pixels,
            "tie_fraction": round(self.tie_fraction, 6),
            "foreground_fraction": round(self.foreground_fraction, 6),
        }


def build_consensus(ann: AnnotationSet, core_id: str, cfg: DataConfig,
                    vote_order: VoteOrder = "binarise_then_vote",
                    tie_policy: TiePolicy = "benign",
                    annotators: Sequence[str] | None = None,
                    min_annotators: int = 1) -> ConsensusResult:
    """Majority-vote consensus mask under an explicit, recorded procedure.

    READER COUNTS VARY BY CORE in this dataset -- not every pathologist read
    every core -- so the majority threshold is not a fixed number. A core
    with four readers needs 3 of 4; one with six needs 4 of 6; one with five
    needs 3 of 5 and cannot tie at all. `ConsensusResult.threshold` records
    which rule was applied, and `consensus_manifest()` tabulates it per core.
    Describing the procedure as "majority vote of six pathologists" would be
    wrong for most cores here.
    """
    names, stack = load_core_maps(ann, core_id, annotators)
    n = len(names)

    if n < min_annotators:
        raise ValueError(
            f"Core {core_id!r} has {n} annotator(s), below min_annotators="
            f"{min_annotators}. Either lower the threshold and disclose that "
            "some consensus masks rest on few readers, or exclude this core "
            "and record the exclusion criterion."
        )

    if vote_order == "binarise_then_vote":
        binary = np.stack([binarise_mask(stack[i], cfg.tumour_grades)
                           for i in range(n)], axis=0)
        votes = binary.sum(axis=0)
        tie = (votes * 2 == n) if n % 2 == 0 else np.zeros(votes.shape, dtype=bool)
        mask = (votes * 2 > n)
        if tie_policy == "tumour":
            mask = mask | tie

    elif vote_order == "vote_then_binarise":
        # Modal grade per pixel, then collapse. Ties in the modal grade are
        # broken toward the LOWER grade, which is the conservative reading
        # and must be stated alongside the tie policy.
        values = np.unique(stack)
        counts = np.stack([(stack == v).sum(axis=0) for v in values], axis=0)
        best = counts.max(axis=0)
        # argmax takes the first maximum, and `values` is ascending, so this
        # is the lowest grade among those tied for the mode.
        modal = values[counts.argmax(axis=0)]
        tie = (counts == best).sum(axis=0) > 1
        mask = binarise_mask(modal, cfg.tumour_grades).astype(bool)
        if tie_policy == "tumour":
            any_tumour = np.stack(
                [binarise_mask(stack[i], cfg.tumour_grades) for i in range(n)],
                axis=0).any(axis=0)
            mask = mask | (tie & any_tumour)

    else:
        raise ValueError(
            f"Unknown vote_order {vote_order!r}; expected "
            '"binarise_then_vote" or "vote_then_binarise"'
        )

    total = float(mask.size)
    need = n // 2 + 1
    threshold = (f"{need} of {n}" if vote_order == "binarise_then_vote"
                 else f"modal grade over {n}")
    return ConsensusResult(
        core_id=core_id,
        mask=mask.astype(np.float32),
        n_annotators=n,
        annotators=names,
        vote_order=vote_order,
        tie_policy=tie_policy,
        n_tie_pixels=int(tie.sum()),
        tie_fraction=float(tie.sum()) / total,
        foreground_fraction=float(mask.sum()) / total,
        threshold=threshold,
    )


def consensus_manifest(ann: AnnotationSet, cfg: DataConfig,
                       core_ids: Sequence[str] | None = None,
                       vote_order: VoteOrder = "binarise_then_vote",
                       tie_policy: TiePolicy = "benign",
                       min_annotators: int = 1) -> list[dict]:
    """Per-core record of how each consensus mask was built.
    """
    ids = list(core_ids) if core_ids is not None else ann.core_ids(min_annotators)
    rows = []
    for core_id in ids:
        con = build_consensus(ann, core_id, cfg, vote_order, tie_policy,
                              min_annotators=min_annotators)
        row = con.summary()
        row["annotators"] = ",".join(con.annotators)
        rows.append(row)
    return rows


def summarise_manifest(rows: list[dict]) -> str:
    """Fold the per-core manifest into the few numbers the methods needs."""
    if not rows:
        return "no cores"
    counts: dict[int, int] = {}
    thresholds: dict[str, int] = {}
    for r in rows:
        counts[r["n_annotators"]] = counts.get(r["n_annotators"], 0) + 1
        thresholds[r["threshold"]] = thresholds.get(r["threshold"], 0) + 1

    ties = np.array([r["tie_fraction"] for r in rows], dtype=float)
    fg = np.array([r["foreground_fraction"] for r in rows], dtype=float)
    n_tied = int((ties > 0).sum())

    lines = [f"{len(rows)} cores", "", "readers per core:"]
    for k in sorted(counts):
        lines.append(f"  {k} reader(s): {counts[k]:>4} cores")
    lines += ["", "majority threshold applied:"]
    for t in sorted(thresholds):
        lines.append(f"  {t:<22}{thresholds[t]:>4} cores")
    lines += [
        "",
        f"cores with any tied pixels: {n_tied} of {len(rows)}",
        f"tied pixels, mean {ties.mean():.4%}, max {ties.max():.4%} of a core",
        f"consensus tumour fraction, mean {fg.mean():.4f} "
        f"(range {fg.min():.4f}-{fg.max():.4f})",
    ]
    if len(counts) > 1:
        lines += [
            "",
            "Reader count VARIES across cores, so there is no single majority",
            "rule to quote. Report the distribution above, not a sentence of",
            "the form 'consensus of N pathologists'.",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Reproducibility check against the labels already in use
# --------------------------------------------------------------------------
def compare_to_existing(ann: AnnotationSet, cfg: DataConfig,
                        existing: dict[str, str],
                        vote_orders: Sequence[VoteOrder] = (
                            "binarise_then_vote", "vote_then_binarise"),
                        tie_policies: Sequence[TiePolicy] = ("benign", "tumour"),
                        max_cores: int = 15) -> list[dict]:
    """Can the label PNGs in use be regenerated from the annotation maps?

    `existing` maps core_id -> path of the label the project already trains
    on.  For each combination of vote order and tie policy this regenerates
    the consensus and reports the Dice against that label.

    A combination scoring ~1.000 identifies exactly how the labels were
    built, which turns the consensus question from an awkward one into a
    documented, verifiable procedure.  If nothing scores near 1.000, the
    labels came from some other process and that needs establishing before
    submission rather than after.
    """
    ids = [c for c in ann.core_ids(1) if c in existing][:max_cores]
    if not ids:
        raise ValueError(
            "No core appears in both the annotation tree and the existing "
            "label set, so they cannot be compared. Check that core ids match "
            "after the filename suffix is stripped."
        )

    rows: list[dict] = []
    for order in vote_orders:
        for policy in tie_policies:
            scores, ties = [], []
            for core_id in ids:
                con = build_consensus(ann, core_id, cfg,
                                      vote_order=order, tie_policy=policy)
                ref_raw = _load_map(existing[core_id], con.mask.shape)
                ref = binarise_mask(ref_raw, cfg.tumour_grades).astype(bool)
                scores.append(_dice(con.mask.astype(bool), ref))
                ties.append(con.tie_fraction)
            arr = np.array(scores, dtype=float)
            rows.append({
                "vote_order": order,
                "tie_policy": policy,
                "n_cores": len(ids),
                "mean_dice_vs_existing": float(arr.mean()),
                "min_dice_vs_existing": float(arr.min()),
                "n_exact": int((arr > 0.9999).sum()),
                "mean_tie_fraction": float(np.mean(ties)),
            })
    return sorted(rows, key=lambda r: -r["mean_dice_vs_existing"])


def format_reproducibility(rows: list[dict]) -> str:
    head = (f"{'vote order':<22}{'ties':<10}{'mean Dice':>11}"
            f"{'min':>9}{'exact':>8}{'tie frac':>11}")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['vote_order']:<22}{r['tie_policy']:<10}"
            f"{r['mean_dice_vs_existing']:>11.4f}"
            f"{r['min_dice_vs_existing']:>9.4f}"
            f"{r['n_exact']:>5}/{r['n_cores']:<2}"
            f"{r['mean_tie_fraction']:>11.5f}"
        )
    best = rows[0]
    exact = [r for r in rows if r["mean_dice_vs_existing"] > 0.9999]
    lines.append("")
    if len(exact) > 1:
        # Several procedures give the same masks on these data. That is a
        # stronger result than a single match, not a weaker one -- but it
        # means the data cannot identify WHICH procedure was used, and the
        # methods section must not claim otherwise.
        combos = ", ".join(f"{r['vote_order']}/{r['tie_policy']}" for r in exact)
        orders = {r["vote_order"] for r in exact}
        policies = {r["tie_policy"] for r in exact}
        lines += [
            f"{len(exact)} procedures reproduce the labels in use exactly: {combos}.",
            "",
            "These data cannot tell you which one was actually run, so do not",
            "name a single procedure as though it were established. The",
            "stronger statement is that the consensus masks are",
            "insensitive to "
            + (" and ".join(
                ([] if len(orders) < 2 else ["vote order"])
                + ([] if len(policies) < 2 else ["tie policy"]))
               or "these choices")
            + " on this dataset,",
            "which removes the question rather than answering it.",
        ]
        if len(policies) > 1:
            lines.append(f"Tied pixels average {best['mean_tie_fraction']:.4%} "
                         "of the core, which is why the tie policy does not bite.")
    elif exact:
        lines += [
            f"The labels in use are reproduced exactly by "
            f"{best['vote_order']} with ties resolved to {best['tie_policy']},",
            "and by no other combination tested. State that procedure in the",
            "methods and the description is complete and verifiable.",
        ]
    elif best["mean_dice_vs_existing"] > 0.95:
        lines += [
            f"Closest match: {best['vote_order']} / ties to {best['tie_policy']} "
            f"at Dice {best['mean_dice_vs_existing']:.4f}, but not exact.",
            "Something else differs -- a grade set, a morphological step, or a",
            "different reader subset. Worth finding before you describe the",
            "procedure in print.",
        ]
    else:
        lines += [
            "No combination reproduces the labels in use. They did not come",
            "from a plain majority vote over these maps. Establish what did",
            "produce them before writing the methods -- describing a",
            "procedure that does not regenerate your own labels is worse than",
            "saying the records are incomplete.",
        ]
    return "\n".join(lines)
