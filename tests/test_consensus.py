"""Exercise the per-pathologist consensus and agreement code.

Builds a synthetic annotations/ tree matching the real layout
(annotations/Map1..Map6/<core>_classimg_nonconvex.png) with known
disagreement, then checks discovery, coverage, agreement, both vote orders,
tie policy, and the reproducibility check.

Run:  python3 -I tests/test_consensus.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                              # noqa: E402
from PIL import Image                                           # noqa: E402

from frozense.config import DataConfig                          # noqa: E402
from frozense.consensus import (                                # noqa: E402
    discover_annotations, coverage_report, format_coverage,
    agreement_report, format_agreement, build_consensus,
    compare_to_existing, format_reproducibility, load_core_maps,
    consensus_manifest, summarise_manifest,
    _fleiss_kappa_binary, _cohen_kappa, _dice,
)

TEST_ROOT = os.environ.get("FROZENSE_TEST_DIR",
                           os.path.join(tempfile.gettempdir(), "frozense_tests"))
ROOT = os.path.join(TEST_ROOT, "anntest")
ANN = os.path.join(ROOT, "annotations")
EXISTING = os.path.join(ROOT, "Generated_Labels_Train")
H = W = 120
SUFFIX = "_classimg_nonconvex"


def write(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def build_tree():
    """Six readers, 10 cores, deliberate and known disagreement."""
    shutil.rmtree(ROOT, ignore_errors=True)
    rng = np.random.default_rng(7)
    readers = [f"Map{i}" for i in range(1, 7)]

    for c in range(10):
        core = f"slide001_core{c:03d}"
        # A shared tumour region all readers find, plus a per-reader fringe.
        base = np.zeros((H, W), dtype=np.uint8)
        base[30:80, 30:80] = 4                      # agreed Gleason 4 core
        for r, reader in enumerate(readers):
            # Uneven coverage, as in the real tree: Map6 reads only the first
            # five cores, and core 9 was read by two pathologists only. So
            # reader counts per core are 6, 5 and 2 -- which is what makes the
            # majority threshold vary.
            if c == 9 and r >= 2:
                continue
            if c >= 5 and reader == "Map6":
                continue
            m = base.copy()
            # Fringe each reader draws slightly differently.
            dy, dx = rng.integers(-6, 7, size=2)
            y0, x0 = 80 + dy, 30 + dx
            m[max(0, y0):y0 + 14, max(0, x0):x0 + 40] = 3
            # Readers 4 and 5 call part of the agreed region grade 5 instead
            # of 4 -- same tumour, different grade.
            if r >= 4:
                m[30:55, 30:80] = 5
            # A band where the two vote orders MUST diverge: four readers
            # call tumour but spread across three different grades, two call
            # it benign. Majority-on-binary says tumour (4 of 6); modal grade
            # is a tie between benign and grade 3, broken low, so
            # vote_then_binarise says benign.
            m[95:105, 10:110] = [0, 0, 3, 3, 4, 5][r]
            write(os.path.join(ANN, reader, core + SUFFIX + ".png"), m)

    # One core annotated but NOT in the training pool, as in the real tree.
    extra = np.zeros((H, W), dtype=np.uint8)
    extra[10:40, 10:40] = 4
    for reader in readers:
        write(os.path.join(ANN, reader, "slide004_core001" + SUFFIX + ".png"), extra)

    return readers


def write_existing(ann, cfg, order, policy):
    """Save a consensus built a known way, to be rediscovered later."""
    paths = {}
    for core in ann.core_ids(1):
        if core.startswith("slide004"):
            continue
        con = build_consensus(ann, core, cfg, vote_order=order, tie_policy=policy)
        # Stored as a grade map, like the real labels: tumour pixels get a
        # value inside tumour_grades, background gets 0.
        grade = (con.mask > 0.5).astype(np.uint8) * 4
        p = os.path.join(EXISTING, core + SUFFIX + ".png")
        write(p, grade)
        paths[core] = p
    return paths


def build_spec_tree(root: str, spec: dict[str, list[str]]):
    """spec maps core_id -> list of reader names. Minimal maps, for coverage
    questions where the pixel content is irrelevant."""
    shutil.rmtree(root, ignore_errors=True)
    m = np.zeros((16, 16), dtype=np.uint8)
    m[4:12, 4:12] = 4
    for core_id, readers in spec.items():
        for r in readers:
            write(os.path.join(root, r, core_id + SUFFIX + ".png"), m)
    return discover_annotations(root)


def check_readers_by_group():
    from frozense.consensus import readers_by_group, format_readers_by_group

    six = [f"Map{i}" for i in range(1, 7)]
    root = os.path.join(ROOT, "bygroup")

    # Balanced: every slide read by the same readers.
    spec = {f"slide{s:03d}_core{c:03d}": six
            for s in (1, 2, 3) for c in range(4)}
    rep = readers_by_group(build_spec_tree(root, spec))
    check("balanced coverage is reported as balanced", rep["balanced"],
          f"spread {rep['mean_reader_spread']:.2f}")
    check("  no reader flagged as missing from slides",
          rep["readers_missing_from_some_slides"] == [])
    check("  three slides tabulated", len(rep["per_group"]) == 3)
    check("  and the text says the confound is ruled out",
          "Coverage is even across slides" in format_readers_by_group(rep))

    # Confounded: Map5 and Map6 read slide001 only, so slide001's labels rest
    # on six readers while the others rest on four.
    spec = {}
    for s in (1, 2, 3):
        for c in range(4):
            spec[f"slide{s:03d}_core{c:03d}"] = six if s == 1 else six[:4]
    rep = readers_by_group(build_spec_tree(root, spec))
    check("uneven coverage is caught", not rep["balanced"],
          f"spread {rep['mean_reader_spread']:.2f}")
    check("  names the readers confined to some slides",
          rep["readers_missing_from_some_slides"] == ["Map5", "Map6"],
          str(rep["readers_missing_from_some_slides"]))
    check("  slide001 has more readers than the rest",
          rep["per_group"]["slide001"]["mean_readers"] == 6.0
          and rep["per_group"]["slide002"]["mean_readers"] == 4.0)
    check("  graded as material", rep["severity"] == "material")
    text = format_readers_by_group(rep)
    check("  and the text calls it a confound",
          "COVERAGE IS MATERIALLY UNEVEN" in text
          and "clean robustness figure" in text)

    # Mild: every reader on every slide, but one covers very different
    # proportions of each. This is the shape the real data has.
    spec = {}
    partial = {1: 8, 2: 5, 3: 2}
    for s in (1, 2, 3):
        for c in range(10):
            readers = six[:4] + (["Map5"] if c < partial[s] else [])
            spec[f"slide{s:03d}_core{c:03d}"] = readers
    rep = readers_by_group(build_spec_tree(root, spec))
    check("mild imbalance is graded mild", rep["severity"] == "mild",
          f"{rep['severity']}, spread {rep['mean_reader_spread']:.2f}")
    check("  not reported as balanced", not rep["balanced"])
    check("  no reader absent from a whole slide",
          rep["readers_missing_from_some_slides"] == [])
    check("  but the uneven-proportion reader is named",
          rep["readers_with_uneven_proportional_coverage"] == ["Map5"],
          str(rep["readers_with_uneven_proportional_coverage"]))
    mild_text = format_readers_by_group(rep)
    check("  text distinguishes mild from material",
          "Mild imbalance" in mild_text
          and "MATERIALLY UNEVEN" not in mild_text)
    check("  and points at the step-5 correlation check",
          "checked and not observed" in mild_text)
    print()
    print("    " + mild_text.replace("\n", "\n    "))
    shutil.rmtree(root, ignore_errors=True)


def check_uniform_subset():
    from frozense.consensus import uniform_reader_subset, format_uniform_subset

    root = os.path.join(ROOT, "subset")
    six = [f"Map{i}" for i in range(1, 7)]
    # Mirrors the real shape: two complete readers, two near-complete, two
    # partial.
    spec = {}
    for c in range(40):
        readers = ["Map1", "Map5"]
        if c < 38:
            readers.append("Map3")
        if c < 37:
            readers.append("Map4")
        if c < 20:
            readers.append("Map2")
        if c < 10:
            readers.append("Map6")
        spec[f"slide001_core{c:03d}"] = readers
    ann = build_spec_tree(root, spec)

    rows = uniform_reader_subset(ann)
    check("one row per subset size", [r["k"] for r in rows] == [2, 3, 4, 5, 6])
    check("complete readers come first",
          set(rows[0]["readers"]) == {"Map1", "Map5"}, str(rows[0]["readers"]))
    check("retention falls as k grows",
          all(rows[i]["n_cores_retained"] >= rows[i + 1]["n_cores_retained"]
              for i in range(len(rows) - 1)))
    check("k=2 keeps every core", rows[0]["n_cores_retained"] == 40)
    check("k=4 keeps the near-complete set", rows[2]["n_cores_retained"] == 37,
          str(rows[2]["n_cores_retained"]))
    check("k=6 collapses to the rarest reader", rows[4]["n_cores_retained"] == 10)
    check("tie-ability tracks parity",
          [r["can_tie"] for r in rows] == [True, False, True, False, True])
    check("thresholds are stated",
          [r["threshold"] for r in rows]
          == ["2 of 2", "2 of 3", "3 of 4", "3 of 5", "4 of 6"])
    text = format_uniform_subset(rows, total=40)
    check("recommends an odd subset where one retains most cores",
          "Best odd-sized option: 3 readers" in text, text.splitlines()[-4:][0])
    print()
    print("    " + text.replace("\n", "\n    "))
    shutil.rmtree(root, ignore_errors=True)


def check(label, condition, detail=""):
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        raise AssertionError(label + " " + detail)


def main() -> int:
    readers = build_tree()
    cfg = DataConfig(tumour_grades=(3, 4, 5))

    print("discovery")
    ann = discover_annotations(ANN)
    check("found six annotators", ann.annotators == readers, str(ann.annotators))
    check("stripped the filename suffix",
          "slide001_core000" in ann.by_core, sorted(ann.by_core)[:2])
    check("11 distinct cores", len(ann.by_core) == 11, str(len(ann.by_core)))
    check("core009 has only two readers", ann.n_annotators("slide001_core009") == 2)

    print("\ncoverage")
    labelled = [f"slide001_core{c:03d}" for c in range(10)]
    cov = coverage_report(ann, labelled)
    check("cores with 2+ readers = 11", cov["n_cores_with_2plus"] == 11)
    check("cores read by all six = 6", cov["n_cores_with_all"] == 6,
          str(cov["n_cores_with_all"]))
    check("reader counts vary (6 / 5 / 2)",
          sorted(cov["cores_by_annotator_count"]) == [2, 5, 6],
          str(cov["cores_by_annotator_count"]))
    check("spots the annotated core outside the training pool",
          cov["annotated_beyond_labelled"] == ["slide004_core001"])
    check("spots no labelled core missing annotations",
          cov["n_labelled_without_annotations"] == 0)
    text = format_coverage(cov)
    check("names the low-coverage reader", "Map6" in text.split("readers who did NOT")[1][:40])
    check("warns that the reader count varies",
          "READER COUNT VARIES ACROSS CORES" in text)
    print()
    print("    " + text.replace("\n", "\n    "))

    print("\nkappa primitives")
    a = np.zeros((10, 10), dtype=bool); a[:5] = True
    check("identical raters give kappa 1", abs(_cohen_kappa(a, a) - 1.0) < 1e-9)
    check("identical raters give Dice 1", abs(_dice(a, a) - 1.0) < 1e-9)
    check("two empty masks agree by convention",
          _dice(np.zeros_like(a), np.zeros_like(a)) == 1.0)
    votes_unan = np.full((10, 10), 3)
    check("unanimous votes give Fleiss 1",
          abs(_fleiss_kappa_binary(votes_unan, 3) - 1.0) < 1e-9)
    # Half the raters say tumour everywhere: maximal disagreement.
    votes_split = np.full((10, 10), 1)
    check("evenly split votes give Fleiss <= 0",
          _fleiss_kappa_binary(votes_split, 2) <= 1e-9,
          f"{_fleiss_kappa_binary(votes_split, 2):.4f}")

    print("\nagreement")
    agr = agreement_report(ann, cfg, downsample=1)
    s = agr.summary()
    check("15 reader pairs", len(s["pairwise_dice"]) == 15, str(len(s["pairwise_dice"])))
    check("mean pairwise Dice is high but not 1 (fringe disagreement)",
          0.80 < s["mean_pairwise_dice"]["mean"] < 0.999,
          f"{s['mean_pairwise_dice']['mean']:.4f}")
    check("Fleiss kappa is finite and positive",
          0 < s["fleiss_kappa"]["mean"] < 1, f"{s['fleiss_kappa']['mean']:.4f}")
    print()
    print("    " + format_agreement(agr).replace("\n", "\n    "))

    print("\nvote orders")
    core = "slide001_core000"
    btv = build_consensus(ann, core, cfg, "binarise_then_vote", "benign")
    vtb = build_consensus(ann, core, cfg, "vote_then_binarise", "benign")
    check("binarise_then_vote records six readers", btv.n_annotators == 6)
    check("both produce non-empty masks",
          btv.foreground_fraction > 0.1 and vtb.foreground_fraction > 0.1,
          f"{btv.foreground_fraction:.4f} / {vtb.foreground_fraction:.4f}")
    agreed = _dice(btv.mask.astype(bool), vtb.mask.astype(bool))
    print(f"    the two vote orders agree at Dice {agreed:.4f} on this core")
    check("vote order is recorded on the result",
          btv.vote_order == "binarise_then_vote" and vtb.vote_order == "vote_then_binarise")
    check("the orders genuinely diverge where grades are split", agreed < 0.999)
    check("binarise_then_vote calls the split-grade band tumour",
          btv.mask[100, 60] == 1.0)
    check("vote_then_binarise calls it benign (modal grade ties low)",
          vtb.mask[100, 60] == 0.0)

    print("\ntie policy")
    # core009 has exactly two readers, so ties are possible and the policy bites.
    t_benign = build_consensus(ann, "slide001_core009", cfg,
                               "binarise_then_vote", "benign")
    t_tumour = build_consensus(ann, "slide001_core009", cfg,
                               "binarise_then_vote", "tumour")
    check("ties exist with an even number of readers", t_benign.n_tie_pixels > 0,
          f"{t_benign.n_tie_pixels} px")
    check("tie fraction is reported",
          0 < t_benign.tie_fraction < 1, f"{t_benign.tie_fraction:.5f}")
    check("tumour policy yields a larger mask than benign",
          t_tumour.foreground_fraction > t_benign.foreground_fraction,
          f"{t_tumour.foreground_fraction:.4f} > {t_benign.foreground_fraction:.4f}")
    check("the difference equals exactly the tied pixels",
          abs((t_tumour.mask.sum() - t_benign.mask.sum()) - t_benign.n_tie_pixels) < 1e-6)
    check("odd reader counts cannot tie",
          build_consensus(ann, core, cfg, annotators=readers[:3]).n_tie_pixels == 0)
    check("five-reader cores cannot tie either",
          build_consensus(ann, "slide001_core006", cfg).n_tie_pixels == 0)

    print("\nvarying majority threshold")
    thresholds = {c: build_consensus(ann, c, cfg).threshold
                  for c in ("slide001_core000", "slide001_core006", "slide001_core009")}
    for c, t in thresholds.items():
        print(f"    {c}: {t}")
    check("threshold tracks the reader count",
          thresholds["slide001_core000"] == "4 of 6"
          and thresholds["slide001_core006"] == "3 of 5"
          and thresholds["slide001_core009"] == "2 of 2",
          str(thresholds))

    try:
        build_consensus(ann, "slide001_core009", cfg, min_annotators=3)
        check("min_annotators guard fires", False)
    except ValueError as exc:
        check("min_annotators guard fires", "below min_annotators" in str(exc))

    print("\nper-core manifest")
    rows = consensus_manifest(ann, cfg)
    check("one row per core", len(rows) == 11, str(len(rows)))
    check("every row records reader count, threshold and tie fraction",
          all({"n_annotators", "threshold", "tie_fraction", "annotators"} <= set(r)
              for r in rows))
    check("annotators are serialisable as text",
          all(isinstance(r["annotators"], str) for r in rows))
    summary = summarise_manifest(rows)
    check("summary warns against a single-number claim",
          "consensus of N pathologists" in summary)
    print()
    print("    " + summary.replace("\n", "\n    "))

    print("\nreproducibility check")
    for truth_order, truth_policy in [("binarise_then_vote", "benign"),
                                      ("vote_then_binarise", "benign"),
                                      ("binarise_then_vote", "tumour")]:
        shutil.rmtree(EXISTING, ignore_errors=True)
        existing = write_existing(ann, cfg, truth_order, truth_policy)
        rows = compare_to_existing(ann, cfg, existing, max_cores=10)
        exact = [r for r in rows if r["mean_dice_vs_existing"] > 0.9999]
        check(f"{truth_order}/{truth_policy}: an exact match is found",
              bool(exact),
              f"best was {rows[0]['vote_order']}/{rows[0]['tie_policy']} "
              f"at {rows[0]['mean_dice_vs_existing']:.4f}")
        check("  and the true procedure is among the exact matches",
              any(r["vote_order"] == truth_order and r["tie_policy"] == truth_policy
                  for r in exact),
              f"exact: {[(r['vote_order'], r['tie_policy']) for r in exact]}")
        check("  every core matches exactly",
              all(r["n_exact"] == r["n_cores"] for r in exact))
        # Ties among exact matches must be reported as ties, not as a verdict.
        text = format_reproducibility(rows)
        if len(exact) > 1:
            check("  several matches are reported as ambiguous, not decided",
                  "cannot tell you which one" in text)
        else:
            check("  a unique match is reported as established",
                  "and by no other combination tested" in text)
    print()
    print("    " + format_reproducibility(rows).replace("\n", "\n    "))

    print("\nreader coverage across slides")
    check_readers_by_group()

    print("\nuniform reader subsets")
    check_uniform_subset()

    print("\nnegative control")
    shutil.rmtree(EXISTING, ignore_errors=True)
    rng = np.random.default_rng(1)
    bogus = {}
    for c in range(6):
        core = f"slide001_core{c:03d}"
        m = (rng.random((H, W)) > 0.5).astype(np.uint8) * 4
        p = os.path.join(EXISTING, core + SUFFIX + ".png")
        write(p, m)
        bogus[core] = p
    rows = compare_to_existing(ann, cfg, bogus, max_cores=6)
    check("random labels are not reproduced",
          rows[0]["mean_dice_vs_existing"] < 0.7,
          f"{rows[0]['mean_dice_vs_existing']:.4f}")
    check("and the report says so",
          "No combination reproduces" in format_reproducibility(rows))

    shutil.rmtree(ROOT, ignore_errors=True)
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
