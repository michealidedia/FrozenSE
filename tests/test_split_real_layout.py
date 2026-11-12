"""Verify the split against the REAL Gleason-2019 layout in this project.

Slide composition measured on the authors' copy of Gleason 2019 (imagesTr +
imagesVal, 193 cores, slide004 absent -- it is part of the unlabelled
imagesTs set):

    slide001  36      slide005  31
    slide002  29      slide006  44
    slide003  28      slide007  25

Run:  python3 -I tests/test_split_real_layout.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frozense.config import SplitConfig, DataConfig            # noqa: E402
from frozense.data import (                                     # noqa: E402
    Core, make_split, n_folds, group_composition,
)

REAL_COUNTS = {
    "slide001": 36, "slide002": 29, "slide003": 28,
    "slide005": 31, "slide006": 44, "slide007": 25,
}
TOTAL = sum(REAL_COUNTS.values())


def build_cores() -> list[Core]:
    cores = []
    for slide, n in REAL_COUNTS.items():
        for i in range(1, n + 1):
            cid = f"{slide}_core{i:03d}"
            cores.append(Core(
                core_id=cid,
                image_path=f"/data/images/{cid}.jpg",
                label_path=f"/data/labels/{cid}_classimg_nonconvex.png",
                group=slide,
            ))
    return cores


def check(label, condition, detail=""):
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        raise AssertionError(label + " " + detail)


def main() -> int:
    cores = build_cores()
    print(f"193-core layout: {TOTAL} cores, {n_folds(cores)} groups\n")
    check("total matches the expected 193", TOTAL == 193, f"got {TOTAL}")
    check("six slides", n_folds(cores) == 6)

    # ---------------------------------------------------------------- folds
    print("\nleave-one-slide-out, n_train_cores=35")
    tested_once: dict[str, int] = {}
    for fold in range(n_folds(cores)):
        cfg = SplitConfig(scheme="leave_one_group_out", fold=fold, n_train_cores=35)
        sp = make_split(cores, cfg)          # __post_init__ runs assert_disjoint

        for c in sp.test:
            tested_once[c.core_id] = tested_once.get(c.core_id, 0) + 1

        print(f"  fold {fold}: test {sp.test_groups[0]} ({len(sp.test):>2} cores)  "
              f"val {sp.val_groups[0]} ({len(sp.val):>2})  "
              f"train {len(sp.train)}/{sp.train_pool_size} from "
              f"{len(sp.train_groups)} slides")

        check(f"  fold {fold} grouping is group-level",
              sp.grouping_used == "group")
        check(f"  fold {fold} train+val+test groups = 6",
              len(sp.train_groups) + len(sp.val_groups) + len(sp.test_groups) == 6)
        check(f"  fold {fold} train pool >= 35", sp.train_pool_size >= 35,
              f"pool={sp.train_pool_size}")
        check(f"  fold {fold} training set is exactly 35", len(sp.train) == 35)

    # ------------------------------------------------- every core tested once
    print("\ncoverage across folds")
    check("every core tested exactly once",
          len(tested_once) == TOTAL and set(tested_once.values()) == {1},
          f"{len(tested_once)} cores, counts {sorted(set(tested_once.values()))}")

    # ------------------------------------------------------- fold stability
    a = make_split(cores, SplitConfig(fold=2, n_train_cores=35))
    b = make_split(cores, SplitConfig(fold=2, n_train_cores=35))
    check("same fold twice gives the same training cores",
          [c.core_id for c in a.train] == [c.core_id for c in b.train])

    c3 = make_split(cores, SplitConfig(fold=3, n_train_cores=35))
    check("different folds draw different training cores",
          [c.core_id for c in a.train] != [c.core_id for c in c3.train])

    # --------------------------------------------------------- manifest keys
    m = a.manifest()
    check("manifest records scheme and fold",
          m["scheme"] == "leave_one_group_out" and m["fold"] == 2)
    check("manifest names the held-out slide", m["groups"]["test"] == ["slide003"])
    check("manifest overlap check passes", "passed" in m["group_overlap_check"])

    # ------------------------------------------- guard rails that must fire
    print("\nguard rails")
    try:
        make_split(cores, SplitConfig(fold=6))
        check("fold out of range raises", False)
    except ValueError as exc:
        check("fold out of range raises", "only 6 groups" in str(exc))

    try:
        make_split(cores, SplitConfig(fold=0, n_train_cores=200))
        check("n_train_cores above pool raises", False)
    except ValueError as exc:
        check("n_train_cores above pool raises", "exceeds the training pool" in str(exc))

    two = [c for c in cores if c.group in ("slide001", "slide002")]
    try:
        make_split(two, SplitConfig(scheme="leave_one_group_out", n_train_cores=10))
        check("fewer than 3 groups raises", False)
    except ValueError as exc:
        check("fewer than 3 groups raises", "at least 3 groups" in str(exc))

    # ------------------------------------------- learning-curve budgets
    # The notebook's budgets must fit every fold's training pool on the real
    # layout, and an impossible budget must be refused before anything trains.
    print("\nlearning-curve budgets against the real training pools")
    import tempfile
    from frozense.config import ExperimentConfig
    from frozense.crossval import run_learning_curve
    pools = {f: make_split(cores, SplitConfig(fold=f)).train_pool_size
             for f in range(n_folds(cores))}
    print(f"  training pool per fold: {pools}")
    with tempfile.TemporaryDirectory() as tmp:
        base = ExperimentConfig(name="proposed_frozense_sam", output_root=tmp)
        rows = run_learning_curve(base, cores, sizes=(5, 10, 20, 35), folds=(0, 2, 4),
                                  train_missing=False, verbose=False)
        check("the notebook's budgets (5, 10, 20, 35) fit folds 0, 2, 4",
              rows == [], f"{len(rows)} rows")
        try:
            run_learning_curve(base, cores, sizes=(5, 130), folds=(0, 2, 4),
                               train_missing=False, verbose=False)
            check("a budget above a fold's pool is refused up front", False)
        except ValueError as exc:
            check("a budget above a fold's pool is refused up front",
                  "[130]" in str(exc) and "fold 0" in str(exc)
                  and f"{pools[0]} cores" in str(exc), str(exc)[:80])

    # ------------------------------------------------------- holdout fallback
    print("\nholdout scheme (kept for comparison)")
    h = make_split(cores, SplitConfig(scheme="holdout", test_frac=0.40,
                                      val_frac=0.20, n_train_cores=35))
    print(f"  test {h.test_groups} ({len(h.test)} cores)  "
          f"val {h.val_groups} ({len(h.val)})  train {len(h.train)}/{h.train_pool_size}")
    check("holdout: 2 test slides", len(h.test_groups) == 2)
    check("holdout: 1 val slide", len(h.val_groups) == 1)
    check("holdout: 3 train slides", len(h.train_groups) == 3)

    # ------------------------------------------------------ composition report
    print("\nslide composition")
    src = {c.core_id: ("imagesTr" if int(c.core_id[-3:]) <= 10 else "imagesVal")
           for c in cores}
    comp = group_composition(cores, src)
    check("composition counts all cores", comp["n_cores"] == TOTAL)
    check("composition detects slides spanning both source folders",
          len(comp["groups_spanning_multiple_sources"]) == 6)
    print(f"  slides in >1 source folder (synthetic test data): "
          f"{comp['groups_spanning_multiple_sources']}")

    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
