"""Preloading must change speed only, never which patches are produced.

Builds synthetic cores (varied sizes including one smaller than a patch,
empty and sparse masks, raw Gleason grade values) and checks that:

  1. preload=True and preload=False yield byte-identical patches;
  2. both match the patch POSITIONS chosen by the previous implementation
     (frozense 1.1.0, passed as an extracted package directory), so
     switching loaders does not silently change the experiment;
  3. preloaded patches are independent copies, not views of whole cores;
  4. the mask is binarised exactly as before.

Run:  python3 -I tests/test_preload_equivalence.py [path/to/old/package/root]
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np                                              # noqa: E402
from PIL import Image                                           # noqa: E402

TEST_ROOT = os.environ.get("FROZENSE_TEST_DIR",
                           os.path.join(tempfile.gettempdir(), "frozense_tests"))
SCRATCH = os.path.join(TEST_ROOT, "preload_test")

SIZES = {  # core_id -> (h, w, mask kind)
    "slide001_core001": (300, 260, "blob"),
    "slide001_core002": (220, 240, "sparse"),
    "slide002_core001": (256, 256, "empty"),
    "slide003_core004": (48, 70, "blob"),       # smaller than a patch: padded
    "slide005_core002": (310, 290, "grades"),   # raw grade values 0/1/3/4/5
}
PATCH = 64
CFG_KW = dict(patch_size=PATCH, patches_per_core_train=7, patches_per_core_eval=5,
              min_foreground_frac_train=0.05, max_sampling_tries=20)


def build_cores():
    shutil.rmtree(SCRATCH, ignore_errors=True)
    os.makedirs(f"{SCRATCH}/images"); os.makedirs(f"{SCRATCH}/labels")
    rng = np.random.default_rng(3)
    for cid, (h, w, kind) in SIZES.items():
        img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        Image.fromarray(img).save(f"{SCRATCH}/images/{cid}.png")   # lossless
        lbl = np.zeros((h, w), np.uint8)
        if kind == "blob":
            lbl[h // 4: 3 * h // 4, w // 5: w // 2] = 4
        elif kind == "sparse":
            lbl[10:18, 200:208] = 3
        elif kind == "grades":
            lbl[:, : w // 3] = 1          # benign tissue: NOT tumour
            lbl[50:150, 100:200] = 3
            lbl[160:260, 120:260] = 5
        Image.fromarray(lbl).save(f"{SCRATCH}/labels/{cid}_classimg_nonconvex.png")


def new_positions_and_patches():
    from frozense.config import DataConfig, SplitConfig
    from frozense.data import discover_cores, PatchDataset

    cfg = DataConfig(images_dir=f"{SCRATCH}/images", labels_dir=f"{SCRATCH}/labels",
                     image_extensions=(".png",), **CFG_KW)
    cores = discover_cores(cfg, SplitConfig(allow_core_level_fallback=True))
    out = {}
    for train in (True, False):
        lazy = PatchDataset(cores, cfg, None, train=train, seed=42, preload=False)
        pre = PatchDataset(cores, cfg, None, train=train, seed=42, preload=True,
                           preload_threads=3)
        positions = []
        for i in range(len(lazy)):
            a_img, a_lbl = lazy.raw_patch(i)
            b_img, b_lbl = pre.raw_patch(i)
            assert a_img.shape == (PATCH, PATCH, 3) and a_lbl.shape == (PATCH, PATCH)
            assert np.array_equal(a_img, b_img) and np.array_equal(a_lbl, b_lbl), \
                f"lazy and preloaded patch {i} differ (train={train})"
            assert b_img.base is None and b_lbl.base is None, \
                f"preloaded patch {i} is a view, would pin a whole core in memory"
            ci, pi = lazy.index[i]
            img, lbl = lazy._load_core(ci)
            positions.append(lazy._coords(ci, pi, img.shape[0], img.shape[1],
                                          lbl if train else None))
        out["train" if train else "eval"] = positions
        out[("train" if train else "eval") + "_fg"] = [
            float(pre.raw_patch(i)[1].mean()) for i in range(len(pre))]
        out[("train" if train else "eval") + "_masks_binary"] = all(
            set(np.unique(pre.raw_patch(i)[1])) <= {0, 1} for i in range(len(pre)))

    # Binarisation on the raw-grade core: grade 1 (benign) must NOT be
    # tumour; grades 3 and 5 must be.
    ci = [c.core_id for c in cores].index("slide005_core002")
    _, mask = lazy._decode_core(ci)
    w = SIZES["slide005_core002"][1]
    out["benign_excluded"] = int(mask[:, : w // 3].sum()) == 0
    out["tumour_included"] = bool(mask[50:150, 100:200].all() and mask[160:260, 120:260].all())
    return [c.core_id for c in cores], out


OLD_PROBE = r"""
import sys, json
sys.path.insert(0, sys.argv[1])
import numpy as np
from frozense.config import DataConfig, SplitConfig
from frozense.data import discover_cores, PatchDataset
cfg = DataConfig(images_dir=sys.argv[2] + "/images", labels_dir=sys.argv[2] + "/labels",
                 image_extensions=(".png",), **json.loads(sys.argv[3]))
cores = discover_cores(cfg, SplitConfig(allow_core_level_fallback=True))
res = {"order": [c.core_id for c in cores]}
for train in (True, False):
    ds = PatchDataset(cores, cfg, None, train=train, seed=42)
    pos, fg = [], []
    for ci, pi in ds.index:
        img, lbl = ds._load_core(ci)
        y, x = ds._sample_coords(ci, pi, img.shape[0], img.shape[1])
        pos.append([y, x])
        crop = lbl[y:y + cfg.patch_size, x:x + cfg.patch_size]
        fg.append(float(np.pad(crop, ((0, cfg.patch_size - crop.shape[0]),
                                      (0, cfg.patch_size - crop.shape[1]))).mean()))
    res["train" if train else "eval"] = pos
    res[("train" if train else "eval") + "_fg"] = fg
print(json.dumps(res))
"""


def old_positions(old_root):
    r = subprocess.run([sys.executable, "-I", "-c", OLD_PROBE, old_root, SCRATCH,
                        json.dumps(CFG_KW)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("old implementation failed:\n" + r.stderr[-2000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def check(label, ok, detail=""):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        raise AssertionError(label + " " + detail)


def main():
    build_cores()
    order, new = new_positions_and_patches()
    n_tr, n_ev = len(new["train"]), len(new["eval"])
    print(f"{len(order)} synthetic cores, {n_tr} train and {n_ev} eval patches\n")
    check("lazy and preloaded loaders give byte-identical patches", True)
    check("preloaded patches are copies, not views of whole cores", True)
    check("masks are binary 0/1", new["train_masks_binary"] and new["eval_masks_binary"])
    check("benign grade (1) is not counted as tumour", new["benign_excluded"])
    check("grades 3 and 5 are counted as tumour", new["tumour_included"])
    check("the empty core yields empty training patches",
          all(f == 0.0 for f in new["train_fg"][14:21]))
    check("training sampling prefers foreground",
          np.mean(new["train_fg"]) > np.mean(new["eval_fg"]),
          f"train {np.mean(new['train_fg']):.3f} vs eval {np.mean(new['eval_fg']):.3f}")

    if len(sys.argv) > 1:
        old = old_positions(sys.argv[1])
        print()
        check("previous version saw the same cores in the same order",
              old["order"] == order)
        check("training patch positions unchanged from the previous version",
              [list(p) for p in new["train"]] == old["train"])
        check("evaluation patch positions unchanged from the previous version",
              [list(p) for p in new["eval"]] == old["eval"])
        check("foreground fractions unchanged",
              np.allclose(new["train_fg"], old["train_fg"])
              and np.allclose(new["eval_fg"], old["eval_fg"]))
    else:
        print("\n(no previous version given; skipped the cross-version check)")

    shutil.rmtree(SCRATCH, ignore_errors=True)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
