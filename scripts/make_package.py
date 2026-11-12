"""Zip the `frozense` package for upload to Kaggle or Colab.

    python scripts/make_package.py [OUT_ZIP]

The notebooks look for a folder named `frozense` or a zip named
`FrozenSE_SAM_Revision_Package*.zip` among their inputs, so keep that file name.
"""
import os
import sys
import zipfile

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "FrozenSE_SAM_Revision_Package.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("frozense/", "")
    for name in sorted(os.listdir(os.path.join(root, "frozense"))):
        if name.endswith(".py"):
            # forward slashes always: Linux extracts backslash paths flat
            z.write(os.path.join(root, "frozense", name), "frozense/" + name)
print(f"wrote {out}")
