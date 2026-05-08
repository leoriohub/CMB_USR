"""
Verify the conda environment has all dependencies needed for the project.
Imports every module used in the codebase and reports versions.
"""

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

packages = [
    ("numpy", "np", "core numeric"),
    ("scipy", "scipy", "ODE, interpolation, special functions"),
    ("matplotlib", "mpl", "plotting"),
    ("jupyter", None, "notebooks"),
]

print("=" * 52)
print(f"  Python {sys.version}")
print("=" * 52)

all_ok = True
for pkg_name, alias, desc in packages:
    try:
        mod = importlib.import_module(pkg_name)
        ver = getattr(mod, "__version__", "?")
        print(f"  [OK]    {pkg_name:20s} {ver}  ({desc})")
    except ImportError as e:
        print(f"  [FAIL]  {pkg_name:20s} — {e}")
        all_ok = False

print("─" * 52)

# try importing project modules
project_modules = [
    "models.base",
    "models.standard",
    "models.higgs",
    "models.Smooth_USR",
    "models.punctuated",
    "scripts.constants",
    "scripts.planck_data",
    "scripts.pspectrum_pipeline",
    "scripts.sachs_wolfe",
]

print("  Project modules:")
for mod_name in project_modules:
    try:
        importlib.import_module(mod_name)
        print(f"    [OK]  {mod_name}")
    except ImportError as e:
        print(f"    [FAIL] {mod_name} — {e}")
        all_ok = False

print("═" * 52)
if all_ok:
    print("  Environment OK")
else:
    print("  Some imports FAILED — check messages above")
    sys.exit(1)
