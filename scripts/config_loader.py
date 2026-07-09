"""
Unified config loader for CMB anomaly project.

All configs share the canonical nested format (used by pspectrum_pipeline.py).
This loader understands both nested and flat formats and maps to argparse
namespaces, so every script can accept the same JSON file.

Canonical format:
{
  "model": "EzquiagaCHIModel",
  "model_params": { "c": 0.77, "lambda_0": 2.23e-7, ... },
  "inflection": { "x_c": 0.784, "beta": 1e-5 },
  "ics": { "x0": 8.0, "y0": -1e-4 },
  "pipeline": { "N_star": 65, "k_min": 1e-10, "k_max": 1e28, ... },
  "description": "optional free text"
}

Flat (legacy) format is auto-detected and returned as-is for back-compat.
"""

import json
import os


def load_config(path):
    """Load a JSON config file, returning a flat dict for argparse.set_defaults().

    If the file is already flat (no 'model' key), returns it unchanged.
    If it's in canonical nested format, flattens it using the mapping below.
    """
    with open(path) as f:
        raw = json.load(f)

    # Sweep configs have completely different schema — always flat
    if any(k in raw for k in ["x_c_lo", "x_c_hi", "beta_vals", "N_star_vals"]):
        return raw

    # Already flat (legacy, e.g. chi0/beta/N_star at top level)
    if "model" not in raw:
        return raw

    # Canonical nested format → flatten for argparse
    flat = {}

    # Preserve description
    if "description" in raw:
        flat["description"] = raw["description"]

    # model_params → flat
    mp = raw.get("model_params", {})
    for k, v in mp.items():
        flat[k] = v

    # ics → flat
    ics = raw.get("ics", {})
    if "x0" in ics:
        flat["chi0"] = ics["x0"]
    if "y0" in ics:
        flat["y0"] = ics["y0"]
    if "phi0" in ics:
        flat["phi0"] = ics["phi0"]

    # inflection → flat
    inf = raw.get("inflection", {})
    if "x_c" in inf:
        flat["xc"] = inf["x_c"]
    if "beta" in inf:
        flat["beta"] = inf["beta"]

    # pipeline → flat (remap worker key names)
    pipe = raw.get("pipeline", {})
    for k, v in pipe.items():
        if k in ("n_cores", "n_workers", "workers"):
            flat["workers"] = v
        elif k == "output_dir":
            flat["output_dir"] = v
        elif k == "N_star":
            flat["N_star"] = v
        elif k == "no_plot":
            flat["no_plot"] = v
        elif k == "normalize_to_As":
            flat["normalize_to_As"] = v
        elif k == "zeta_c":
            # Could be single float or list
            flat["zeta_c"] = v if isinstance(v, list) else [v]
        else:
            flat[k] = v

    # zeta_c values may be specified at top level (common in best-config files)
    if "zeta_c" in raw and "zeta_c" not in flat:
        flat["zeta_c"] = raw["zeta_c"] if isinstance(raw["zeta_c"], list) else [raw["zeta_c"]]
    if "output_dir" in raw and "output_dir" not in flat:
        flat["output_dir"] = raw["output_dir"]

    return flat
