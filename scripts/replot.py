"""
Regenerate plots from existing JSON files.
No MS solver, no CAMB — just matplotlib.
"""
import sys, json, numpy as np

sys.path.insert(0, ".")
from scripts.plotting import plot_ps, plot_camb_fullsky

CONFIGS = [
    ("6.55", "-0.780", "50"),
    ("6.50", "-0.730", "50"),
    ("6.60", "-0.830", "50"),
    ("6.60", "-0.805", "51"),
    ("6.55", "-0.755", "51"),
    ("5.75", "-0.170", "55"),
    ("5.70", "-0.170", "52"),
    ("6.50", "-0.500", "63"),
    ("5.65", "-0.145", "53"),
    ("6.40", "-0.475", "59"),
]

for phi0, y0, nstar in CONFIGS:
    # Load PS
    ps_path = f"outputs/simulations/pspectra/ps_phi{phi0}_y0{y0}_nstar{nstar}.0.json"
    with open(ps_path) as f:
        ps_rec = json.load(f)
    k_phys = np.array(ps_rec["spectrum"]["k_phys"])
    P_S = np.array(ps_rec["spectrum"]["P_S"])

    # Load CAMB
    camb_path = f"outputs/simulations/c_ell/camb_phi{phi0}_y0{y0}_nstar{nstar}.0.json"
    with open(camb_path) as f:
        rec = json.load(f)

    camb_data = {
        "ells": np.array(rec["c_ell"]["model"]["ells"]),
        "D_camb": np.array(rec["c_ell"]["model"]["D_ell"]),
        "D_pl": np.array(rec["c_ell"]["lcdm"]["D_ell"]),
        "ells_lcdm": np.array(rec["c_ell"]["lcdm"]["ells"]),
        "planck_ells": np.array(rec["c_ell"]["planck_data"]["ells"]),
        "D_planck": np.array(rec["c_ell"]["planck_data"]["D_ell"]),
        "D_err_lower": np.array(rec["c_ell"]["planck_data"]["D_ell_err_lower"]),
        "D_err_upper": np.array(rec["c_ell"]["planck_data"]["D_ell_err_upper"]),
    }

    plot_ps(k_phys, P_S, filename=f"ps_phi{phi0}_y0{y0}_nstar{nstar}", category="powerloss")
    plot_camb_fullsky(camb_data, filename=f"camb_fullsky_phi{phi0}_y0{y0}_nstar{nstar}", category="powerloss")
    print(f"  Plotted: phi0={phi0} y0={y0} N*={nstar}")
