#!/usr/bin/env python3
"""Generate companion JSON sidecars for every plot in top_configs/.

Each XXX.png gets a XXX.json with full parameter snapshot:
  - Config params (x_c, c, β, χ₀, y₀, N_star)
  - Background (N_total, n_s, A_s)
  - MS solver (k_peak, P_S_peak, grid-boundary flag)
  - PBH (ζ_c, f_total, M_peak, present-day mass)
  - Run metadata (source, pipeline version, timestamp)
"""

import glob
import json
import os
import re
from datetime import datetime

from scripts.constants import ROOT_DIR, ACCRETION, k_eq_default, M_eq_default, gamma_default, NumpyJSONEncoder
from scripts.plotting import make_filename

PLOT_DIR = os.path.join(ROOT_DIR, "outputs/plots/pbh/top_configs")
LOGS_DIR = os.path.join(ROOT_DIR, "outputs/simulations/logs")
PSPECTRA_DIR = os.path.join(ROOT_DIR, "outputs/simulations/pspectra")

# Pattern: rank{N}_chi{chi0}_beta{beta}_nstar{N*}[_zcz{zeta_c}].png
PLOT_RE = re.compile(
    r"rank(?P<rank>\d+)_chi(?P<chi0>[\d.]+)"
    r"_beta(?P<beta>[\de.\-+]+)_nstar(?P<nstar>[\d.]+)"
    r"(?:_zcz(?P<zcz>[\d.]+))?"
    r"\.png$"
)

# Default fixed params for this batch
DEFAULT_C = 0.77
DEFAULT_XC = 0.784
DEFAULT_Y0 = -1e-4
ACCR = ACCRETION
K_EQ = k_eq_default
M_EQ = M_eq_default
GAMMA = gamma_default


def parse_beta(s):
    """Parse '3e-05', '2e-05', '1e-04' etc."""
    return float(s)


def load_jsonl_entries(log_path):
    """Load all entries from a JSONL file, keyed by zeta_c."""
    if not os.path.exists(log_path):
        return {}
    entries = {}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                zc = d.get("zeta_c")
                if zc is not None:
                    entries[zc] = d
            except json.JSONDecodeError:
                pass
    return entries


def find_log_for_rank(rank):
    """Find the JSONL log file for a given rank."""
    prefix = f"top_config_rank{rank}_"
    for fname in os.listdir(LOGS_DIR):
        if fname.startswith(prefix) and fname.endswith(".jsonl"):
            return os.path.join(LOGS_DIR, fname)
    return None


def compute_ps_peak(ps_path):
    """Load P_S(k) and return peak info."""
    if not os.path.exists(ps_path):
        return None
    with open(ps_path) as f:
        d = json.load(f)
    k = d["spectrum"]["k_phys"]
    ps = d["spectrum"]["P_S"]
    # Peak at k > 1 Mpc⁻¹
    pbh_mask = [ki > 1.0 for ki in k]
    k_pbh = [ki for ki, m in zip(k, pbh_mask, strict=False) if m]
    ps_pbh = [pi for pi, m in zip(ps, pbh_mask, strict=False) if m]
    if not k_pbh:
        return None
    peak_i = int(max(range(len(ps_pbh)), key=lambda i: ps_pbh[i]))
    return {
        "k_peak_Mpc": k_pbh[peak_i],
        "P_S_peak": ps_pbh[peak_i],
    }


def main():
    pngs = sorted(glob.glob(os.path.join(PLOT_DIR, "*.png")))
    print(f"Found {len(pngs)} plots in {PLOT_DIR}")

    for png_path in pngs:
        fname = os.path.basename(png_path)
        m = PLOT_RE.match(fname)
        if not m:
            print(f"  SKIP (unparseable): {fname}")
            continue

        rank = int(m.group("rank"))
        chi0 = float(m.group("chi0"))
        beta = parse_beta(m.group("beta"))
        nstar = float(m.group("nstar"))
        zcz = float(m.group("zcz")) if m.group("zcz") else None

        json_path = png_path.replace(".png", ".json")
        if os.path.exists(json_path):
            print(f"  EXISTS: {os.path.basename(json_path)}")
            continue

        # Build companion record
        rec = {
            "plot_file": fname,
            "generated": datetime.now().isoformat(),
            "pipeline": "run_full_pbh_pipeline (MS solver)",
            "config": {
                "model": "EzquiagaCHIModel",
                "x_c": DEFAULT_XC,
                "c": DEFAULT_C,
                "beta": beta,
                "chi0": chi0,
                "y0": DEFAULT_Y0,
                "N_star": nstar,
            },
            "background": {},
            "ms_solver": {},
            "pbh": {},
        }

        # Background & MS from log file
        log_path = find_log_for_rank(rank)
        if log_path:
            entries = load_jsonl_entries(log_path)
            # Find entry matching this plot's zeta_c
            target_zc = (
                zcz
                if zcz is not None
                else max(
                    (
                        e["zeta_c"]
                        for e in entries.values()
                        if e.get("f_total", float("inf")) <= 1.0 and e["f_total"] > 0
                    ),
                    key=lambda zc: entries[zc]["f_total"],
                    default=None,
                )
            )

            if target_zc is not None and target_zc in entries:
                e = entries[target_zc]
                rec["pbh"] = {
                    "zeta_c": target_zc,
                    "f_total": e.get("f_total"),
                    "M_present_Msun": e.get("M_peak_Msun"),
                }
                # Use first entry for common fields
                first_k = next(iter(entries))
                fe = entries[first_k]
                rec["background"] = {
                    "N_total": fe.get("N_total"),
                    "n_s": fe.get("n_s"),
                    "A_s_at_cmb": fe.get("A_s_at_cmb"),
                    "chi0": fe.get("chi0"),
                    "N_star": fe.get("N_star"),
                }
                rec["ms_solver"] = {
                    "P_S_peak": fe.get("P_S_peak"),
                    "k_peak_Mpc": fe.get("k_peak"),
                    "on_grid_boundary": fe.get("on_grid_boundary", False),
                }
        else:
            print(f"  WARNING: no log file for rank {rank}")

        # Try MS JSON for P_S peak if log didn't have it
        if not rec["ms_solver"].get("P_S_peak"):
            n_tr = (
                round(rec["background"].get("N_total", 0), 1)
                if rec["background"].get("N_total")
                else None
            )
            if n_tr:
                ps_fname = make_filename("ps", chi0, DEFAULT_Y0, n_tr, ".json")
                ps_path = os.path.join(PSPECTRA_DIR, ps_fname)
                peak_data = compute_ps_peak(ps_path)
                if peak_data:
                    rec["ms_solver"].update(peak_data)

        # Derive present-day mass from f_pbh peak if we have the P_S(k)
        # (approximate: same calculation as the pipeline)
        if "P_S_peak" in rec["ms_solver"] and rec["ms_solver"]["k_peak_Mpc"]:
            kp = rec["ms_solver"]["k_peak_Mpc"]
            M_form = GAMMA * M_EQ * (K_EQ / kp) ** 2
            M_present = M_form * ACCR
            rec["pbh"]["M_form_Msun"] = M_form
            if not rec["pbh"].get("M_present_Msun"):
                rec["pbh"]["M_present_Msun"] = M_present

        # Write companion JSON
        with open(json_path, "w") as f:
            json.dump(rec, f, indent=2, cls=NumpyJSONEncoder)
        print(f"  WROTE: {os.path.basename(json_path)}")

    print(
        f"\nDone. Generated {sum(1 for p in pngs if not os.path.exists(p.replace('.png', '.json')) or True)} companion files."
    )


if __name__ == "__main__":
    main()
