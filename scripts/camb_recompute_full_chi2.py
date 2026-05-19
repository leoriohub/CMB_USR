"""
Recompute full-ℓ D_ell, chi² (unbinned + binned) for existing camb_scan results.

Reads P_S(k) from existing JSONL logs, runs CAMB at ell_max=2500 for each
config, computes full-ℓ chi² vs Planck TT.

Usage:
  python scripts/camb_recompute_full_chi2.py
  python scripts/camb_recompute_full_chi2.py --resume <existing_output>
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

from scripts.camb_wrapper import compute_cl_full_camb, compute_cl_camb_powerlaw
from scripts.chi2_analysis import chi2_unbinned, chi2_binned
from scripts.planck_data import C_ell_to_d_ell
from scripts.plotting import get_path

ELL_MAX = 2500


def load_jsonls(paths):
    records = []
    for label, path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("_type") == "header":
                    continue
                if rec.get("status") == "ok":
                    rec["_source"] = label
                    records.append(rec)
    return records


def load_completed(output_path):
    completed = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    k = (round(rec["phi0"], 4), round(rec["y0"], 4), round(rec["N_star"], 2))
                    completed.add(k)
            except (json.JSONDecodeError, KeyError):
                pass
    return completed


def write_log(f, entry):
    f.write(json.dumps(entry) + "\n")
    f.flush()


def find_kphys_header(paths):
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                try:
                    h = json.loads(line)
                    if h.get("_type") == "header" and "k_phys" in h:
                        return np.array(h["k_phys"])
                except Exception:
                    continue
    return None


def build_ps_data(rec, k_phys, P_key="P_S"):
    P = np.array(rec[P_key])
    n_extra = len(P) - len(k_phys)
    if n_extra > 0:
        P = P[:len(k_phys)]
    return {"k_phys": k_phys.tolist(), "P_S": P.tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase1", type=str, default=None)
    p.add_argument("--phase2", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    # Auto-detect
    if not args.phase1 and not args.phase2:
        log_dir = get_path("logs", "")
        candidates = [f for f in os.listdir(log_dir) if f.startswith("camb_phase") and f.endswith(".jsonl")]
        p1 = sorted([f for f in candidates if "phase1" in f])
        p2 = sorted([f for f in candidates if "phase2" in f])
        args.phase1 = os.path.join(log_dir, p1[-1]) if p1 else None
        args.phase2 = os.path.join(log_dir, p2[-1]) if p2 else None
        print(f"Phase 1: {args.phase1}")
        print(f"Phase 2: {args.phase2}")

    if not args.phase1 and not args.phase2:
        print("ERROR: No log files")
        sys.exit(1)

    # Output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.resume or get_path("logs", f"camb_full_chi2_{ts}.jsonl")
    completed = set()
    if args.resume:
        completed = load_completed(output_path)
        print(f"Resume: {len(completed)} already done")

    # Load records
    sources = [("phase1", args.phase1), ("phase2", args.phase2)]
    records = load_jsonls(sources)
    print(f"OK records: {len(records)}")

    if completed:
        records = [r for r in records if (round(r.get("phi0", 0), 4), round(r.get("y0", 0), 4), round(r.get("N_star", 0), 2)) not in completed]
        print(f"Remaining: {len(records)}")

    # Get k_phys from headers
    k_phys = find_kphys_header([args.phase1, args.phase2])
    if k_phys is None:
        print("ERROR: k_phys not found in headers")
        sys.exit(1)
    print(f"k_phys: {k_phys[0]:.2e} - {k_phys[-1]:.2e} ({len(k_phys)})")

    # LCDM baseline (computed once)
    print("Computing LCDM baseline...")
    ells_lcdm, C_lcdm, _, _ = compute_cl_camb_powerlaw(ell_max=ELL_MAX)
    D_lcdm = C_ell_to_d_ell(ells_lcdm, C_lcdm)

    # Main loop
    log_file = open(output_path, "a")
    t0 = time.time()
    done = 0
    errors = 0
    total = len(records)

    print(f"\nProcessing {total} configs (ell_max={ELL_MAX})...")

    for rec in records:
        done += 1
        phi0 = rec.get("phi0", "?")
        y0 = rec.get("y0", "?")
        nstar = rec.get("N_star", "?")

        ps_data = build_ps_data(rec, k_phys)

        try:
            ells, C_TT, C_TE, C_EE = compute_cl_full_camb(ps_data, ell_max=ELL_MAX)
        except Exception as e:
            errors += 1
            entry = dict(
                _type="data",
                phi0=rec.get("phi0"), y0=rec.get("y0"), N_star=rec.get("N_star"),
                N_total=rec.get("N_total"), k_dip=rec.get("k_dip"),
                suppression_pct=rec.get("suppression_pct"),
                status="camb_error", error=str(e),
                _source=rec.get("_source"),
            )
            write_log(log_file, entry)
            eta = (time.time() - t0) / done * (total - done)
            print(f"\r  [{done}/{total}] phi0={phi0} y0={y0:+.3f} N*={nstar} ERROR  ETA {eta/60:.0f}m", end="", flush=True)
            continue

        D_ell = C_ell_to_d_ell(ells, C_TT)

        # Chi² computations
        try:
            chi2_unb, chi2_l_unb, n_unb = chi2_unbinned(D_ell, ells, D_lcdm, ells_lcdm)
        except Exception as e:
            chi2_unb, chi2_l_unb, n_unb = None, None, 0

        try:
            chi2_bin, chi2_l_bin, n_bin = chi2_binned(D_ell, ells, D_lcdm, ells_lcdm)
        except Exception as e:
            chi2_bin, chi2_l_bin, n_bin = None, None, 0

        entry = dict(
            _type="data",
            phi0=rec.get("phi0"), y0=rec.get("y0"), N_star=rec.get("N_star"),
            N_total=rec.get("N_total"), k_dip=rec.get("k_dip"),
            suppression_pct=rec.get("suppression_pct"),
            _source=rec.get("_source"),
            status="ok",
            chi2_unbinned=round(chi2_unb, 2) if chi2_unb is not None else None,
            chi2_lcdm_unbinned=round(chi2_l_unb, 2) if chi2_l_unb is not None else None,
            n_unbinned=n_unb,
            chi2_binned=round(chi2_bin, 2) if chi2_bin is not None else None,
            chi2_lcdm_binned=round(chi2_l_bin, 2) if chi2_l_bin is not None else None,
            n_binned=n_bin,
            D_ell_full=D_ell.tolist(),
            C_ell_TT_full=C_TT.tolist(),
            C_ell_TE_full=C_TE.tolist(),
            C_ell_EE_full=C_EE.tolist(),
        )
        write_log(log_file, entry)

        eta = (time.time() - t0) / done * (total - done)
        chi2_str = f"unb={entry.get('chi2_unbinned','?'):.1f} bin={entry.get('chi2_binned','?'):.1f}"
        print(f"\r  [{done:4d}/{total}] phi0={phi0:5.2f} y0={y0:+.3f} N*={nstar:5.1f}  {chi2_str}  ETA {eta/60:.0f}m", end="", flush=True)

    log_file.close()
    elapsed = time.time() - t0
    rate = total / elapsed if elapsed > 0 else 0
    print(f"\n\nDone: {done} configs in {elapsed:.0f}s ({elapsed/60:.1f}m, {rate:.1f}/s)")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
