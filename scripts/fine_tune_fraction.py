"""Fine-tune fraction scan — Prof. Cárdenas Correction 3.

Compute the fraction of the physically-reasonable (x0, y0) initial-condition
space that produces >=20% suppression of the quadrupole D2 relative to LCDM
(D2 <= 0.8 * D2_LCDM = 823.0 uK^2).

A grid point "produces the effect" if ANY pivot alignment N_* in [55, 70]
yields that suppression.

Subcommands:
    explore  — no compute; mine existing camb phase/full_chi2 logs for a
               preliminary fraction + heatmap (instant, local).
    scan     — uniform grid over (x0, y0); production bg params; final number.
    report   — read a scan JSONL; print fraction + binomial error +
               cap-sensitivity table; write scan summary JSON + heatmap/bar PNGs.

Model conventions (AGENTS.md): Planck units M_P=1, S=5e-5, x0=phi0, y0=dx/dT.
Production params: T_max=500, bg_steps=1000, ms_steps=5000, grid-150 k-grid.
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Thread caps BEFORE any threading-capable import (numpy/matplotlib/CAMB/fortran):
# default to 1 thread per worker so --workers scales out via the process pool, and the
# parent must NOT initialize a full-width OpenMP pool before forking. setdefault lets
# a power user export OMP_NUM_THREADS=N to scale up per-solve instead
# (optimize_pbh_botorch.py precedent).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np

from models.higgs import HiggsModel
from pspectrum_pipeline import (
    build_weighted_kgrid,
    find_end_of_inflation,
    run_pspectrum_pipeline,
)
import inf_dyn_background as bg_solver
from scripts.constants import As
from scripts.planck_data import C_ell_to_d_ell
from scripts.camb_wrapper import compute_cl_full_camb, compute_cl_camb_powerlaw
from scripts.plotting import get_path, save_fig, PAPER_RCPARAMS, TOL

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
S = 5e-5
T_MAX = 2000.0          # large enough to reach end of inflation for the relevant band (N_total<=73)
BG_STEPS = 2000
MS_STEPS = 5000
K_PIVOT = 0.002
N_DENSE = 100
N_OUTER = 50

D2_LCDM_THRESHOLD_FRAC = 0.8          # >=20% suppression
N_STAR_SWEEP = list(range(55, 71))    # 55..70 inclusive — user decision
N_TOTAL_MIN = 55.0                    # viable iff N_total > 55 (N_star=55 runnable)
# The USR freeze is at the very start (N_pivot ~ 1-4 from the beginning). The pivot
# scale reaches it only if N_total - N_star ~ 1-4 for some N_star in [55,70], i.e.
# N_total in (55, 73]. Configs with N_total > 73 have the dip on unobservably large
# scales (CMB on the plateau -> no suppression, not relevant). See correction note.
N_TOTAL_RELEVANT_MAX = 73.0
X0_MIN, X0_MAX = 5.2, 6.5            # relevant band is x0 <= ~6.3 (N_total>73 beyond)
Y0_MIN, Y0_MAX = -0.35, -0.005

# Expected D2_LCDM (camb_scan.py:38); startup assert range.
D2_LCDM_EXPECTED = 1028.7
D2_LCDM_LO, D2_LCDM_HI = 1000.0, 1060.0


def _build_k_grid():
    """Shared weighted k-grid (grid-150). D2 stable to 0.0001% vs grid-300."""
    return build_weighted_kgrid(
        k_min=1e-5, k_max=1.0, k_pivot_phys=K_PIVOT,
        dense_min=1e-4, dense_max=1e-2,
        n_dense=N_DENSE, n_outer=N_OUTER,
    )


def _fractional_n_total(model):
    """Compute N_total exactly as the pipeline does (pspectrum_pipeline.py:326-338).

    Returns float N_total, or None if background fails.
    """
    T_span = np.linspace(0.0, T_MAX, BG_STEPS)
    try:
        sol = bg_solver.run_background_simulation(model, T_span)
        derived = bg_solver.get_derived_quantities(sol, model)
    except Exception:
        return None

    end_idx = find_end_of_inflation(derived["epsH"])
    if end_idx == -1:
        end_idx = len(derived["epsH"]) - 1

    epsH = derived["epsH"]
    N = derived["N"]
    N_total = float(N[end_idx])
    if 0 < end_idx < len(epsH) and epsH[end_idx] > 1.0 >= epsH[end_idx - 1]:
        f = (1.0 - epsH[end_idx - 1]) / (epsH[end_idx] - epsH[end_idx - 1])
        if 0 <= f <= 1:
            N_total = float(N[end_idx - 1] + f * (N[end_idx] - N[end_idx - 1]))
    return N_total


def viability(model):
    """Return (N_total, t_over_v); point is viable iff N_total>55 and t_over_v<=1.

    Assumes y0 < 0 (caller enforces). t_over_v = 0.5*(y0*S)^2 / (v0*f(x0)).
    N_total computed exactly as the pipeline (identical to metadata["N_total"]).
    Threshold N_total > 55 admits any config that can run at least N_star=55
    (pipeline requires N_star < N_total); user decision (plan smoke test).
    """
    assert model.y0 < 0, "viability requires y0 < 0"
    t_over_v = 0.5 * (model.y0 * S) ** 2 / (model.v0 * model.f(model.x0))
    N_total = _fractional_n_total(model)
    return N_total, float(t_over_v)


def evaluate_point(x0, y0, N_star, k_grid, executor=None, n_workers=4):
    """Run one pipeline config + CAMB, return the D2 quadrupole.

    Returns {"status", "N_total", "d2", "d2_lcdm"} on success, or
    {"status": "error", "message"} on failure.
    """
    model = HiggsModel(lam=0.13, xi=15000.0)
    model.S = S
    model.x0 = x0
    model.y0 = y0

    try:
        result = run_pspectrum_pipeline(
            model=model,
            phi0=x0,
            y0=y0,
            N_star=N_star,
            k_phys_grid=k_grid,
            ms_steps=MS_STEPS,
            T_max=T_MAX,
            bg_steps=BG_STEPS,
            normalize_to_As=True,
            As=As,
            k_pivot_phys=K_PIVOT,
            save_outputs=False,
            n_workers=n_workers,
            executor=executor,
            backend="fortran",
        )
    except Exception as e:
        return {"status": "error", "message": str(e)}

    if result["status"] != "success":
        return {"status": "error", "message": result.get("message", "")}

    try:
        ells, C_TT, _, _ = compute_cl_full_camb(result, ell_max=30)
        D_ell = C_ell_to_d_ell(ells, C_TT)
        d2 = float(D_ell[0])
    except Exception as e:
        return {"status": "error", "message": f"camb: {e}"}

    return {
        "status": "success",
        "N_total": float(result["metadata"]["N_total"]),
        "d2": d2,
        "d2_lcdm": D2_LCDM_EXPECTED,
    }


def _scan_point(x0, y0, k_grid, n_total, threshold, d2_lcdm, t_over_v, completed):
    """Run the N* sweep for one (x0, y0) inside a ProcessPoolExecutor worker.

    Mirrors the serial per-point loop in scan(): skips already-completed
    (resume) triples and early-exits at the first suppressed N*. Returns
    (records, suppressed).

    Must be module-level (picklable) for ProcessPoolExecutor.
    """
    records = []
    suppressed = False
    for ns in N_STAR_SWEEP:
        if ns >= n_total:
            break  # pipeline errors when N_star >= N_total
        key = (x0, y0, round(float(ns), 1))
        if key in completed:
            continue
        res = evaluate_point(x0, y0, ns, k_grid)
        rec = {
            "x0": x0, "y0": y0, "N_star": float(ns),
            "N_total": res.get("N_total"),
            "t_over_v": t_over_v,
            "status": res["status"],
            "d2": res.get("d2"),
            "d2_lcdm": res.get("d2_lcdm", d2_lcdm),
            "suppressed": bool(res.get("status") == "success"
                               and res.get("d2") is not None
                               and res["d2"] <= threshold),
        }
        if res["status"] == "error":
            print(f"  WARN ({x0},{y0},N*={ns}): {res.get('message')}",
                  file=sys.stderr)
        else:
            rec["suppressed"] = rec["d2"] <= threshold
            if rec["suppressed"]:
                suppressed = True
        records.append(rec)
        if suppressed:
            break  # early exit at first suppressed N*
    return records, suppressed


# ---------------------------------------------------------------------------
# explore — no compute, mine existing logs
# ---------------------------------------------------------------------------
def _load_existing_records():
    """Load all data records from camb_phase*.jsonl + camb_full_chi2_*.jsonl.

    Skips headers and malformed lines. Returns list of dicts.
    """
    logs_dir = get_path("logs", "")
    records = []
    skipped = 0
    for fname in sorted(os.listdir(logs_dir)):
        if not (fname.startswith("camb_phase") or fname.startswith("camb_full_chi2")):
            continue
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(logs_dir, fname)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if rec.get("_type") != "data":
                    continue
                if "phi0" not in rec or "y0" not in rec:
                    skipped += 1
                    continue
                records.append(rec)
    return records, skipped


def _d2_of(rec):
    if "d2" in rec:
        return rec["d2"]
    if "D_ell_full" in rec:
        return rec["D_ell_full"][0]
    return None


def _d2_lcdm_of(rec):
    return rec.get("d2_lcdm", D2_LCDM_EXPECTED)


def explore(args):
    """Mine existing logs; print preliminary fraction + heatmap."""
    records, skipped = _load_existing_records()
    if not records:
        print("ERROR: no data records found in camb_phase*/camb_full_chi2* logs.",
              file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(records)} records from camb_phase*/camb_full_chi2* "
          f"logs ({skipped} malformed lines skipped).", flush=True)

    # Build one model to compute t_over_v for each (x0,y0) (closed-form, no bg).
    model = HiggsModel(lam=0.13, xi=15000.0)
    model.S = S

    # Aggregate per (x0, y0): collect records, classify viability + suppression.
    points = {}   # (x0, y0) -> {"y0", "x0", "viable", "t_over_v", "N_total", "suppressed", "nstar_vals"}
    nstar_histo = {}
    for rec in records:
        x0 = round(float(rec["phi0"]), 4)
        y0 = round(float(rec["y0"]), 4)
        nstar = rec.get("N_star")
        status = rec.get("status")
        d2 = _d2_of(rec)
        d2_lcdm = _d2_lcdm_of(rec)
        n_total = rec.get("N_total")

        if nstar is not None:
            nstar_histo[round(float(nstar), 1)] = nstar_histo.get(
                round(float(nstar), 1), 0) + 1

        key = (x0, y0)
        if key not in points:
            model.x0 = x0
            model.y0 = y0
            tv = 0.5 * (y0 * S) ** 2 / (model.v0 * model.f(x0))
            points[key] = {
                "x0": x0, "y0": y0, "t_over_v": float(tv),
                "viable": bool(n_total is not None and n_total > N_TOTAL_MIN
                               and tv <= 1.0),
                "N_total": n_total, "suppressed": False, "nstar_vals": [],
                "records": [], "best_ratio": None,
            }
        p = points[key]
        p["records"].append(rec)
        if n_total is not None:
            p["N_total"] = n_total
        if nstar is not None:
            p["nstar_vals"].append(round(float(nstar), 1))
        # Track best (min) D2 ratio across this point's records.
        if (status == "ok" and d2 is not None and d2_lcdm
                and d2_lcdm > 0):
            r = d2 / d2_lcdm
            p["best_ratio"] = (r if p["best_ratio"] is None
                                else min(p["best_ratio"], r))
        # Any record with N_star in [55,70] suppressed qualifies the point.
        if (status == "ok" and d2 is not None and nstar is not None
                and 55 <= float(nstar) <= 70
                and d2 <= 0.8 * d2_lcdm):
            p["suppressed"] = True

    viable = [p for p in points.values() if p["viable"]]
    suppressed = [p for p in viable if p["suppressed"]]

    # Cap at x0 <= 7.5 (existing data max) for the headline fraction.
    viable_cap = [p for p in viable if p["x0"] <= 7.5 + 1e-9]
    suppressed_cap = [p for p in suppressed if p["x0"] <= 7.5 + 1e-9]

    n_viable = len(viable_cap)
    n_supp = len(suppressed_cap)
    frac = n_supp / n_viable if n_viable else 0.0
    sigma = math.sqrt(frac * (1 - frac) / n_viable) if n_viable else 0.0

    print("\n=== EXPLORE (preliminary, existing logs) ===", flush=True)
    print(f"Distinct (x0,y0) points: {len(points)}")
    print(f"Viable points (x0<=7.5): {n_viable}")
    print(f"Suppressed points (x0<=7.5): {n_supp}")
    print(f"Fraction: {n_supp}/{n_viable} = {100*frac:.1f}% "
          f"(binomial {100*sigma:.1f}%)", flush=True)

    # N_star coverage histogram on viable points.
    hist = {}
    for p in viable:
        for ns in p["nstar_vals"]:
            hist[ns] = hist.get(ns, 0) + 1
    print("\nN_star coverage (viable points, all records):")
    for ns in sorted(hist):
        print(f"  N*={ns:5.1f}: {hist[ns]}")

    # Effect bounding box.
    if suppressed:
        xs = [p["x0"] for p in suppressed]
        ys = [p["y0"] for p in suppressed]
        print(f"\nEffect bounding box: x0 in [{min(xs):.3f}, {max(xs):.3f}], "
              f"y0 in [{min(ys):.3f}, {max(ys):.3f}]")
    else:
        print("\nEffect bounding box: none")

    # Gap report: viable-region cells with zero coverage.
    gaps = _gap_report(points, viable)
    print("\nGap report (viable-region cells with zero coverage):")
    if gaps:
        for x0, y0 in gaps:
            print(f"  cell x0={x0:.1f} y0={y0:.2f}: no coverage")
        print(f"  Total gap cells: {len(gaps)}")
    else:
        print("  none")

    # Cross-checks from the plan.
    ref = (5.75, -0.170)
    if ref in points:
        rp = points[ref]
        print(f"\n(5.75, -0.170): viable={rp['viable']} "
              f"suppressed={rp['suppressed']} N_total={rp['N_total']}")
    ip = (6.40, -0.475)
    if ip in points:
        rp = points[ip]
        print(f"(6.40, -0.475): viable={rp['viable']} "
              f"t_over_v={rp['t_over_v']:.3f}")

    # Write summary JSON + plots.
    summary = {
        "metadata": {
            "source": "existing logs (mapping-grade, model-default bg params)",
            "nstar_sweep": [55, 70],
            "x0_cap": 7.5,
            "d2_lcdm": D2_LCDM_EXPECTED,
            "threshold": 0.8 * D2_LCDM_EXPECTED,
            "date": datetime.now().isoformat(),
        },
        "summary": {
            "n_points": len(points),
            "n_viable": n_viable,
            "n_suppressed": n_supp,
            "fraction": frac,
            "fraction_err": sigma,
        },
        "points": [{"x0": p["x0"], "y0": p["y0"], "viable": p["viable"],
                    "suppressed": p["suppressed"]} for p in points.values()],
    }
    out = get_path("scans", "fine_tune_explore_existing.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}", flush=True)

    _plot_heatmap(points, filename="fine_tune_fraction_heatmap_preliminary",
                  category="paper")
    _plot_bar(points, viable, suppressed, args,
              filename="fine_tune_fraction_bar_preliminary",
              category="diagnostics")


def _gap_report(points, viable):
    """Count viable-region cells (0.1 x 0.05 bins) with zero coverage.

    Bins over x0 in [5.2, 8.0], y0 in [-0.34, 0).
    """
    gaps = []
    x_edges = np.arange(X0_MIN, X0_MAX + 1e-9, 0.1)
    y_edges = np.arange(-0.34, 0, 0.05)
    for i in range(len(x_edges) - 1):
        for j in range(len(y_edges) - 1):
            xlo, xhi = x_edges[i], x_edges[i + 1]
            ylo, yhi = y_edges[j], y_edges[j + 1]
            covered = any(
                xlo <= p["x0"] < xhi and ylo <= p["y0"] < yhi
                for p in viable
            )
            if not covered:
                gaps.append((round(xlo, 1), round(ylo, 2)))
    return gaps


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------
def _load_resume_keys(resume_path):
    """Load completed (x0, y0, N_star) triples from a resume JSONL."""
    keys = set()
    if not resume_path or not os.path.exists(resume_path):
        return keys
    with open(resume_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "x0" in rec and "y0" in rec:
                ns = rec.get("N_star")
                keys.add((round(rec["x0"], 4), round(rec["y0"], 4),
                          round(ns, 1) if ns is not None else None))
    return keys


def _write_record(log_file, rec):
    log_file.write(json.dumps(rec) + "\n")
    log_file.flush()


def scan(args):
    """Uniform grid scan; the final-number data source."""
    # Startup: verify LCDM baseline.
    ells, C_TT, _, _ = compute_cl_camb_powerlaw(ell_max=30)
    d2_lcdm = float(C_ell_to_d_ell(ells, C_TT)[0])
    print(f"D2_LCDM = {d2_lcdm:.2f} uK^2", flush=True)
    if not (D2_LCDM_LO <= d2_lcdm <= D2_LCDM_HI):
        print(f"ERROR: D2_LCDM {d2_lcdm:.2f} outside expected range "
              f"[{D2_LCDM_LO}, {D2_LCDM_HI}]. Aborting.", file=sys.stderr)
        sys.exit(1)

    x0_min, x0_max = args.x0_min, args.x0_max
    y0_min, y0_max = args.y0_min, args.y0_max
    x0_vals = np.linspace(x0_min, x0_max, args.n_phi0)
    y0_vals = np.linspace(y0_min, y0_max, args.n_y0)
    total = len(x0_vals) * len(y0_vals)
    threshold = D2_LCDM_THRESHOLD_FRAC * d2_lcdm

    log_path = get_path("logs",
                        f"fine_tune_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    if args.resume:
        log_path = args.resume
        completed = _load_resume_keys(args.resume)
        print(f"Resuming; {len(completed)} completed triples already present.",
              flush=True)
    else:
        completed = set()

    print(f"\n{'='*60}", flush=True)
    print(f"  FINE-TUNE SCAN", flush=True)
    print(f"  x0: [{x0_min}, {x0_max}] x {args.n_phi0} = {x0_vals}", flush=True)
    print(f"  y0: [{y0_min}, {y0_max}] x {args.n_y0} = {y0_vals}", flush=True)
    print(f"  Total points: {total}, N* sweep {N_STAR_SWEEP}", flush=True)
    print(f"  Threshold: D2 <= {threshold:.1f} uK^2 (0.8 * {d2_lcdm:.1f})", flush=True)
    print(f"  Log: {log_path}", flush=True)
    print(f"{'='*60}", flush=True)

    k_grid = _build_k_grid()

    with open(log_path, "a") as log_file, \
         ProcessPoolExecutor(args.workers) as executor:
        if not args.resume:
            _write_record(log_file, {
                "_type": "header",
                "x0_min": float(x0_min), "x0_max": float(x0_max),
                "y0_min": float(y0_min), "y0_max": float(y0_max),
                "n_phi0": args.n_phi0, "n_y0": args.n_y0,
                "nstar_sweep": N_STAR_SWEEP,
                "d2_lcdm": d2_lcdm,
                "threshold": threshold,
                "T_max": T_MAX, "bg_steps": BG_STEPS, "ms_steps": MS_STEPS,
                "date": datetime.now().isoformat(),
            })
        done = 0
        n_viable = 0
        n_supp = 0
        t0 = time.time()
        futures = {}
        for x0 in x0_vals:
            for y0 in y0_vals:
                x0 = round(float(x0), 4)
                y0 = round(float(y0), 4)
                base_key = (x0, y0)

                # Pre-filter viability (background ~ms).
                model = HiggsModel(lam=0.13, xi=15000.0)
                model.S = S
                model.x0 = x0
                model.y0 = y0
                n_total, t_over_v = viability(model)
                viable = (n_total is not None and n_total > N_TOTAL_MIN
                          and t_over_v <= 1.0)

                if not viable:
                    key = (x0, y0, None)
                    if key not in completed:
                        _write_record(log_file, {
                            "x0": x0, "y0": y0, "N_star": None,
                            "status": "not_viable",
                            "N_total": n_total, "t_over_v": t_over_v,
                        })
                    done += 1
                    continue

                # Not relevant: USR freeze too far from the pivot scale (N_total > 73)
                # -> CMB sits on the plateau, suppression impossible. Excluded from
                # the fraction (neither numerator nor denominator).
                if n_total is not None and n_total > N_TOTAL_RELEVANT_MAX:
                    key = (x0, y0, None)
                    if key not in completed:
                        _write_record(log_file, {
                            "x0": x0, "y0": y0, "N_star": None,
                            "status": "not_relevant",
                            "N_total": n_total, "t_over_v": t_over_v,
                        })
                    done += 1
                    continue

                n_viable += 1
                # One task per (x0, y0): run the N* sweep (with early-exit) in a
                # pool worker so --workers actually parallelizes the grid points.
                futures[executor.submit(
                    _scan_point, x0, y0, k_grid, n_total,
                    threshold, d2_lcdm, t_over_v, completed)] = (x0, y0)

        # Consume futures as they complete: write records incrementally
        # (crash-safe JSONL) and tally the per-point suppression count.
        for fut in as_completed(futures):
            x0, y0 = futures[fut]
            records, suppressed = fut.result()
            for rec in records:
                _write_record(log_file, rec)
            if suppressed:
                n_supp += 1
            done += 1
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done else 0
                print(f"  [{done}/{total}] viable={n_viable} "
                      f"supp={n_supp} ETA {eta/60:.1f}m", flush=True)

    print(f"\nScan complete. viable={n_viable} suppressed={n_supp} "
          f"fraction={n_supp/max(1,n_viable):.3f}", flush=True)
    print(f"Log: {log_path}", flush=True)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _scan_points(log_path):
    """Read scan JSONL -> (points dict, header dict or None)."""
    points = {}
    header = None
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("_type") == "header":
                header = rec
                continue
            if "x0" not in rec or "y0" not in rec:
                continue
            x0 = round(float(rec["x0"]), 4)
            y0 = round(float(rec["y0"]), 4)
            key = (x0, y0)
            if key not in points:
                points[key] = {
                    "x0": x0, "y0": y0,
                    "viable": False, "relevant": True,
                    "suppressed": False,
                    "N_total": None, "t_over_v": None,
                    "best_ratio": None,
                    "records": [],
                }
            p = points[key]
            p["records"].append(rec)
            if rec.get("N_total") is not None:
                p["N_total"] = rec["N_total"]
            if rec.get("t_over_v") is not None:
                p["t_over_v"] = rec["t_over_v"]
            if rec.get("status") == "not_relevant":
                p["relevant"] = False
            if rec.get("status") == "not_viable":
                p["viable"] = False
            elif rec.get("status") == "success":
                p["viable"] = True
                d2 = rec.get("d2")
                d2_lcdm = rec.get("d2_lcdm", D2_LCDM_EXPECTED)
                if d2 is not None and d2_lcdm and d2_lcdm > 0:
                    r = d2 / d2_lcdm
                    p["best_ratio"] = (r if p["best_ratio"] is None
                                        else min(p["best_ratio"], r))
            if rec.get("suppressed"):
                p["suppressed"] = True
    return points, header


def report(args):
    """Read a scan JSONL; print fraction + sensitivity; write summary + plots."""
    if not args.log or not os.path.exists(args.log):
        print("ERROR: --log is required and must exist.", file=sys.stderr)
        sys.exit(1)
    points, header = _scan_points(args.log)
    if not points:
        print("ERROR: no points found in log.", file=sys.stderr)
        sys.exit(1)

    # Grid bounds: prefer the log header; fall back to module constants.
    if header:
        x0_min = float(header.get("x0_min", X0_MIN))
        x0_max = float(header.get("x0_max", X0_MAX))
        y0_min = float(header.get("y0_min", Y0_MIN))
        y0_max = float(header.get("y0_max", Y0_MAX))
        n_phi0 = int(header.get("n_phi0", args.n_phi0))
        n_y0 = int(header.get("n_y0", args.n_y0))
        d2_lcdm = float(header.get("d2_lcdm", D2_LCDM_EXPECTED))
        threshold = float(header.get("threshold",
                                     0.8 * D2_LCDM_EXPECTED))
    else:
        x0_min, x0_max = X0_MIN, X0_MAX
        y0_min, y0_max = Y0_MIN, Y0_MAX
        n_phi0, n_y0 = args.n_phi0, args.n_y0
        d2_lcdm = D2_LCDM_EXPECTED
        threshold = 0.8 * D2_LCDM_EXPECTED

    print(f"Grid: x0 [{x0_min}, {x0_max}] x {n_phi0}, "
          f"y0 [{y0_min}, {y0_max}] x {n_y0}", flush=True)

    viable = [p for p in points.values() if p["viable"] and p["relevant"]]
    suppressed = [p for p in viable if p["suppressed"]]
    n_relevant = len(viable)
    n_supp = len(suppressed)
    n_notrelevant = sum(1 for p in points.values()
                        if not p["relevant"])
    frac = n_supp / n_relevant if n_relevant else 0.0
    sigma = math.sqrt(frac * (1 - frac) / n_relevant) if n_relevant else 0.0

    print("\n=== REPORT ===", flush=True)
    print(f"Grid points: {len(points)}", flush=True)
    print(f"Relevant & viable points (N_total in (55,{N_TOTAL_RELEVANT_MAX:.0f}]): "
          f"{n_relevant}", flush=True)
    print(f"  not relevant (N_total > {N_TOTAL_RELEVANT_MAX:.0f}): {n_notrelevant}",
          flush=True)
    print(f"Suppressed points: {n_supp}", flush=True)
    print(f"Fraction: {n_supp}/{n_relevant} = {frac:.3f} "
          f"(binomial error {sigma:.3f})", flush=True)
    print(f"  = {100*frac:.1f}% +/- {100*sigma:.1f}%", flush=True)

    # Cap-sensitivity table (no new compute).
    print("\nCap-sensitivity (fraction for x0 <= cap):", flush=True)
    print(f"  {'cap':>6} {'viable':>8} {'supp':>8} {'fraction':>10}", flush=True)
    sensitivity = {}
    for cap in [6.0, 6.5, 7.0]:
        v = [p for p in viable if p["x0"] <= cap + 1e-9]
        s = [p for p in suppressed if p["x0"] <= cap + 1e-9]
        f = len(s) / len(v) if v else 0.0
        se = math.sqrt(f * (1 - f) / len(v)) if v else 0.0
        sensitivity[cap] = {"n_viable": len(v), "n_suppressed": len(s),
                            "fraction": f, "fraction_err": se}
        print(f"  {cap:6.1f} {len(v):8d} {len(s):8d} {f:10.3f}", flush=True)

    summary = {
        "metadata": {
            "grid": {"x0": [x0_min, x0_max], "y0": [y0_min, y0_max]},
            "n_phi0": n_phi0, "n_y0": n_y0,
            "nstar_sweep": [55, 70],
            "n_total_relevant_window": [N_TOTAL_MIN, N_TOTAL_RELEVANT_MAX],
            "d2_lcdm": d2_lcdm,
            "threshold": threshold,
            "bg_params": {"T_max": T_MAX, "bg_steps": BG_STEPS},
            "date": header.get("date", datetime.now().isoformat())
                     if header else datetime.now().isoformat(),
            "log": os.path.basename(args.log),
        },
        "summary": {
            "n_points": len(points),
            "n_relevant_viable": n_relevant,
            "n_suppressed": n_supp,
            "n_not_relevant": n_notrelevant,
            "fraction": frac,
            "fraction_err": sigma,
        },
        "sensitivity": {
            str(cap): val for cap, val in sensitivity.items()
        },
        "points": [{"x0": p["x0"], "y0": p["y0"], "viable": p["viable"],
                    "suppressed": p["suppressed"]} for p in points.values()],
    }
    out = get_path("scans", "fine_tune_fraction_scan.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}", flush=True)

    _plot_heatmap(points, filename="fine_tune_fraction_heatmap",
                  category="paper")
    _plot_bar(points, viable, suppressed, args,
              filename="fine_tune_fraction_bar", category="diagnostics")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_heatmap(points, filename="fine_tune_fraction_heatmap",
                  category="paper"):
    """Phase-space map of best-N* ratio d2/d2_lcdm over (x0, |y0|)."""
    from scipy.interpolate import griddata
    plt.rcParams.update(PAPER_RCPARAMS)

    # Dense grid for the image + overlay contours.
    n_phi = 200
    n_y = 200
    x_centers = np.linspace(X0_MIN, X0_MAX, n_phi)
    y_abs_centers = np.linspace(0.0, 0.35, n_y)  # |y0| axis
    X_c, Y_c = np.meshgrid(x_centers, y_abs_centers)

    # Gather (x0, |y0|) samples with their best ratio and N_total.
    # Ratio: viable points only. N_total: ALL points (non-viable ones carry it
    # too and are needed for the N_total=55 contour to actually cross 55).
    xs, ys, rs, ns = [], [], [], []
    for p in points.values():
        # R: relevant & viable only (the measured data). N_total: ALL points,
        # so the N_total=55 and N_total=73 contours (band boundaries) render.
        if p.get("N_total") is None:
            continue
        xs.append(p["x0"])
        ys.append(abs(p["y0"]))
        rs.append(p.get("best_ratio") if (p.get("viable") and p.get("relevant")) else np.nan)
        ns.append(p.get("N_total"))
    xs, ys, rs, ns = map(np.asarray, (xs, ys, rs, ns))

    R = np.full_like(X_c, np.nan)
    N_field = np.full_like(X_c, np.nan)
    if len(xs) > 0:
        mask_r = np.isfinite(rs)
        if mask_r.sum() >= 4:
            R = griddata((xs[mask_r], ys[mask_r]), rs[mask_r],
                         (X_c, Y_c), method="linear")
        mask_n = np.isfinite(ns)
        if mask_n.sum() >= 4:
            N_field = griddata((xs[mask_n], ys[mask_n]), ns[mask_n],
                               (X_c, Y_c), method="linear")

    x_edges = np.linspace(X0_MIN, X0_MAX, n_phi + 1)
    y_edges = np.linspace(0.0, 0.35, n_y + 1)
    X_e, Y_e = np.meshgrid(x_edges, y_edges)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.set_facecolor("0.85")  # masked gray for non-viable/no-data cells
    mesh = ax.pcolormesh(X_e, Y_e, R, shading="flat", cmap="magma",
                         vmin=0.5, vmax=1.1)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$D_2 / D_2^{\mathrm{LCDM}}$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # T<=V boundary curve: |y0| = sqrt(2*v0*f(x0))/S
    model = HiggsModel(lam=0.13, xi=15000.0)
    model.S = S
    x_curve = np.linspace(X0_MIN, X0_MAX, 300)
    yv_curve = np.sqrt(2 * model.v0 * model.f(x_curve)) / S
    ax.plot(x_curve, yv_curve, "--", color=TOL["grey"], lw=1.2,
            label=r"$T \leq V$")
    # N_total=55 and N_total=73 contours (relevant band boundaries), dashed.
    if np.any(np.isfinite(N_field)):
        ax.contour(X_c, Y_c, N_field, levels=[N_TOTAL_MIN], colors=[TOL["blue"]],
                   linestyles="--", linewidths=1.2)
        ax.contour(X_c, Y_c, N_field, levels=[N_TOTAL_RELEVANT_MAX],
                   colors=[TOL["blue"]], linestyles="--", linewidths=1.2)

    # Reference config marker.
    ax.plot(5.75, 0.170, marker="*", color=TOL["green"], markersize=9)

    ax.set_xlim(X0_MIN, X0_MAX)
    ax.set_ylim(0.0, 0.35)
    ax.set_xlabel(r"$x_0$", fontsize=9)
    ax.set_ylabel(r"$|y_0|$", fontsize=9)
    # Explicit proxy handles (ax.legend() does not auto-collect contour labels).
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=TOL["grey"], ls="--", lw=1.2,
               label=r"$T \leq V$"),
        Line2D([0], [0], color=TOL["blue"], ls="--", lw=1.2,
               label=r"$N_{\mathrm{total}} \in (55, 73]$"),
        Line2D([0], [0], marker="*", color=TOL["green"], markersize=9,
               ls="none", label="ref (5.75, -0.170)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6,
              framealpha=0.8)
    fig.tight_layout()
    save_fig(fig, filename, category)


def _plot_bar(points, viable, suppressed, args,
              filename="fine_tune_fraction_bar", category="diagnostics"):
    """Fraction summary bar: relevant-viable split + suppressed split."""
    plt.rcParams.update(PAPER_RCPARAMS)

    n_total = len(points)
    n_viable = len(viable)
    n_relevant = sum(1 for p in points.values()
                     if p.get("viable") and p.get("relevant"))
    n_notrelevant = sum(1 for p in points.values()
                        if not p.get("relevant"))
    n_nonviable = n_total - n_relevant - n_notrelevant
    n_supp = len(suppressed)
    n_nosupp = n_viable - n_supp
    frac = n_supp / n_viable if n_viable else 0.0
    sigma = math.sqrt(frac * (1 - frac) / n_viable) if n_viable else 0.0

    # Non-viable reason split (within the relevant window).
    n_low_n = sum(1 for p in points.values()
                  if p.get("relevant") and not p["viable"]
                  and p.get("N_total") is not None
                  and p["N_total"] <= N_TOTAL_MIN)
    n_tv = sum(1 for p in points.values()
               if p.get("relevant") and not p["viable"]
               and p.get("t_over_v") is not None
               and p["t_over_v"] > 1.0)

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    # Bar 1: all grid points relevant-viable + not-relevant + non-viable.
    y1 = 1.0
    ax.barh(y1, n_relevant, color=TOL["blue"], label="viable & relevant")
    ax.barh(y1, n_notrelevant, left=n_relevant, color=TOL["grey"],
            label="not relevant (N_total>73)")
    ax.barh(y1, n_nonviable, left=n_relevant + n_notrelevant,
            color="#CCCCCC", label="non-viable")
    ax.text(n_total + 0.5, y1, f"{n_relevant}/{n_total}",
            va="center", fontsize=7)

    # Bar 2: relevant-viable points suppressed vs not.
    y2 = 0.0
    ax.barh(y2, n_supp, color=TOL["red"], label="suppressed")
    ax.barh(y2, n_nosupp, left=n_supp, color=TOL["yellow"],
            label="not suppressed")
    ax.text(n_viable + 0.5, y2, f"{n_supp}/{n_viable} = {100*frac:.1f}% "
            f"$\\pm$ {100*sigma:.1f}%", va="center", fontsize=7)

    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(["all grid", "viable"], fontsize=8)
    ax.set_xlabel("points", fontsize=9)
    ax.set_xlim(0, n_total * 1.7)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=6, framealpha=0.8)
    fig.subplots_adjust(bottom=0.24)
    # Failure-reason caption below the axes (avoids in-bar crowding).
    fig.text(0.28, 0.03,
             f"non-viable: low N*={n_low_n}, T>V={n_tv}",
             fontsize=6, color=TOL["dark"])
    save_fig(fig, filename, category)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def setup_args():
    p = argparse.ArgumentParser(
        description="Fine-tune fraction scan (Cardenas correction 3)")
    sub = p.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("explore", help="mine existing logs (no compute)")
    pe.set_defaults(func=explore)

    ps = sub.add_parser("scan", help="uniform grid scan (final number)")
    ps.add_argument("--n-phi0", type=int, default=35)
    ps.add_argument("--n-y0", type=int, default=35)
    ps.add_argument("--workers", type=int, default=4)
    ps.add_argument("--x0-min", type=float, default=X0_MIN)
    ps.add_argument("--x0-max", type=float, default=X0_MAX)
    ps.add_argument("--y0-min", type=float, default=Y0_MIN)
    ps.add_argument("--y0-max", type=float, default=Y0_MAX)
    ps.add_argument("--resume", type=str, default=None,
                    help="resume from an existing scan JSONL")
    ps.set_defaults(func=scan)

    pr = sub.add_parser("report", help="read scan JSONL -> summary + plots")
    pr.add_argument("--log", type=str, required=True,
                    help="scan JSONL path")
    pr.add_argument("--n-phi0", type=int, default=35)
    pr.add_argument("--n-y0", type=int, default=35)
    pr.set_defaults(func=report)

    return p.parse_args()


def main():
    args = setup_args()
    args.func(args)


if __name__ == "__main__":
    main()