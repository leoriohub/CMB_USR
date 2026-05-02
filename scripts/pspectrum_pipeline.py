import argparse
import json
import os
import sys
import uuid

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import inf_dyn_background as bg_solver
import inf_dyn_MS_full as ms_solver
from models import HiggsModel, FullHiggsModel


def find_end_of_inflation(epsH):
    in_inflation = False
    for idx, eps in enumerate(epsH):
        if not in_inflation:
            if eps < 1.0:
                in_inflation = True
        else:
            if eps >= 1.0:
                return idx
    return -1


def ensure_k_pivot(k_grid, k_pivot_phys, rtol=1e-6):
    if np.any(np.isclose(k_grid, k_pivot_phys, rtol=rtol, atol=0.0)):
        pivot_idx = int(np.where(np.isclose(k_grid, k_pivot_phys, rtol=rtol, atol=0.0))[0][0])
        return k_grid, pivot_idx
    k_grid = np.sort(np.append(k_grid, k_pivot_phys))
    pivot_idx = int(np.where(k_grid == k_pivot_phys)[0][0])
    return k_grid, pivot_idx


def get_k_pivot_code(bg_sol, derived_bg, end_idx, N_pivot):
    N_total = derived_bg["N"][end_idx]
    if N_total < N_pivot:
        return None, None, None
    N_pivot_val = N_total - N_pivot
    pivot_idx = int(np.argmin(np.abs(derived_bg["N"][:end_idx] - N_pivot_val)))
    z_pivot = bg_sol[2][pivot_idx]
    a_pivot = np.exp(bg_sol[3][pivot_idx])
    k_pivot_code = a_pivot * z_pivot
    return k_pivot_code, pivot_idx, N_total


def extract_mode_initial_conditions(bg_sol, T_span_bg, end_idx, k_code, k_start_factor):
    n_bg = bg_sol[3]
    z_bg = bg_sol[2]
    log_az = n_bg + np.log(z_bg)
    target_start = np.log(k_code) - np.log(k_start_factor)
    start_idx = int(np.argmin(np.abs(log_az[:end_idx] - target_start)))
    start_idx = max(start_idx, 0)
    xi = bg_sol[0][start_idx]
    yi = bg_sol[1][start_idx]
    zi = bg_sol[2][start_idx]
    ni = bg_sol[3][start_idx]
    t_start = T_span_bg[start_idx]
    t_end = T_span_bg[end_idx]
    return xi, yi, zi, ni, t_start, t_end, start_idx


def _compute_single_mode(args):
    idx, k_code, bg_sol, T_span_bg, end_idx, bg_interp, k_start_factor, ms_steps, model = args
    xi, yi_val, zi, ni, t_start, t_end, start_idx = extract_mode_initial_conditions(
        bg_sol, T_span_bg, end_idx, k_code, k_start_factor
    )
    T_ms = np.linspace(t_start, t_end, ms_steps)
    ms_sol = ms_solver.run_ms_simulation(bg_interp, ni, T_ms, k_code, model)
    derived_ms = ms_solver.get_ms_derived_quantities_with_bg(ms_sol, bg_interp, T_ms, model, k_code, ni)
    return idx, float(derived_ms["P_S"][-1]), float(derived_ms["P_T"][-1]), start_idx


def run_pspectrum_pipeline(
    model,
    phi0=None,
    yi=None,
    k_min=1e-5,
    k_max=1.0,
    num_k=80,
    k_pivot_phys=0.05,
    N_pivot=60.0,
    k_start_factor=100.0,
    T_span_bg=None,
    bg_steps=10000,
    T_max=5000.0,
    ms_steps=5000,
    normalize_to_As=True,
    As=2.1e-9,
    output_dir="outputs/cmb_results/pspectra",
    save_outputs=True,
    k_phys_grid=None,
    n_workers=1,
):
    if phi0 is not None:
        model.phi0 = float(phi0)
    if yi is not None:
        model.yi = float(yi)

    if T_span_bg is None:
        T_span_bg = np.linspace(0.0, T_max, bg_steps)

    bg_sol = bg_solver.run_background_simulation(model, T_span_bg)
    derived_bg = bg_solver.get_derived_quantities(bg_sol, model)

    end_idx = find_end_of_inflation(derived_bg["epsH"])
    if end_idx == -1:
        return {"status": "error", "message": "Inflation did not end in background window."}

    k_pivot_code, pivot_bg_idx, N_total = get_k_pivot_code(bg_sol, derived_bg, end_idx, N_pivot)
    if k_pivot_code is None:
        return {
            "status": "error",
            "message": f"Total inflation ({derived_bg['N'][end_idx]:.2f}) is less than N_pivot ({N_pivot}).",
        }

    if not (k_min <= k_pivot_phys <= k_max):
        return {
            "status": "error",
            "message": "k_pivot_phys must be within [k_min, k_max] to normalize spectrum.",
        }

    if k_phys_grid is not None and len(k_phys_grid) > 0:
        k_phys_grid = np.asarray(k_phys_grid, dtype=float)
    else:
        k_phys_grid = np.logspace(np.log10(k_min), np.log10(k_max), num_k)
    k_phys_grid, pivot_idx = ensure_k_pivot(k_phys_grid, k_pivot_phys)
    k_code_grid = k_pivot_code * (k_phys_grid / k_pivot_phys)

    P_S_raw = np.zeros_like(k_phys_grid)
    P_T_raw = np.zeros_like(k_phys_grid)
    start_indices = np.zeros_like(k_phys_grid, dtype=int)

    # Build background interpolation once — avoids re-integrating background per k-mode
    bg_interp = ms_solver.build_bg_interpolators(bg_sol, T_span_bg)

    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        tasks = [
            (idx, k_code, bg_sol, T_span_bg, end_idx, bg_interp, k_start_factor, ms_steps, model)
            for idx, k_code in enumerate(k_code_grid)
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for idx, ps, pt, si in executor.map(_compute_single_mode, tasks):
                P_S_raw[idx] = ps
                P_T_raw[idx] = pt
                start_indices[idx] = si
    else:
        for idx, k_code in enumerate(k_code_grid):
            xi, yi_val, zi, ni, t_start, t_end, start_idx = extract_mode_initial_conditions(
                bg_sol, T_span_bg, end_idx, k_code, k_start_factor
            )
            T_ms = np.linspace(t_start, t_end, ms_steps)
            ms_sol = ms_solver.run_ms_simulation(bg_interp, ni, T_ms, k_code, model)
            derived_ms = ms_solver.get_ms_derived_quantities_with_bg(ms_sol, bg_interp, T_ms, model, k_code, ni)
            P_S_raw[idx] = float(derived_ms["P_S"][-1])
            P_T_raw[idx] = float(derived_ms["P_T"][-1])
            start_indices[idx] = start_idx

    scale_factor = 1.0
    if normalize_to_As:
        P_S_pivot = P_S_raw[pivot_idx]
        if P_S_pivot <= 0:
            return {"status": "error", "message": "Pivot P_S is non-positive; cannot normalize."}
        scale_factor = As / P_S_pivot

    P_S = P_S_raw * scale_factor
    P_T = P_T_raw * scale_factor

    metadata = {
        "model": model.name,
        "phi0": float(model.phi0),
        "yi": float(model.yi),
        "xi": getattr(model, "xi_val", None),
        "lam": getattr(model, "lam", None),
        "v_vev": getattr(model, "v_vev", None),
        "k_min": float(k_min),
        "k_max": float(k_max),
        "num_k": int(len(k_phys_grid)),
        "k_pivot_phys": float(k_pivot_phys),
        "k_pivot_code": float(k_pivot_code),
        "N_pivot": float(N_pivot),
        "N_total": float(N_total),
        "pivot_bg_idx": int(pivot_bg_idx),
        "pivot_k_idx": int(pivot_idx),
        "k_start_factor": float(k_start_factor),
        "normalize_to_As": bool(normalize_to_As),
        "As_target": float(As),
        "scale_factor": float(scale_factor),
        "bg_steps": int(bg_steps),
        "ms_steps": int(ms_steps),
        "T_max": float(T_max),
    }

    output_path = None
    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)
        run_id = str(uuid.uuid4())[:8]
        safe_model = model.name.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"{safe_model}_phi{model.phi0:.2f}_yi{model.yi:.3f}_run_{run_id}.json"
        output_path = os.path.join(output_dir, filename)

        def convert(val):
            if isinstance(val, (np.floating,)):
                return float(val)
            if isinstance(val, (np.integer,)):
                return int(val)
            if isinstance(val, np.ndarray):
                return val.tolist()
            return val

        record = {
            "metadata": {k: convert(v) for k, v in metadata.items()},
            "spectrum": {
                "k_phys": convert(k_phys_grid),
                "k_code": convert(k_code_grid),
                "P_S": convert(P_S),
                "P_T": convert(P_T),
                "P_S_raw": convert(P_S_raw),
                "P_T_raw": convert(P_T_raw),
                "start_idx": convert(start_indices),
            },
        }
        with open(output_path, "w") as f:
            json.dump(record, f, indent=2)

    return {
        "status": "success",
        "k_phys": k_phys_grid,
        "P_S": P_S,
        "P_T": P_T,
        "metadata": metadata,
        "output_file": output_path,
    }


def build_model(args):
    if args.model == "HiggsModel":
        return HiggsModel(lam=args.lam, xi=args.xi)
    if args.model == "FullHiggsModel":
        return FullHiggsModel(lam=args.lam, xi=args.xi, v_vev=args.v_vev)
    raise ValueError(f"Unknown model: {args.model} — use HiggsModel or FullHiggsModel")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute P_S(k) across k grid for a model.")
    parser.add_argument("--model", default="HiggsModel", choices=[
        "HiggsModel", "FullHiggsModel"
    ])
    parser.add_argument("--phi0", type=float, default=None)
    parser.add_argument("--yi", type=float, default=None)
    parser.add_argument("--xi", type=float, default=1000.0)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--v-vev", type=float, default=0.0)

    parser.add_argument("--k-min", type=float, default=1e-5)
    parser.add_argument("--k-max", type=float, default=1.0)
    parser.add_argument("--num-k", type=int, default=80)
    parser.add_argument("--k-pivot-phys", type=float, default=0.05)
    parser.add_argument("--N-pivot", type=float, default=60.0)
    parser.add_argument("--k-start-factor", type=float, default=100.0)

    parser.add_argument("--bg-steps", type=int, default=10000)
    parser.add_argument("--T-max", type=float, default=5000.0)
    parser.add_argument("--ms-steps", type=int, default=5000)
    parser.add_argument("--n-workers", type=int, default=1, help="parallel workers (1 = serial)")
    parser.add_argument("--normalize-to-As", action="store_true")
    parser.add_argument("--As", type=float, default=2.1e-9)
    parser.add_argument("--output-dir", default="outputs/cmb_results/pspectra")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model = build_model(args)
    result = run_pspectrum_pipeline(
        model=model,
        phi0=args.phi0,
        yi=args.yi,
        k_min=args.k_min,
        k_max=args.k_max,
        num_k=args.num_k,
        k_pivot_phys=args.k_pivot_phys,
        N_pivot=args.N_pivot,
        k_start_factor=args.k_start_factor,
        T_span_bg=None,
        bg_steps=args.bg_steps,
        T_max=args.T_max,
        ms_steps=args.ms_steps,
        n_workers=args.n_workers,
        normalize_to_As=args.normalize_to_As,
        As=args.As,
        output_dir=args.output_dir,
        save_outputs=not args.no_save,
    )

    if result["status"] != "success":
        print(result["message"])
        return
    print(f"Saved: {result['output_file']}")


def load_pspectrum(path):
    with open(path) as f:
        record = json.load(f)
    meta = record["metadata"]
    spec = record["spectrum"]
    return {
        "metadata": meta,
        "k_phys": np.array(spec["k_phys"]),
        "k_code": np.array(spec["k_code"]),
        "P_S": np.array(spec["P_S"]),
        "P_T": np.array(spec["P_T"]),
        "P_S_raw": np.array(spec["P_S_raw"]),
        "P_T_raw": np.array(spec["P_T_raw"]),
        "start_idx": np.array(spec["start_idx"]),
    }


if __name__ == "__main__":
    main()
