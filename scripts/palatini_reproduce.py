"""Reproduce the metric-Higgs power-suppression study in the Palatini formulation.

Reuses the existing evaluation chain in `scripts.camb_scan` rather than
re-implementing it: `evaluate_config` already runs
P_S(k) -> CAMB C_ell -> D_ell -> chi2 against Planck 2018, and returns the
suppression / chi2 quantities used in the AGENTS.md section 12 reference table.
Its only model coupling is an optional `model` argument, so any
`models.registry` model can be driven through the identical chain.

Each input JSON is one (model, initial conditions, N_star) point:

    {
      "model": "PalatiniHiggsModel",
      "model_params": {"lam": 0.13, "xi": 15000.0},
      "ics": {"x0": 4.49e-4, "y0": -1.2e-4},
      "N_star": 55.0
    }

The metric-Higgs reference row is emitted alongside every record so the
suppression comparison is always in the same units and conventions.

Usage:
    PYTHONPATH=. python scripts/palatini_reproduce.py \
        --config configs/palatini/palatini_usr_a.json configs/palatini/palatini_usr_b.json \
        --n-workers 20 --label palatini_xi15000
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from models.registry import MODEL_REGISTRY
from scripts.camb_scan import (
    D2_LCDM,
    CHI2_LCDM,
    evaluate_config,
)


# AGENTS.md section 12 reference rows, metric Higgs (xi=15000, lam=0.13),
# current post-solve_ivp pipeline. Emitted for direct comparison.
METRIC_REFERENCE = {
    "6.40/-0.475/59": {"chi2": 2574.2, "d2": 997.0, "suppression_pct": 31.0},
    "5.70/-0.170/52": {"chi2": 2582.6, "d2": 851.0, "suppression_pct": 36.0},
    "5.75/-0.170/55": {"chi2": 2613.8, "d2": 784.0, "suppression_pct": 39.0},
    "LCDM": {"chi2": 2573.0, "d2": D2_LCDM, "suppression_pct": 0.0},
}


def load_config(path):
    with open(path) as fh:
        cfg = json.load(fh)
    for key in ("model", "N_star"):
        if key not in cfg:
            raise ValueError(f"{path}: missing required key {key!r}")
    return cfg


def run_one(job):
    """Evaluate one config through the existing camb_scan chain."""
    (path, args_dict) = job
    cfg = load_config(path)
    factory = MODEL_REGISTRY[cfg["model"]]
    model = factory(cfg.get("model_params", {}), cfg.get("inflection", {}))
    ics = cfg.get("ics", {})
    x0 = float(ics.get("x0", model.x0))
    y0 = float(ics.get("y0", model.y0))
    n_star = float(cfg["N_star"])

    ns = argparse.Namespace(**args_dict)
    out = evaluate_config(x0, y0, n_star, ns, model=model)
    out["config_file"] = os.path.basename(path)
    out["model"] = cfg["model"]
    out["model_params"] = cfg.get("model_params", {})
    out["x0"] = x0
    out["y0"] = y0
    out["metric_reference"] = METRIC_REFERENCE
    out["d2_lcdm"] = D2_LCDM
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", nargs="+", required=True,
                   help="one or more config JSON files, one point each")
    p.add_argument("--label", default="palatini", help="output/log label")
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--full-chi2", action="store_true",
                   help="use binned+unbinned chi2 over the full ell range")
    p.add_argument("--k-min", type=float, default=1e-5)
    p.add_argument("--k-max", type=float, default=1.0)
    p.add_argument("--num-k", type=int, default=80)
    p.add_argument("--workers", type=int, default=1,
                   help="MS k-mode workers INSIDE one pipeline call")
    p.add_argument("--ell-max", type=int, default=2500)
    p.add_argument("--output-dir", default=os.path.join("outputs", "simulations", "logs"))
    return p.parse_args()


def main():
    args = parse_args()
    args_dict = {
        "k_min": args.k_min,
        "k_max": args.k_max,
        "num_k": args.num_k,
        "workers": args.workers,
        "ell_max": args.ell_max,
        "full_chi2": args.full_chi2,
    }
    jobs = [(c, args_dict) for c in args.config]
    print(f"{len(jobs)} configs, model chain = scripts.camb_scan.evaluate_config, "
          f"n_workers={args.n_workers}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"reproduce_{args.label}.jsonl")

    records = []
    with open(out_path, "a") as fh:
        if args.n_workers > 1:
            with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
                futs = [pool.submit(run_one, j) for j in jobs]
                for n, fut in enumerate(as_completed(futs), 1):
                    rec = fut.result()
                    records.append(rec)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    print(f"[{n}/{len(jobs)}] {_line(rec)}", flush=True)
        else:
            for n, j in enumerate(jobs, 1):
                rec = run_one(j)
                records.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"[{n}/{len(jobs)}] {_line(rec)}", flush=True)

    _summary(records)
    print(f"\nwrote {len(records)} records -> {out_path}")


def _line(rec):
    if rec.get("status") != "ok":
        return f"{rec.get('config_file')} FAILED: {rec.get('status')} {rec.get('error','')}"
    return (f"{rec['config_file']:<28} x0={rec['x0']:.4g} y0={rec['y0']:.4g} "
            f"N*={rec['N_star']:.0f} Ntot={rec['N_total']:.1f} "
            f"chi2={rec['chi2']:.1f} (LCDM {rec['chi2_lcdm']:.1f}) "
            f"dchi2={rec['dchi2']:+.1f} D2={rec['d2']:.0f} "
            f"supp={rec['suppression_pct']:.0f}%")


def _summary(records):
    ok = [r for r in records if r.get("status") == "ok"]
    if not ok:
        print("no successful records")
        return
    print("\n=== suppression comparison (D2 in uK^2) ===")
    print(f"{'config':<30}{'chi2':>9}{'dchi2':>9}{'D2':>9}{'supp%':>8}")
    for key, ref in METRIC_REFERENCE.items():
        print(f"{'metric ref ' + key:<30}{ref['chi2']:>9.1f}{'':>9}{ref['d2']:>9.1f}"
              f"{ref['suppression_pct']:>8.0f}")
    print(f"{'LCDM (camb)':<30}{CHI2_LCDM:>9.1f}{'':>9}{D2_LCDM:>9.1f}{0.0:>8.0f}")
    print("-" * 64)
    for r in sorted(ok, key=lambda r: r["dchi2"]):
        print(f"{r['config_file']:<30}{r['chi2']:>9.1f}{r['dchi2']:>9.1f}"
              f"{r['d2']:>9.1f}{r['suppression_pct']:>8.0f}")


if __name__ == "__main__":
    main()
