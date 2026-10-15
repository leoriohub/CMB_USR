"""Report table for the Palatini-vs-metric-Higgs comparison.

Reads the JSONL written by `scripts/palatini_reproduce.py` and prints one row
per config with the quantities needed for the suppression comparison, plus a
fingerprint of the P_S(k) arrays so identical spectra cannot be mistaken for
independent runs.

Usage:
    PYTHONPATH=. python scripts/palatini_report.py \
        outputs/simulations/logs/reproduce_palatini_vs_metric.jsonl
"""

import argparse
import hashlib
import json


def fingerprint(ps):
    if not ps:
        return "none", 0
    return hashlib.md5(json.dumps(ps).encode()).hexdigest()[:12], len(ps)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl", help="reproduce_*.jsonl output")
    p.add_argument("--show-ps", action="store_true",
                   help="print P_S(k) min/max alongside the fingerprint")
    args = p.parse_args()

    with open(args.jsonl) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    rows.sort(key=lambda r: (r.get("model", ""), r.get("config_file", "")))

    hdr = (f"{'config':<30}{'model':<22}{'status':<12}{'chi2':>11}{'dchi2':>10}"
           f"{'D2':>10}{'k_dip':>12}{'supp%':>8}{'Ntot':>8}{'P_S hash':>15}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        fp, n = fingerprint(r.get("P_S"))
        print(f"{str(r.get('config_file','?')):<30}{str(r.get('model','?')):<22}"
              f"{str(r.get('status','?')):<12}"
              f"{float(r.get('chi2', float('nan'))):>11.1f}"
              f"{float(r.get('dchi2', float('nan'))):>10.1f}"
              f"{float(r.get('d2', float('nan'))):>10.1f}"
              f"{float(r.get('k_dip', float('nan'))):>12.3e}"
              f"{float(r.get('suppression_pct', float('nan'))):>8.1f}"
              f"{float(r.get('N_total', float('nan'))):>8.1f}{fp:>15}")
        if args.show_ps:
            ps = r.get("P_S") or []
            if ps:
                print(f"{'':<30}P_S n={n} min={min(ps):.4e} max={max(ps):.4e}")

    print("\nNotes on the columns:")
    print("  chi2   : chi2 vs Planck over the requested ell range (--full-chi2 => unbinned, full range)")
    print("  dchi2  : chi2(model) - chi2(LCDM) computed on the SAME ell range")
    print("  D2     : quadrupole D_ell[ell=2] in uK^2")
    print("  k_dip  : comoving wavenumber of the P_S(k) dip")
    print("  supp%  : P_S dip depth relative to P_S(k_pivot), NOT the D2 suppression")
    print("  P_S hash: identical fingerprints => identical spectra (not independent evidence)")

    dup = {}
    for r in rows:
        fp, _ = fingerprint(r.get("P_S"))
        dup.setdefault(fp, []).append(r.get("config_file"))
    for fp, files in dup.items():
        if fp != "none" and len(files) > 1:
            print(f"\n[warning] identical P_S(k) for: {', '.join(map(str, files))}")


if __name__ == "__main__":
    main()
