#!/usr/bin/env python3
"""Compare n_s from lsq-fit vs logarithmic derivative methods."""
import json, glob, os, sys
import numpy as np
sys.path.insert(0, '.')
from scripts.observables import extract_ns

def load_ps(f):
    with open(f) as fh:
        d = json.load(fh)
    if 'spectrum' in d:
        return np.array(d['spectrum']['k_phys']), np.array(d['spectrum']['P_S']), d.get('metadata', {})
    elif 'k_phys' in d and 'P_S' in d:
        return np.array(d['k_phys']), np.array(d['P_S']), d
    else:
        return None, None, None

PS_DIR = 'outputs/simulations/pspectra'
# All non-checkpoint files
files = sorted(glob.glob(os.path.join(PS_DIR, '*.json')))
files = [f for f in files if '_checkpoint' not in os.path.basename(f)]

for f in files:
    k, p, md = load_ps(f)
    if k is None or len(k) < 5:
        print(f'{os.path.basename(f)}: SKIP (format unknown or too few points)')
        continue
    label = os.path.basename(f)
    N_star = md.get('N_star', '?')

    # Determine pivot based on k-range
    has_cmb = np.any(k < 0.1)
    has_pbh = np.any(k > 1e6)

    results = []
    # CMB pivot tests
    if has_cmb:
        for kp, w in [(0.002, 4.0), (0.05, 3.0)]:
            ns_lsq, _ = extract_ns(k, p, k_pivot=kp, ns_window=w, method='lsq')
            ns_der, m = extract_ns(k, p, k_pivot=kp, method='derivative')
            results.append((kp, w, ns_lsq, ns_der, m.get('error')))

    print(f'{label}  N_star={N_star}')
    if not results:
        print('  (no CMB scales in this file)')
    for kp, w, ns_l, ns_d, err in results:
        diff = abs(ns_l - ns_d) if ns_l is not None and ns_d is not None else -1
        lsq_s = f'{ns_l:.6f}' if ns_l else 'NONE'
        der_s = f'{ns_d:.6f}' if ns_d else f'FAIL({err})' if err else 'NONE'
        print(f'  k={kp}:  lsq(w={w})={lsq_s}  der={der_s}  Δ={diff:.2e}')
    print()
