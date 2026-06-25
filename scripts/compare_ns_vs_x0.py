#!/usr/bin/env python3
"""Compare n_s(LSQ) vs n_s(derivative) on all available cached P_S(k) data.

Output: outputs/plots/paper/ns_method_comparison.png
"""
import json, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from scripts.observables import extract_ns
from scripts.plotting import TOL, PAPER_RCPARAMS

PS_DIR = 'outputs/simulations/pspectra'
files = sorted(glob.glob(os.path.join(PS_DIR, 'ps_phi*.json')))

rows = []
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    if 'spectrum' in d:
        md = d.get('metadata', {})
        k, p = np.array(d['spectrum']['k_phys']), np.array(d['spectrum']['P_S'])
    else:
        continue
    
    x0 = md.get('x0'); y0 = md.get('y0'); ns = md.get('N_star')
    if x0 is None or y0 is None or ns is None:
        continue
    if k.min() > 0.002 or k.max() < 0.002:
        continue
    
    ns_lsq, _ = extract_ns(k, p, k_pivot=0.002, method='lsq')
    ns_der, _ = extract_ns(k, p, k_pivot=0.002, method='derivative')
    if ns_lsq is None or ns_der is None:
        continue
    
    rows.append((x0, y0, ns, ns_lsq, ns_der, abs(ns_lsq-ns_der)))

# Sort by Δ descending
rows.sort(key=lambda r: r[5], reverse=True)

print(f'{"x0":<7} {"y0":<9} {"N*":<6} {"n_s(LSQ)":<11} {"n_s(der)":<11} {"|Δ|":<10}  {"Δ rel":<10}')
print('-'*70)
for x0, y0, ns, nl, nd, d in rows:
    rel = f'{(nd-nl)/max(abs(nl),1e-10)*100:+.1f}%' if nl != 0 else '-'
    print(f'{x0:<7.2f} {y0:<9.3f} {ns:<6.0f} {nl:<11.4f} {nd:<11.4f} {d:<10.3f}  {rel}')

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))

# Left: n_s(LSQ) vs n_s(derivative) scatter
xl, yl = [r[3] for r in rows], [r[4] for r in rows]
ax1.scatter(xl, yl, c=[r[5] for r in rows], cmap='viridis', s=30, alpha=0.8)
mn, mx = min(min(xl), min(yl)), max(max(xl), max(yl))
ax1.plot([mn, mx], [mn, mx], 'k--', lw=0.8, alpha=0.4, label='1:1 line')
ax1.set_xlabel(r'$n_s$ (LSQ)', fontsize=11)
ax1.set_ylabel(r'$n_s$ (derivative)', fontsize=11)
ax1.grid(True, alpha=0.2)

# Right: Δ = n_s der - n_s lsq
ax2.scatter([r[0] for r in rows], [r[4]-r[3] for r in rows],
            c=[r[2] for r in rows], cmap='viridis', s=30, alpha=0.8)
ax2.axhline(0, color='k', ls='--', lw=0.8, alpha=0.4)
ax2.set_xlabel(r'$x_0$', fontsize=11)
ax2.set_ylabel(r'$\Delta n_s = n_s^{der} - n_s^{LSQ}$', fontsize=11)
ax2.grid(True, alpha=0.2)

fig.tight_layout()
out = 'outputs/plots/paper/ns_method_comparison.png'
os.makedirs('outputs/plots/paper', exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f'\nPlot: {out}')
