"""Post-process P_S(k) to extract derived physical quantities.

This is the "classifier" — takes P_S(k) array, returns:
- n_s at CMB pivot
- k_peak (USR peak location)
- P_S_peak (peak amplitude)
- usr_type (weak/transitional/strong)
- mass_bin (asteroid_gap/sub_solar_gap/etc.)
- f_total (PBH abundance)
"""
import numpy as np
from scipy.special import erfc

# Constants
A_S_REF = 2.1e-9
K_PIVOT = 0.05
K_EQ = 1.0
M_EQ = 1.0
GAMMA = 0.4
ACCRETION = 3e7


def extract_ns(k_phys, P_S, pivot_k=K_PIVOT, window=3.0):
    """Spectral index via log-log linear fit around pivot."""
    lo = pivot_k / window
    hi = pivot_k * window
    idx = (k_phys >= lo) & (k_phys <= hi) & np.isfinite(P_S) & (P_S > 1e-30)
    k_fit, ps_fit = k_phys[idx], P_S[idx]
    if len(k_fit) < 3:
        return None, None
    coeffs = np.polyfit(np.log(k_fit), np.log(ps_fit), 1)
    return float(coeffs[0] + 1), float(np.exp(np.interp(
        np.log(pivot_k), np.log(k_fit), np.log(ps_fit))))


def find_peak(k_phys, P_S, min_k=1.0):
    """Find the USR peak (max P_S for k > min_k). Returns dict or None."""
    valid = np.isfinite(P_S) & (P_S > 1e-10) & (k_phys > min_k)
    k, ps = k_phys[valid], P_S[valid]
    if len(k) < 5:
        return None
    i = int(np.argmax(ps))
    return {"k_peak": float(k[i]), "P_S_peak": float(ps[i])}


def classify_usr(N_total):
    """Infer USR type from total e-folds."""
    if N_total > 165:
        return "strong"
    elif N_total > 130:
        return "transitional"
    return "weak"


def classify_mass(M_peak):
    """Classify PBH mass into observational bins."""
    if np.isnan(M_peak) or M_peak <= 0:
        return "unknown"
    if M_peak < 1e-17:
        return "too_light"
    if M_peak < 1e-15:
        return "asteroid_gap"
    if M_peak < 1e-6:
        return "intermediate"
    if M_peak < 1e-2:
        return "sub_solar_gap"
    if M_peak < 1:
        return "sub_stellar"
    if M_peak < 100:
        return "stellar_ligo_ruled_out"
    return "massive"


def mass_from_k(k_peak):
    """Formation mass M_⊙ from k_peak (scalar or array)."""
    k = np.asarray(k_peak, dtype=float)
    return GAMMA * M_EQ * (K_EQ / np.maximum(k, 1e-300)) ** 2 * ACCRETION


def compute_f_total(k_phys, P_S, zeta_c=0.07):
    """Approximate PBH abundance fraction (simplified Press-Schechter)."""
    k = np.asarray(k_phys, dtype=float)
    M = mass_from_k(k)
    bf = erfc(zeta_c / np.sqrt(2 * np.clip(P_S, 1e-300, None)))
    beq = bf * k / K_EQ
    ok = np.isfinite(M) & np.isfinite(beq) & (M > 0) & (beq >= 0)
    if not ok.any():
        return 0.0
    return float(np.trapezoid(beq[ok], np.log(np.maximum(M[ok], 1e-300))))


def extract_usr_info(k_phys, P_S, N_total=None):
    """Extract all derived USR metrics from P_S(k).

    Returns a dict with n_s, A_s, peak info, usr_type, mass_bin, f_total.
    """
    result = {}
    n_s, A_s = extract_ns(k_phys, P_S)
    result["n_s"] = n_s
    result["A_s"] = A_s

    peak = find_peak(k_phys, P_S)
    if peak:
        result["k_peak"] = peak["k_peak"]
        result["P_S_peak"] = peak["P_S_peak"]
        result["PS_ratio"] = peak["P_S_peak"] / A_S_REF
        m_peak = mass_from_k(peak["k_peak"])
        result["M_peak"] = m_peak
        result["mass_bin"] = classify_mass(m_peak)
    else:
        for k in ("k_peak", "P_S_peak", "PS_ratio", "M_peak", "mass_bin"):
            result[k] = None

    result["f_total"] = compute_f_total(k_phys, P_S)

    if N_total is not None:
        result["usr_type"] = classify_usr(N_total)
    else:
        result["usr_type"] = None

    result["suppression_pct"] = None
    if result["A_s"] is not None and result["A_s"] > 0:
        peak_ps = result.get("P_S_peak")
        if peak_ps and peak_ps > result["A_s"]:
            result["suppression_pct"] = 100 * (1 - result["A_s"] / peak_ps)

    return result
