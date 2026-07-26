"""Scalar-Induced Gravitational Wave (SIGW) module.

Kohri & Terada (2018) — oscillation-averaged radiation-era kernel,
adaptive double integral, present-day dilution, LISA sensitivity
curve (Robson+ 2019), and SNR.
"""

import numpy as np
from numpy import trapezoid
from scipy.interpolate import CubicSpline

from scripts.constants import KM_PER_S, MPC_CM, C_LIGHT, CAMB_COSMOLOGY

# ── Physical constants ──────────────────────────────────────────────────────

OMEGA_GAMMA0_H2 = 2.473e-5  # Ω_γ,0 h² (photons only)

# Dilution from radiation era to present: standard value from
# Kohri & Terada (2018), accounts for Ω_γ,0 h², entropy dilution,
# and g_* transition at the QCD epoch.
DILUTION = 1.62e-5

# ── Frequency conversion ────────────────────────────────────────────────────

def k_to_f(k_phys):
    """Convert comoving wavenumber k [Mpc⁻¹] to GW frequency f [Hz]."""
    k_cgs = np.asarray(k_phys, dtype=float) / MPC_CM
    return k_cgs * C_LIGHT / (2.0 * np.pi)


def f_to_k(f_hz):
    """Convert GW frequency f [Hz] to comoving wavenumber k [Mpc⁻¹]."""
    f = np.asarray(f_hz, dtype=float)
    return f * 2.0 * np.pi / C_LIGHT * MPC_CM


# ── SIGW kernel ─────────────────────────────────────────────────────────────

def kernel_rd(u, v):
    """Oscillation-averaged SIGW kernel in radiation domination.

    Full integrand = projection factor × I²_RD(u,v), where I²_RD is
    the Kohri & Terada (2018) Eq. (29) time integral.

    The Ω_GW formula is:
      Ω_GW(k) = (1/6) ∫ dv ∫ du  K(u,v)  P_R(uk) P_R(vk)

    Parameters
    ----------
    u, v : ndarray
        Dimensionless ratios k_i/k. Both positive.

    Returns
    -------
    K : ndarray
        Kernel values.
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    v = np.atleast_1d(np.asarray(v, dtype=float))

    shape = np.broadcast_shapes(u.shape, v.shape)
    u_b = np.broadcast_to(u, shape)
    v_b = np.broadcast_to(v, shape)

    domain = (np.abs(1.0 - u_b) <= v_b) & (v_b <= 1.0 + u_b)
    # Exclude singularities at u=0 or v=0 (kernel returns 0 there)
    domain = domain & (u_b > 1e-300) & (v_b > 1e-300)
    K = np.zeros(shape, dtype=float)
    if not np.any(domain):
        return K if K.shape != (1,) else 0.0

    ud = u_b[domain]
    vd = v_b[domain]

    u2 = ud * ud
    v2 = vd * vd
    u3v3 = np.clip(ud ** 3 * vd ** 3, 1e-300, None)

    # --- Projection factor (scalar-to-tensor transfer) ---
    # P(u,v) = [4v² - (1 - u² + v²)²]² / (4uv)²
    proj_num = (4.0 * v2 - (1.0 - u2 + v2) ** 2) ** 2
    proj_den = np.clip(4.0 * ud * vd, 1e-300, None)
    proj = proj_num / (proj_den * proj_den)

    # --- Time integral I²_RD (Kohri & Terada 2018 Eq. 29) ---
    u2v2_3 = u2 + v2 - 3.0
    coeff = 0.5 * (3.0 * u2v2_3 / (4.0 * u3v3)) ** 2

    # Log argument
    num = 3.0 - (ud + vd) ** 2
    den = 3.0 - (ud - vd) ** 2
    ratio = np.abs(num / np.clip(den, 1e-300, None))
    log_val = np.log(np.clip(ratio, 1e-300, None))

    body = -4.0 * ud * vd + u2v2_3 * log_val

    # Resonance: π² (u²+v²-3)² θ(u+v > √3)
    resonance = np.pi ** 2 * u2v2_3 ** 2
    resonance = np.where(ud + vd > np.sqrt(3.0), resonance, 0.0)

    I2 = coeff * (body * body + resonance)
    K[domain] = proj * I2

    return K if K.shape != (1,) else K.item()


# ── SIGW spectrum (adaptive per-k_GW integration) ───────────────────────────

def compute_sigw_spectrum(k_phys, P_S, f_gw, num_grid=150):
    """Compute present-day Omega_GW,0(f) h^2 from P_S(k).

    Adaptive per-k_GW integration: u-grid adapts to the domain where
    P_S(u*k_GW) is non-negligible, avoiding the kernel divergence
    at u,v→0 and polynomial growth at u,v→∞.

    Parameters
    ----------
    k_phys : ndarray  — wavenumbers [Mpc⁻¹]
    P_S : ndarray     — P_R(k)
    f_gw : ndarray   — GW frequencies [Hz]
    num_grid : int    — points per dimension

    Returns
    -------
    omega_gw_h2 : ndarray  — Omega_GW,0(f) h^2
    """
    ok = np.isfinite(P_S) & (P_S > 0) & (k_phys > 0)
    k_ok = k_phys[ok]
    ps_ok = P_S[ok]
    k_min_ps = k_ok.min()
    k_max_ps = k_ok.max()

    ps_spl = CubicSpline(np.log(k_ok), np.log(ps_ok),
                         bc_type='natural', extrapolate=False)

    def ps_log(k):
        kc = np.clip(np.asarray(k, dtype=float),
                     k_min_ps * 1.001, k_max_ps * 0.999)
        return np.exp(ps_spl(np.log(kc)))

    omega_gw_h2 = np.zeros_like(f_gw, dtype=float)
    n_u = num_grid
    n_v = max(int(num_grid * 1.15), 100)

    for i, f in enumerate(f_gw):
        k_gw = f_to_k(f)

        u_lo = max(k_min_ps / k_gw, 1e-8)
        u_hi = min(k_max_ps / k_gw, 1e8)

        if u_lo >= u_hi * 0.999:
            continue

        u_grid = np.exp(np.linspace(np.log(u_lo), np.log(u_hi), n_u))

        integrand = np.zeros(n_u)
        for ui, u_val in enumerate(u_grid):
            v_lo = max(abs(1.0 - u_val), k_min_ps / k_gw)
            v_hi = min(1.0 + u_val, k_max_ps / k_gw)

            if v_lo >= v_hi:
                continue

            v_grid = np.exp(np.linspace(np.log(max(v_lo, 1e-10)),
                                         np.log(v_hi), n_v))

            pu = ps_log(u_val * k_gw)
            pv = ps_log(v_grid * k_gw)
            K = kernel_rd(u_val, v_grid)

            integrand[ui] = pu * np.trapezoid(pv * K, v_grid)

        omega_gw_h2[i] = np.trapezoid(integrand, u_grid) / 6.0 * DILUTION

    return omega_gw_h2


# ── LISA noise curve ────────────────────────────────────────────────────────

def get_lisa_noise(f):
    """LISA sensitivity curve (Robson, Cornish & Liu 2019).

    Parameters
    ----------
    f : ndarray  — frequencies [Hz]

    Returns
    -------
    omega_noise_h2 : ndarray  — Omega_noise(f) h^2
    """
    f = np.asarray(f, dtype=float)
    L = 2.5e9  # m, arm length
    f_star = C_LIGHT / (2.0 * np.pi * L)  # ~0.019 Hz

    f_clip = np.clip(f, 1e-12, None)

    # Acc noise
    P_acc = (3.0e-15) ** 2 * (1.0 + (4.0e-4 / f_clip) ** 2) * \
            (1.0 + (f_clip / 8.0e-3) ** 4)

    # OMS noise
    P_oms = (1.5e-11) ** 2 * (1.0 + (2.0e-3 / f_clip) ** 4)

    # Strain PSD
    S_n = (10.0 / (3.0 * L ** 2)) * (
        P_oms + 2.0 * (1.0 + np.cos(f_clip / f_star) ** 2) * P_acc / (2.0 * np.pi * f_clip) ** 4
    )

    # Galactic confusion
    A_gal = 1.15e-44
    S_gal = A_gal * f_clip ** (-7.0 / 3.0) * np.exp(-f_clip / 2.0e-3)
    S_n_total = S_n + S_gal

    # Convert to Omega_noise
    h = CAMB_COSMOLOGY["H0"] / 100.0
    H0_sq = (h * 100.0 * KM_PER_S / MPC_CM) ** 2
    omega_noise = (4.0 * np.pi ** 2) / (3.0 * H0_sq) * f ** 3 * S_n_total
    omega_noise_h2 = omega_noise * h ** 2

    return omega_noise_h2


# ── SNR computation ─────────────────────────────────────────────────────────

def compute_snr(f, omega_gw_h2, omega_noise_h2, t_obs_yr=4.0):
    """LISA cross-correlation SNR (A+E+T channels).

    Parameters
    ----------
    f : ndarray  — frequencies [Hz]
    omega_gw_h2 : ndarray  — Omega_GW(f) h^2
    omega_noise_h2 : ndarray  — Omega_noise(f) h^2
    t_obs_yr : float  — observation time [yr]

    Returns
    -------
    snr : float
    """
    t_obs = t_obs_yr * 365.25 * 86400.0
    integrand = (omega_gw_h2 / np.clip(omega_noise_h2, 1e-300, None)) ** 2
    snr_sq = 2.0 * t_obs * trapezoid(integrand, f)
    return np.sqrt(max(snr_sq, 0.0))


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_sigw_lisa(frequency_hz, omega_gw_h2, lisa_noise_h2, snr,
                   filename="sigw_lisa", chi0=None, y0=None, nstar=None,
                   formation=None, accretion=None):
    """Publication-quality LISA Omega_GW overlay plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.plotting import (TOL, PAPER_RCPARAMS, get_path, save_fig,
                                   make_pbh_filename)

    with plt.rc_context(PAPER_RCPARAMS):
        fig, ax = plt.subplots(figsize=(3.4, 2.6))

        ax.loglog(frequency_hz, omega_gw_h2, "-", color=TOL["red"], lw=1.3,
                  label=r"$\Omega_{\rm GW,0}(f)h^2$")
        ax.loglog(frequency_hz, lisa_noise_h2, "--", color=TOL["grey"], lw=1.0,
                  label=r"LISA A+E+T ($T_{\rm obs}=4$ yr)")

        # Planck Omega_r0 h^2 reference
        ax.axhline(OMEGA_GAMMA0_H2, color=TOL["grey"], ls=":", lw=0.8, alpha=0.5)

        ax.text(0.97, 0.05, f"SNR = {snr:.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color=TOL["dark"])

        ax.set_xlabel(r"$f$ [Hz]")
        ax.set_ylabel(r"$\Omega_{\rm GW,0}(f)h^2$")
        ax.legend(loc="upper right", fontsize=6)

        ax.set_xlim(frequency_hz[0], frequency_hz[-1])
        y_min = max(min(omega_gw_h2.min(), lisa_noise_h2[lisa_noise_h2 > 0].min()) * 0.5, 1e-30)
        ax.set_ylim(y_min, None)

        fig.tight_layout()

        if chi0 is not None and formation is not None:
            fname = make_pbh_filename(
                filename, chi0, 0.0 if y0 is None else y0,
                0.0 if nstar is None else nstar,
                formation=formation, accretion=accretion, ext=".png",
            )
            path = get_path("sigw", fname)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return path
        else:
            save_fig(fig, filename, "sigw")
            fname = filename if filename.endswith(".png") else filename + ".png"
            return get_path("sigw", fname)


# ── Convenience: full SIGW pipeline ─────────────────────────────────────────

def run_sigw(k_phys, P_S, f_min=1e-5, f_max=1.0, num_f=150,
             num_grid=150, t_obs_yr=4.0):
    """Compute full SIGW spectrum + LISA noise + SNR in one call."""
    f_gw = np.logspace(np.log10(f_min), np.log10(f_max), num_f)
    omega_gw_h2 = compute_sigw_spectrum(k_phys, P_S, f_gw, num_grid=num_grid)
    omega_noise = get_lisa_noise(f_gw)

    snr = compute_snr(f_gw, omega_gw_h2, omega_noise, t_obs_yr=t_obs_yr)

    if np.all(omega_gw_h2 <= 0):
        f_peak = None
        omega_peak = None
    else:
        peak_idx = np.argmax(omega_gw_h2)
        f_peak = float(f_gw[peak_idx])
        omega_peak = float(omega_gw_h2[peak_idx])

    return {
        "frequency_hz": f_gw,
        "omega_gw_h2": omega_gw_h2,
        "lisa_noise_h2": omega_noise,
        "snr": float(snr),
        "f_peak": f_peak,
        "omega_peak": omega_peak,
        "dilution": float(DILUTION),
        "t_obs_yr": t_obs_yr,
    }
