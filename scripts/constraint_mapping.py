"""
PBH constraint mapping and epoch-aware plotting utilities.
# allow: SIZE_OK — BOUND_REGISTRY data table (~65 lines) inflates count;
# code-only portion is 208 pure LOC.

Provides a registry of observational constraints on PBH abundance
(:math:`f_\\text{PBH} = \\Omega_\\text{PBH}/\\Omega_\\text{DM}`) and
functions to map high-redshift constraints to present-day mass space
following De Luca, Franciolini & Riotto (2020), arXiv:2003.12589 Eq. 17.

The registry (:py:data:`BOUND_REGISTRY`) covers ~15 key constraints with
metadata including probe redshift and plotting style.  Constraint data
are loaded from ``data/PBHbounds/bounds/`` (bradkav/PBHbounds format).

Exports
-------
BOUND_REGISTRY : dict
    Constraint metadata keyed by short name.
load_bound_dataset : load constraint data + metadata from registry.
map_constraint_to_present : epoch-mapping of (M, f) pairs.
plot_modern_constraints : plot constraints with optional epoch-mapping.
"""

import os
import warnings
from typing import Any

import numpy as np

from scripts.constants import ROOT_DIR

_BOUNDS_DIR: str = os.path.join(ROOT_DIR, "data", "PBHbounds", "bounds")

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BOUND_REGISTRY: dict[str, dict[str, Any]] = {
    "cmb_agius2024": {
        "file": "CMB-Agius2024.txt",
        "z_probe": 500,
        "label": "CMB (PR+halos, Agius+2024)",
        "color": "#4477AA",
    },
    "cmb_standard": {
        "file": "CMB.txt",
        "z_probe": 500,
        "label": "CMB (BHL, Planck)",
        "color": "#4477AA",
    },
    "microlensing_hsc": {
        "file": "HSC.txt",
        "z_probe": 0,
        "label": "HSC Microlensing",
        "color": "#66CCEE",
    },
    "microlensing_kepler": {
        "file": "K.txt",
        "z_probe": 0,
        "label": "Kepler",
        "color": "#228833",
    },
    "microlensing_eros": {
        "file": "EROS.txt",
        "z_probe": 0,
        "label": "EROS",
        "color": "#CCBB44",
    },
    "microlensing_macho": {
        "file": "M.txt",
        "z_probe": 0,
        "label": "MACHO",
        "color": "#AA3377",
    },
    "microlensing_ogle": {
        "file": "OGLE.txt",
        "z_probe": 0,
        "label": "OGLE",
        "color": "#BBBBBB",
    },
    "microlensing_ogle_hint": {
        "file": "OGLE-hint.txt",
        "z_probe": 0,
        "label": "OGLE (hint)",
        "color": "#BBBBBB",
    },
    "lvk_sgwb": {
        "file": "LIGO-SGWB-O1-O2-O3.txt",
        "z_probe": 0,
        "label": "LVK SGWB O1+O2+O3",
        "color": "#EE6677",
    },
    "lvk_subsolar": {
        "file": "LIGO-subsolar.txt",
        "z_probe": 0,
        "label": "LIGO subsolar",
        "color": "#EE6677",
    },
    "egrb": {
        "file": "EGRB.txt",
        "z_probe": 0,
        "label": "EGRB (Fermi)",
        "color": "#CCBB44",
    },
    "eridanus_ii": {
        "file": "UFdwarfs.txt",
        "z_probe": 0,
        "label": "Eri II",
        "color": "#228833",
    },
    "lyman_alpha": {
        "file": "Lyalphaforest.txt",
        "z_probe": 5,
        "label": "Lyman-α forest",
        "color": "#AA3377",
    },
    "wide_binaries": {
        "file": "WideBinaries.txt",
        "z_probe": 0,
        "label": "Wide Binaries",
        "color": "#4477AA",
    },
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_bound_dataset(name: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load bound data and metadata from the registry.

    Parameters
    ----------
    name : str
        Key in :py:data:`BOUND_REGISTRY`.

    Returns
    -------
    M : ndarray
        PBH masses [:math:`M_\\odot`].
    f_pbh : ndarray
        Upper limit on :math:`f_\\text{PBH} = \\Omega_\\text{PBH}/\\Omega_\\text{DM}`.
    meta : dict
        Full registry entry for this constraint.

    Raises
    ------
    KeyError
        If *name* is not found in the registry.
    FileNotFoundError
        If the corresponding data file does not exist.
    """
    if name not in BOUND_REGISTRY:
        raise KeyError(f"Unknown constraint '{name}'. Available: {sorted(BOUND_REGISTRY)}")

    meta = dict(BOUND_REGISTRY[name])
    path = os.path.join(_BOUNDS_DIR, meta["file"])
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Bound data file not found: {path}  "
            f"(entry '{name}' references '{meta['file']}')"
        )

    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Expected two-column data in {path}, got shape {data.shape}"
        )

    M: np.ndarray = data[:, 0]
    f_pbh: np.ndarray = data[:, 1]
    return M, f_pbh, meta


# ---------------------------------------------------------------------------
# Epoch mapping (De Luca+20, Eq. 17)
# ---------------------------------------------------------------------------


def _invert_mass(M_z: float, accretion: Any,
                 z_probe: float, z_form: float = 3400.0) -> float:
    """Find *M_form* such that ``accretion.M_of_redshift(M_form, z_probe) = M_z``.

    Uses Brent's method on the monotonic function
    ``g(M) = M_of_redshift(M, z_probe) - M_z``.
    """
    from scipy.optimize import root_scalar

    def _objective(M_form: float) -> float:
        return accretion.M_of_redshift(M_form, z_probe, z_form) - M_z

    # Short-circuit: if the accretion model doesn't change mass up to z_probe
    # (identity mapping), then M_form = M_z is exact.
    test_grown = accretion.M_of_redshift(M_z, z_probe, z_form)
    if abs(test_grown - M_z) / max(M_z, 1e-300) < 1e-12:
        return M_z

    # Initial bracket: M_form must be ≤ M_z (accretion only adds mass).
    lo = max(M_z * 1e-12, 1e-300)
    hi = M_z * 1.0

    # If g(hi) > 0, the root is already bracketed between lo and hi.
    # If g(hi) < 0, we need to widen hi (unusual — means accretion reduces mass).
    g_lo = _objective(lo)
    g_hi = _objective(hi)

    while g_lo * g_hi > 0 and hi < 1e30 * M_z:
        hi *= 10.0
        g_hi = _objective(hi)

    if g_lo * g_hi >= 0:
        warnings.warn(
            f"_invert_mass: cannot bracket root for M_z={M_z:.4e}, "
            f"accretion={type(accretion).__name__}, z_probe={z_probe}. "
            f"g(lo)={g_lo:.4e}, g(hi)={g_hi:.4e}. Falling back to M_z.",
            stacklevel=2,
        )
        return M_z

    try:
        sol = root_scalar(_objective, bracket=(lo, hi), method="brentq",
                          xtol=1e-12, rtol=1e-6)
    except ValueError as exc:
        warnings.warn(
            f"_invert_mass: root_scalar failed for M_z={M_z:.4e}: {exc}. "
            f"Falling back to M_z.",
            stacklevel=2,
        )
        return M_z

    if not sol.converged:
        warnings.warn(
            f"_invert_mass: root_scalar did not converge for M_z={M_z:.4e}. "
            f"Falling back to M_z.",
            stacklevel=2,
        )
        return M_z
    return sol.root


def map_constraint_to_present(
    M_z: np.ndarray, f_pbh_z: np.ndarray,
    accretion: Any, z_probe: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a constraint from *z_probe* to present-day *(M, f)* space.

    Implements the number-conservation mapping from De Luca, Franciolini &
    Riotto (2020), :arxiv:`2003.12589` Eq. 17:

    .. math::

        f_\\text{PBH}^\\text{present}(M_\\text{present}) =
        f_\\text{PBH}^z(M_z) \\, \\frac{M_z}{M_\\text{present}}

    Parameters
    ----------
    M_z : ndarray
        Masses at the probe redshift [:math:`M_\\odot`].
    f_pbh_z : ndarray
        :math:`f_\\text{PBH}` upper limits at *z_probe*.
    accretion : PBHAccretion or ChisholmAccretion instance
        Accretion model providing ``M_of_redshift(M_form, z)``.
    z_probe : float
        Redshift at which the constraint was derived.  Must be ≥ 0.

    Returns
    -------
    M_present : ndarray
        Mapped masses at :math:`z = 0` [:math:`M_\\odot`].
    f_pbh_present : ndarray
        Mapped :math:`f_\\text{PBH}` limits.
    """
    if z_probe < 0:
        raise ValueError(f"z_probe must be ≥ 0, got {z_probe}")

    if z_probe == 0:
        return M_z.copy(), f_pbh_z.copy()

    M_z_arr = np.asarray(M_z, dtype=np.float64)
    f_z_arr = np.asarray(f_pbh_z, dtype=np.float64)

    M_present = np.empty_like(M_z_arr)
    for i in range(len(M_z_arr)):
        M_form = _invert_mass(M_z_arr[i], accretion, z_probe)
        M_present[i] = accretion.M_of_redshift(M_form, 0.0)

    # De Luca+20 number-conservation mapping (Eq. 17)
    f_pbh_present = f_z_arr * M_z_arr / M_present

    return M_present, f_pbh_present


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_modern_constraints(
    ax: Any, accretion_model: Any | None = None,
    show_set: set[str] | None = None, style: str = "modern",
) -> list[tuple[str, Any]]:
    """Plot modern PBH observational constraints with optional epoch mapping.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        The axes to draw on.
    accretion_model : PBHAccretion, ChisholmAccretion, or None
        If *None*, constraints are plotted at their probe redshift (identity
        mapping).  If given, high-redshift constraints are epoch-mapped to
        present-day masses via :func:`map_constraint_to_present`.
    show_set : set of str or None
        Subset of constraint names from :py:data:`BOUND_REGISTRY` to include.
        *None* means all.
    style : {``'modern'``, ``'legacy'``}
        *modern* — PR-era colors; *legacy* — pre-2020 greyscale style.

    Returns
    -------
    list of (name, line_handle)
        Handles for legend construction.

    Raises
    ------
    TypeError
        If *ax* is *None*.
    """
    if ax is None:
        raise TypeError("plot_modern_constraints requires a valid Axes (got None)")

    # Attempt to use Paul Tol colours from scripts.plotting.TOL
    try:
        from scripts.plotting import TOL as tol_colors  # type: ignore[import-untyped]
    except Exception:  # noqa: BLE001
        tol_colors = {}  # fallback – use registry colours

    names = sorted(
        show_set & BOUND_REGISTRY.keys() if show_set else BOUND_REGISTRY
    )
    handles: list[tuple[str, Any]] = []

    for name in names:
        M, f, meta = load_bound_dataset(name)
        z_probe = meta["z_probe"]
        color = meta["color"]

        # Apply epoch mapping
        if accretion_model is not None and z_probe > 0:
            M_plot, f_plot = map_constraint_to_present(
                M, f, accretion_model, z_probe,
            )
        else:
            M_plot, f_plot = M.copy(), f.copy()

        # Sort by mass for clean fill
        idx = np.argsort(M_plot)
        M_plot = M_plot[idx]
        f_plot = f_plot[idx]

        # Remove non-finite points
        ok = np.isfinite(M_plot) & np.isfinite(f_plot) & (M_plot > 0) & (f_plot > 0)
        M_plot = M_plot[ok]
        f_plot = f_plot[ok]
        if len(M_plot) < 2:
            continue

        # Use TOL colour if available, else registry colour
        plot_color = tol_colors.get(meta.get("tol_key", ""), color)

        if style == "legacy":
            ax.fill_between(M_plot, f_plot, 10.0,
                            color=plot_color, alpha=0.05)
            (line,) = ax.loglog(M_plot, f_plot, "--", color=plot_color,
                                lw=0.6, label=meta["label"])
        else:
            ax.fill_between(M_plot, f_plot, 10.0,
                            color=plot_color, alpha=0.08)
            (line,) = ax.loglog(M_plot, f_plot, "-", color=plot_color,
                                lw=0.8, label=meta["label"])

        handles.append((name, line))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-19, 1e8)
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel(r"$M\ [M_\odot]$")
    ax.set_ylabel(r"$f_\mathrm{PBH} = \Omega_\mathrm{PBH}/\Omega_\mathrm{DM}$")

    return handles
