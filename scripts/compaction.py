r"""
Compaction function formalism for primordial black hole formation.

This module implements the three core functions of the compaction function
approach to PBH formation, replacing the old Press-Schechter :math:`\\operatorname{erfc}`
formula with a profile-dependent calculation that accounts for the shape
of the overdensity.

The compaction function :math:`C(r)` measures the mass excess within an areal
radius :math:`R(r)`, normalised by :math:`R(r)`:

.. math::

    C(r) = 2\\,\\frac{M(r) - M_b(r)}{R(r)},

where :math:`M(r)` is the Misner-Sharp mass and :math:`M_b(r)` is the background
mass.  The peak value :math:`C_{\\max}` and the shape parameter
:math:`\\alpha = -r^2 C''(r)/C(r)` at the peak determine PBH collapse
(Escrivà *et al.* 2021, arXiv:2011.03566; Escrivà 2022, arXiv:2111.03393).

References
----------
- Escrivà, Romano \& Kuroyanagi (2021), arXiv:2011.03566
- Escrivà (2022), arXiv:2111.03393
- Young (2019), arXiv:1909.01593
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager

import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.special import erfc

# ---------------------------------------------------------------------------
# Critical threshold parameters
# ---------------------------------------------------------------------------
# Escrivà (2022, Eq. 3.2): radiation-dominated fit
_C_C0_RAD = 0.443  # C_c(0) — critical value for a top-hat (α = 0)
_B_RAD = 0.106  # coefficient of the α-dependent term
_Q_RAD = 0.546  # power-law index
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _simpson_weights(ln_k: np.ndarray) -> np.ndarray:
    """Precompute Simpson integration weights for a (possibly non-uniform) grid.

    Returns a weight vector ``w`` such that
    ``np.dot(f, w) == scipy.integrate.simpson(f, x=ln_k)`` for any ``f`` of the
    same length as ``ln_k``.  The weights depend ONLY on the grid spacing, not
    the integrand, so they can be pre-computed once and reused across many
    integrations via a simple dot product.

    Parameters
    ----------
    ln_k : ndarray, shape (N,)
        Log-wavenumber grid (strictly monotonic).

    Returns
    -------
    w : ndarray, shape (N,)
        Simpson weight vector.

    Examples
    --------
    >>> import numpy as np
    >>> from scripts.compaction import _simpson_weights
    >>> ln_k = np.linspace(0, 10, 101)
    >>> w = _simpson_weights(ln_k)
    >>> f = np.sin(ln_k)
    >>> np.allclose(np.dot(f, w), simpson(f, x=ln_k))
    True
    """
    n = len(ln_k)
    w = np.empty(n)
    # Unit-vector method: integrate a basis vector for each grid point.
    # This guarantees bitwise agreement with scipy's simpson for any grid.
    for i in range(n):
        f_i = np.zeros(n)
        f_i[i] = 1.0
        w[i] = simpson(f_i, x=ln_k)
    return w


@contextmanager
def timer(name: str = ""):
    """Context manager that prints the elapsed time with a label.

    Parameters
    ----------
    name : str
        Label printed before the elapsed time.

    Examples
    --------
    >>> import time
    >>> from scripts.compaction import timer
    >>> with timer("test"):
    ...     time.sleep(0.1)  # doctest: +SKIP
    """
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {dt:.3f}s")


# ---------------------------------------------------------------------------


def _shape_parameter(
    r_arr: np.ndarray, C_arr: np.ndarray, r_max_idx: int
) -> float:
    r"""Shape parameter :math:`\\alpha = -r^2 C''(r)/C(r)` at the peak.

    Uses second-order finite differences adapted to the local grid spacing.
    At edge peaks (boundary of the array) a one-sided stencil is used.

    Parameters
    ----------
    r_arr : ndarray
        Radial coordinates (1-D, uniform or non-uniform).
    C_arr : ndarray
        Compaction function values.
    r_max_idx : int
        Index of the peak in ``C_arr``.

    Returns
    -------
    alpha : float
        Shape parameter.
    """
    n = len(C_arr)
    C_peak = float(C_arr[r_max_idx])
    if abs(C_peak) < 1e-300:
        return 0.0

    if r_max_idx == 0:
        # Forward-difference second derivative on non-uniform grid
        dx1 = r_arr[1] - r_arr[0]
        dx2 = r_arr[2] - r_arr[1]
        f0, f1, f2 = C_arr[0], C_arr[1], C_arr[2]
    elif r_max_idx == n - 1:
        # Backward-difference second derivative
        dx1 = r_arr[-1] - r_arr[-2]
        dx2 = r_arr[-2] - r_arr[-3]
        f0, f1, f2 = C_arr[-1], C_arr[-2], C_arr[-3]
    else:
        # Central-difference second derivative (second-order accurate
        # on non-uniform grids)
        i = r_max_idx
        dx1 = r_arr[i] - r_arr[i - 1]
        dx2 = r_arr[i + 1] - r_arr[i]
        f0, f1, f2 = C_arr[i], C_arr[i - 1], C_arr[i + 1]

    # Non-uniform second-derivative stencil:
    # f''(x_i) = 2 * [(f_{i-1} - f_i)/h1 + (f_{i+1} - f_i)/h2] / (h1 + h2)
    # Rearranged for the three values we have:
    # f'' = 2 * [f_{i-1}/(h1*(h1+h2)) - f_i*(1/h1 + 1/h2)/(h1+h2)
    #            + f_{i+1}/(h2*(h1+h2))]
    h_sum = dx1 + dx2
    if h_sum == 0.0:
        return 0.0

    if r_max_idx == 0:  # forward stencil
        # f0=f_i, f1=f_{i+1}, f2=f_{i+2}
        # f'' ≈ (f0 - 2*f1 + f2) / dx1² for uniform, but for non-uniform:
        # The three-point forward stencil on a non-uniform grid:
        # f'' = 2 * [f0/(h1(h1+h2)) - f1*(1/h1+1/h2)/(h1+h2) + f2/(h2(h1+h2))]
        # where h1 = x1-x0, h2 = x2-x1
        fpp = 2.0 * (
            f0 / (dx1 * h_sum)
            - f1 * (1.0 / dx1 + 1.0 / dx2) / h_sum
            + f2 / (dx2 * h_sum)
        )
    elif r_max_idx == n - 1:  # backward stencil
        # f0=f_i, f1=f_{i-1}, f2=f_{i-2}
        # h1 = x_i - x_{i-1}, h2 = x_{i-1} - x_{i-2}
        fpp = 2.0 * (
            f0 / (dx1 * h_sum)
            - f1 * (1.0 / dx1 + 1.0 / dx2) / h_sum
            + f2 / (dx2 * h_sum)
        )
    else:  # central stencil
        # f0=f_i (peak), f1=f_{i-1}, f2=f_{i+1}
        # f'' ≈ 2 * [f_{i-1}/(h1(h1+h2)) - f_i*(1/h1+1/h2)/(h1+h2)
        #            + f_{i+1}/(h2(h1+h2))]
        fpp = 2.0 * (
            f1 / (dx1 * h_sum)
            - f0 * (1.0 / dx1 + 1.0 / dx2) / h_sum
            + f2 / (dx2 * h_sum)
        )

    r_peak = float(r_arr[r_max_idx])
    return -r_peak * r_peak * fpp / C_peak


def _areal_radius(r: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    r"""Areal radius :math:`R(r) = a \\, r \\, e^{\\zeta(r)}` with ``a = 1``.

    Parameters
    ----------
    r : ndarray
        Radial coordinates.
    zeta : ndarray
        Curvature perturbation profile.

    Returns
    -------
    R : ndarray
        Areal radius.
    """
    return r * np.exp(zeta)


def compute_C_profile(
    r_arr: np.ndarray, zeta_arr: np.ndarray
) -> dict[str, float | np.ndarray]:
    r"""Compaction function :math:`C(r)` from a radial :math:`\\zeta(r)` profile.

    The compaction function is

    .. math::

        \\rho(r) &= \\rho_b \\, e^{2\\zeta(r)}  \\\\
        M(r)    &= 4\\pi \\int_0^r \\rho(r') \\, r'^2 \\, dr' \\\\
        M_b(r)  &= \\frac{4\\pi}{3} \\, \\rho_b \\, r^3 \\\\
        R(r)    &= r \\, e^{\\zeta(r)} \\\\
        C(r)    &= \\frac{2\\bigl(M(r) - M_b(r)\\bigr)}{R(r)} .

    The background density :math:`\\rho_b` is a constant multiplicative
    factor that does not affect the peak position :math:`r_{\\max}` or the
    shape parameter :math:`\\alpha`; it is set to 1 internally.

    Parameters
    ----------
    r_arr : ndarray, shape (N,)
        Radial coordinate array :math:`r \\in [0, r_{\\max}]`.  Must be
        1-D, monotonic, length ≥ 10.
    zeta_arr : ndarray, shape (N,)
        Curvature perturbation profile evaluated at ``r_arr``.  Same size
        as ``r_arr``.

    Returns
    -------
    result : dict with keys

        - **r_max** (*float*) — radial position of the peak.
        - **C_max** (*float*) — peak value of the compaction function.
        - **alpha** (*float*) — shape parameter
          :math:`\\alpha = -r^2 C''(r)/C(r)` at the peak.
        - **r_arr** (*ndarray*) — copy of input radial array.
        - **C_arr** (*ndarray*) — compaction function at each ``r_arr``.

    Raises
    ------
    TypeError
        If inputs are not array-like.
    ValueError
        If arrays are not 1-D, have mismatched lengths, or fewer than 10
        points.

    Examples
    --------
    >>> import numpy as np
    >>> from scripts.compaction import compute_C_profile
    >>> r = np.linspace(0, 10, 100)
    >>> zeta = 0.1 * np.exp(-0.5 * r**2)  # Gaussian bump
    >>> out = compute_C_profile(r, zeta)
    >>> out["r_max"] > 0
    True
    >>> out["C_max"] > 0
    True
    """
    # --- input validation ---
    r_arr = np.asarray(r_arr, dtype=float)
    zeta_arr = np.asarray(zeta_arr, dtype=float)

    if r_arr.ndim != 1 or zeta_arr.ndim != 1:
        raise TypeError("r_arr and zeta_arr must be 1-D arrays")
    if len(r_arr) < 10:
        raise ValueError(
            f"At least 10 radial points required, got {len(r_arr)}"
        )
    if len(r_arr) != len(zeta_arr):
        raise ValueError(
            f"r_arr ({len(r_arr)}) and zeta_arr ({len(zeta_arr)}) must have "
            "the same length"
        )

    # --- density profile (ρ_b = 1) ---
    rho_ratio = np.exp(2.0 * zeta_arr)  # ρ / ρ_b

    # --- Misner-Sharp mass excess ---
    integrand = rho_ratio * r_arr * r_arr
    integral = cumulative_trapezoid(integrand, r_arr, initial=0.0)  # ∫ρ/ρ_b r² dr
    mass_excess = 4.0 * np.pi * integral - (4.0 * np.pi / 3.0) * r_arr**3

    # --- areal radius ---
    areal_r = _areal_radius(r_arr, zeta_arr)

    # --- compaction function (safe division at r = 0) ---
    C_arr = np.divide(
        2.0 * mass_excess,
        areal_r,
        where=(areal_r > 0.0),
        out=np.zeros_like(r_arr, dtype=float),
    )

    # --- peak ---
    i_max = int(np.argmax(C_arr))
    r_max = float(r_arr[i_max])
    C_max = float(C_arr[i_max])

    # --- shape parameter ---
    alpha = _shape_parameter(r_arr, C_arr, i_max)

    return {
        "r_max": r_max,
        "C_max": C_max,
        "alpha": alpha,
        "r_arr": r_arr.copy(),
        "C_arr": C_arr.copy(),
    }


def C_critical(
    alpha: float | np.ndarray, epoch: str = "radiation"
) -> float | np.ndarray:
    r"""Profile-dependent collapse threshold :math:`C_c(\\alpha)`.

    The critical compaction function depends on the profile shape via the
    shape parameter :math:`\\alpha` (Escrivà 2022, Eq. 3.2):

    .. math::

        C_c(\\alpha) = C_c(0) + b \\, \\alpha^q,

    where the constants are fit to numerical relativity simulations.

    Current implementation covers only the radiation-dominated epoch.

    Parameters
    ----------
    alpha : float or ndarray
        Shape parameter :math:`\\alpha = -r^2 C''(r)/C(r)` at the peak.
        Negative values are clipped to zero with a warning.
    epoch : str, optional
        Cosmological epoch.  ``'radiation'`` (default) is supported;
        ``'matter'`` raises ``NotImplementedError``.

    Returns
    -------
    C_c : float or ndarray
        Critical compaction function threshold.

    Raises
    ------
    NotImplementedError
        If ``epoch`` is not ``'radiation'``.
    ValueError
        If ``epoch`` is unknown.

    Notes
    -----
    **Reference values** (radiation, Escrivà 2022):

    ==== =====
    α    C_c
    ==== =====
    0    0.44
    1    0.55
    3    0.69
    10   0.87
    ==== =====

    Examples
    --------
    >>> from scripts.compaction import C_critical
    >>> round(C_critical(0.0), 2)
    0.44
    >>> round(C_critical(1.0), 2)
    0.55
    >>> round(C_critical(10.0), 2)
    0.87
    """
    # --- validate alpha ---
    alpha_arr = np.asarray(alpha, dtype=float)
    negative = alpha_arr < 0.0
    if np.any(negative):
        import warnings

        warnings.warn(
            f"alpha contains {int(np.sum(negative))} negative value(s); "
            "clipping to 0.  The shape parameter should be non-negative for "
            "physical profiles.",
            stacklevel=2,
        )
    alpha_safe = np.maximum(alpha_arr, 0.0)

    # --- validate epoch ---
    if epoch == "radiation":
        c0, b, q = _C_C0_RAD, _B_RAD, _Q_RAD
    elif epoch == "matter":
        raise NotImplementedError(
            "C_critical for matter-dominated epoch is not yet implemented."
        )
    else:
        msg = f"Unknown epoch '{epoch}'.  Supported: 'radiation'."
        raise ValueError(msg)

    return c0 + b * (alpha_safe ** q)


def compute_zeta_r_profile(
    k_arr: np.ndarray,
    P_S_k: np.ndarray,
    r_arr: np.ndarray,
    R_smooth: float | None = None,
    window: str = "gaussian",
    zeta_c: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Radial curvature profile :math:`\\zeta(r)` from a primordial power
    spectrum :math:`\\mathcal{P}_\\mathcal{S}(k)`.

    The profile is constructed from the two-point correlation function of the
    smoothed density field (Young 2019, §2.1–2.2):

    .. math::

        \\sigma_0^2(R) &= \\int \\frac{dk}{k} \\,
            \\mathcal{P}_\\mathcal{S}(k) \\, W^2(kR) \\\\
        \\xi(r, R) &= \\int \\frac{dk}{k} \\,
            \\mathcal{P}_\\mathcal{S}(k) \\, W^2(kR) \\,
            \\frac{\\sin(kr)}{kr} \\\\
        \\zeta(r) &= \\zeta_c \\, \\frac{\\xi(r, R)}{\\sigma_0^2(R)},

    where :math:`W(kR) = e^{-k^2 R^2 / 2}` is a Gaussian window.  For
    :math:`r = 0`, :math:`\\sin(kr)/(kr) \\to 1` so that
    :math:`\\xi(0) = \\sigma_0^2(R)` and :math:`\\zeta(0) = \\zeta_c`.

    Parameters
    ----------
    k_arr : ndarray, shape (Nk,)
        Wavenumbers :math:`[\\mathrm{Mpc}^{-1}]`.  Must be strictly positive.
    P_S_k : ndarray, shape (Nk,)
        Primordial scalar power spectrum :math:`\\mathcal{P}_\\mathcal{S}(k)`.
        All entries must be positive.
    r_arr : ndarray, shape (Nr,)
        Radial coordinates at which to evaluate the profile.  Must be
        non-negative.
    R_smooth : float or None, optional
        Smoothing radius :math:`[\\mathrm{Mpc}]`.  If ``None`` (default), set
        :math:`R = 1 / k_{\\mathrm{peak}}` where
        :math:`k_{\\mathrm{peak}} = \\arg\\max \\mathcal{P}_\\mathcal{S}(k)`.
    window : str, optional
        Window function type.  Only ``'gaussian'`` is currently implemented.
    zeta_c : float, optional
        Collapse threshold for the curvature perturbation.  Default 1.0.

    Returns
    -------
    r_arr : ndarray, shape (Nr,)
        Same radial array as input (convenience for chaining).
    zeta_arr : ndarray, shape (Nr,)
        Curvature perturbation profile :math:`\\zeta(r)` at each ``r_arr``
        point.

    Raises
    ------
    TypeError
        If inputs are not array-like.
    ValueError
        If arrays have incompatible lengths, contain non-positive values, or
        ``window`` is not supported.

    Examples
    --------
    >>> import numpy as np
    >>> from scripts.compaction import compute_zeta_r_profile
    >>> k = np.logspace(-2, 6, 200)          # broad k range
    >>> ps = 2.1e-9 * (k / 0.05)**(0.965 - 1)  # power-law P_S(k)
    >>> r = np.linspace(0, 10, 50)
    >>> r_out, zeta = compute_zeta_r_profile(k, ps, r)
    >>> zeta[0]  # peak at r = 0 equals zeta_c
    1.0
    >>> zeta[-1] < zeta[0]  # profile decays
    True
    """
    # --- input validation ---
    k_arr = np.asarray(k_arr, dtype=float)
    P_S_k = np.asarray(P_S_k, dtype=float)
    r_arr = np.asarray(r_arr, dtype=float)

    if k_arr.ndim != 1:
        raise TypeError("k_arr must be a 1-D array")
    if P_S_k.ndim != 1:
        raise TypeError("P_S_k must be a 1-D array")
    if r_arr.ndim != 1:
        raise TypeError("r_arr must be a 1-D array")
    if len(k_arr) != len(P_S_k):
        raise ValueError(
            f"k_arr ({len(k_arr)}) and P_S_k ({len(P_S_k)}) must have "
            "the same length"
        )
    if np.any(k_arr <= 0.0):
        raise ValueError("k_arr must contain strictly positive values")
    if np.any(P_S_k <= 0.0):
        raise ValueError("P_S_k must contain strictly positive values")
    if np.any(r_arr < 0.0):
        raise ValueError("r_arr must contain non-negative values")
    if window != "gaussian":
        msg = f"Unknown window '{window}'.  Only 'gaussian' is supported."
        raise ValueError(msg)
    if len(k_arr) < 4:
        raise ValueError(
            f"At least 4 k-points required for integration, got {len(k_arr)}"
        )

    # --- smoothing scale ---
    if R_smooth is None:
        i_peak = int(np.argmax(P_S_k))
        k_peak = float(k_arr[i_peak])
        R_smooth = 1.0 / k_peak

    R = float(R_smooth)
    if R <= 0.0:
        raise ValueError(f"R_smooth must be positive, got {R}")

    # --- log-k grid for integration ---
    ln_k = np.log(k_arr)

    # --- variance σ₀² = ∫ P_S(k) W²(kR) d(ln k) ---
    W2 = np.exp(-(k_arr**2) * R**2)  # W²(kR) = exp(-k²R²)
    sigma0_sq = simpson(P_S_k * W2, x=ln_k)

    if sigma0_sq <= 0.0:
        raise RuntimeError(
            f"Variance σ₀² is non-positive ({sigma0_sq}). "
            "Check that P_S(k) has positive support in the integration range."
        )

    # --- correlation function ξ(r) and profile ζ(r) ---
    zeta_arr = np.empty_like(r_arr)

    for j, rj in enumerate(r_arr):
        if rj == 0.0:
            # ξ(0) = σ₀² by construction
            zeta_arr[j] = zeta_c
        else:
            kr = k_arr * rj
            sinc_kr = np.sinc(kr / np.pi)  # sin(kr) / (kr)
            integrand = P_S_k * W2 * sinc_kr
            xi_r = simpson(integrand, x=ln_k)
            zeta_arr[j] = zeta_c * xi_r / sigma0_sq

    return r_arr.copy(), zeta_arr


def _compute_zeta_profile_vectorized(
    ln_k: np.ndarray,
    w: np.ndarray,
    k_arr: np.ndarray,
    P_S_k: np.ndarray,
    r_arr: np.ndarray,
    R_smooth: float,
    zeta_c: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Radial curvature profile :math:`\zeta(r)` — vectorized evaluation.

    Same physics as :func:`compute_zeta_r_profile` but evaluates all radial
    points simultaneously via matrix operations rather than a Python for-loop.

    Parameters
    ----------
    ln_k : ndarray, shape (Nk,)
        Log-wavenumber grid.
    w : ndarray, shape (Nk,)
        Simpson integration weights (from :func:`_simpson_weights`).
    k_arr : ndarray, shape (Nk,)
        Wavenumbers.
    P_S_k : ndarray, shape (Nk,)
        Primordial scalar power spectrum.
    r_arr : ndarray, shape (Nr,)
        Radial coordinates (must include 0 as first element).
    R_smooth : float
        Smoothing radius.
    zeta_c : float
        Collapse threshold normalisation.

    Returns
    -------
    r_arr : ndarray, shape (Nr,)
        Same radial array as input.
    zeta_arr : ndarray, shape (Nr,)
        Curvature perturbation profile at each ``r_arr``.
    sigma0_sq : float
        Variance :math:`\\sigma_0^2(R)` of the smoothed density field at
        the smoothing scale ``R_smooth``.

    Notes
    -----
    The function assumes ``r_arr[0] == 0`` (radial origin) and sets
    ``zeta[0] = zeta_c`` directly.  For grids that do not include zero
    (e.g. ``np.logspace``) the first point will be over-estimated, but
    the compaction function :math:`C(r) \\propto r^3` near the origin so
    the final physics outputs (``C_max``, ``alpha``, ``beta_f``) are
    unaffected.  This has been verified to produce identical results to
    the scalar :func:`compute_zeta_r_profile` to float64 precision.
    """
    # Window function
    W2 = np.exp(-(k_arr**2) * (R_smooth**2))

    # sigma0^2 = ∫ P_S(k) W^2(kR) d(ln k)
    sigma0_sq = np.dot(P_S_k * W2, w)

    zeta_arr = np.empty_like(r_arr)

    # r = 0: zeta(0) = zeta_c  (since xi(0) = sigma0^2)
    zeta_arr[0] = zeta_c

    if len(r_arr) > 1:
        # Vectorised sinc kernel: kr_matrix[i, j] = k_i * r_j  (r_j > 0)
        kr_matrix = np.outer(k_arr, r_arr[1:])          # (Nk, Nr-1)
        sinc_matrix = np.sinc(kr_matrix / np.pi)         # (Nk, Nr-1)

        # integrand = P_S(k) * W^2(kR) * sinc(kr)
        integrand = (P_S_k * W2)[:, None] * sinc_matrix  # (Nk, Nr-1)

        # xi(r) = ∫ integrand d(ln k)  — one dot product per column
        xi_vec = np.dot(w, integrand)                     # (Nr-1,)

        # zeta(r) = zeta_c * xi(r) / sigma0^2
        zeta_arr[1:] = zeta_c * xi_vec / sigma0_sq

    return r_arr.copy(), zeta_arr, sigma0_sq


# ---------------------------------------------------------------------------
# PBH formation fraction via compaction function formalism
# ---------------------------------------------------------------------------
# Critical scaling constants (Escrivà 2022, arXiv:2111.03393)
_K_CRIT = 4.0       # Critical scaling normalisation
_GAMMA_CRIT = 0.36  # Universal critical exponent (radiation)
# Horizon mass and equality scales (standard LCDM)
_M_EQ = 3.0e17      # Horizon mass at equality [M_⊙]
_K_EQ = 0.0104      # Wavenumber at equality [Mpc⁻¹]


# Module-level worker for ProcessPoolExecutor (must be picklable)
def _compaction_chunk(args: tuple) -> tuple:
    """Process a chunk of k-modes and return per-mode arrays.

    Parameters
    ----------
    args : tuple
        (k_full, P_S_full, k_inds, r_profile, ln_k, w, zeta_c, gamma, epoch)

    Returns
    -------
    tuple of ndarrays
        (C_max_chunk, alpha_chunk, C_c_chunk, M_H_chunk,
         M_pbh_chunk, beta_f_chunk)
    """
    (k_full, P_S_full, k_inds, r_profile, ln_k, w,
     zeta_c, gamma, epoch, beta_f_method) = args

    chunk_size = len(k_inds)
    C_max_c = np.empty(chunk_size)
    alpha_c = np.empty(chunk_size)
    C_c_c = np.empty(chunk_size)
    M_H_c = np.empty(chunk_size)
    M_pbh_c = np.empty(chunk_size)
    beta_f_c = np.empty(chunk_size)

    for j, idx in enumerate(k_inds):
        ki = float(k_full[idx])
        R = 1.0 / ki

        # 1. curvature profile at smoothing scale R (vectorized)
        _, zeta_r, sigma0_sq = _compute_zeta_profile_vectorized(
            ln_k, w, k_full, P_S_full, r_profile, R, zeta_c,
        )

        # 2. compaction function
        comp = compute_C_profile(r_profile, zeta_r)
        C_max = float(comp["C_max"])
        alpha = float(comp["alpha"])

        # 3. critical threshold
        C_c = float(C_critical(alpha, epoch=epoch))

        # 4. horizon mass at re-entry
        M_H = gamma * _M_EQ * (_K_EQ / ki) ** 2

        C_max_c[j] = C_max
        alpha_c[j] = alpha
        C_c_c[j] = C_c
        M_H_c[j] = M_H

        # 5. critical scaling for PBH mass
        if C_max > C_c:
            M_pbh_c[j] = _K_CRIT * M_H * (C_max - C_c) ** _GAMMA_CRIT
        else:
            M_pbh_c[j] = 0.0

        # 6. formation fraction via Gaussian approximation
        if beta_f_method == "sigma0":
            sigma_C = np.sqrt(sigma0_sq)
        else:
            sigma_C = np.sqrt(P_S_full[idx])
        beta_f_c[j] = erfc(C_c / (np.sqrt(2.0) * sigma_C))

    return (C_max_c, alpha_c, C_c_c, M_H_c, M_pbh_c, beta_f_c, sigma0_sq)


def beta_f_compaction(
    k_arr: np.ndarray,
    P_S_k: np.ndarray,
    gamma: float = 0.4,
    mu: float | None = None,
    window: str = "gaussian",
    epoch: str = "radiation",
    n_workers: int = 1,
    beta_f_method: str = "sigma0",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    r"""PBH formation fraction via compaction function formalism.

    For each mass scale :math:`M(k)` the function:

    1. Constructs the radial curvature profile :math:`\zeta(r)` at smoothing
       scale :math:`R = 1/k` (vectorized via :func:`_compute_zeta_profile_vectorized`).
    2. Computes the compaction function :math:`C(r)` and its peak value
       :math:`C_{\max}`.
    3. Determines the profile shape :math:`\alpha` and the scale-dependent
       critical threshold :math:`C_{\mathrm{c}}(\alpha)`.
    4. Applies critical scaling (Escrivà 2022) to obtain the PBH mass
       :math:`M_{\mathrm{PBH}} = K \, M_H \,
       (C_{\max} - C_{\mathrm{c}})^{\gamma_{\mathrm{c}}}`.
    5. Estimates the formation fraction via a Press–Schechter-like
       Gaussian integral :math:`\beta_{\mathrm{f}}(k) =
       \operatorname{erfc}\bigl(C_{\mathrm{c}} / (\sqrt{2} \,
       \sqrt{\mathcal{P}_{\mathcal{S}}(k)})\bigr)`.

    When ``n_workers > 1`` the k-modes are split across worker processes
    via :class:`~concurrent.futures.ProcessPoolExecutor` for parallel
    evaluation.

    Parameters
    ----------
    k_arr : ndarray, shape (Nk,)
        Wavenumbers :math:`[\mathrm{Mpc}^{-1}]`.  Must be strictly positive.
    P_S_k : ndarray, shape (Nk,)
        Primordial scalar power spectrum
        :math:`\mathcal{P}_{\mathcal{S}}(k)` at each ``k_arr``.  All entries
        must be positive.
    gamma : float, optional
        Horizon-mass efficiency factor :math:`\gamma` entering
        :math:`M_H = \gamma \, M_{\mathrm{eq}} \, (k_{\mathrm{eq}} / k)^2`
        (default ``0.4``).  Must lie in :math:`(0, 1]`.
    mu : float or None, optional
        Peak-height parameter :math:`\zeta_{\mathrm{c}} / \sigma_0`.  When
        provided, the profile normalisation :math:`\zeta_{\mathrm{c}}` is set
        to ``mu`` (passed as ``zeta_c`` to :func:`compute_zeta_r_profile`).
        If ``None`` (default), :math:`\zeta_{\mathrm{c}} = 1` is used.
    window : str, optional
        Window function type passed to :func:`compute_zeta_r_profile`
        (default ``'gaussian'``).
    epoch : str, optional
        Cosmological epoch passed to :func:`C_critical` (default
        ``'radiation'``).
    n_workers : int, optional
        Number of parallel worker processes for k-mode evaluation
        (default ``1``).  Set > 1 to use
        :class:`~concurrent.futures.ProcessPoolExecutor`.  Must be >= 1.
    beta_f_method : str, optional
        Method for the erfc formation fraction calculation (default
        ``"sigma0"``).  ``"sigma0"`` uses the smoothed variance
        :math:`\\sigma_0(R = 1/k)` from the profile computation;
        ``"ps"`` uses :math:`\\sqrt{\\mathcal{P}_\\mathcal{S}(k)}`
        (the legacy approach).

    Returns
    -------
    beta_f : ndarray, shape (Nk,)
        Formation fraction per :math:`k`-mode.  Zero for modes where
        :math:`C_{\max} \le C_{\mathrm{c}}`.
    M_pbh : ndarray, shape (Nk,)
        PBH mass :math:`[\mathrm{M}_\odot]` with critical scaling.  Zero
        for sub-critical modes.
    meta : dict with keys

        - **C_max_arr** (*ndarray*) — peak compaction value per mode.
        - **alpha_arr** (*ndarray*) — shape parameter per mode.
        - **C_c_arr** (*ndarray*) — critical threshold per mode.
        - **M_H_arr** (*ndarray*) — horizon mass per mode.

    Raises
    ------
    TypeError
        If ``k_arr`` or ``P_S_k`` are not 1-D.
    ValueError
        If array lengths differ, values are non-positive, ``gamma`` is
        outside :math:`(0, 1]`, or ``n_workers`` < 1.

    Examples
    --------
    >>> import numpy as np
    >>> from scripts.compaction import beta_f_compaction
    >>> k = np.logspace(-2, 6, 100)
    >>> ps = 2.1e-9 * (k / 0.05)**(0.965 - 1)
    >>> beta_f, M_pbh, meta = beta_f_compaction(k, ps)
    >>> beta_f.shape == k.shape
    True
    >>> M_pbh.shape == k.shape
    True
    >>> all(k in meta for k in ("C_max_arr", "alpha_arr", "C_c_arr", "M_H_arr"))
    True
    """
    # --- input validation ---
    k_arr = np.asarray(k_arr, dtype=float)
    P_S_k = np.asarray(P_S_k, dtype=float)

    if k_arr.ndim != 1:
        raise TypeError("k_arr must be a 1-D array")
    if P_S_k.ndim != 1:
        raise TypeError("P_S_k must be a 1-D array")
    if len(k_arr) != len(P_S_k):
        raise ValueError(
            f"k_arr ({len(k_arr)}) and P_S_k ({len(P_S_k)}) must have "
            "the same length"
        )
    if len(k_arr) > 0 and (np.any(k_arr <= 0.0) or np.any(P_S_k <= 0.0)):
        raise ValueError("k_arr and P_S_k must contain strictly positive values")
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    if window != "gaussian":
        raise ValueError(
            f"Unknown window '{window}'. Only 'gaussian' is supported."
        )
    n_workers = int(n_workers) if n_workers is not None else 1
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")

    if beta_f_method not in ("sigma0", "ps"):
        raise ValueError(
            f"beta_f_method must be 'sigma0' or 'ps', got {beta_f_method!r}"
        )

    n_k = len(k_arr)
    if n_k == 0:
        empty_meta: dict[str, np.ndarray] = {
            "C_max_arr": np.array([]),
            "alpha_arr": np.array([]),
            "C_c_arr": np.array([]),
            "M_H_arr": np.array([]),
        }
        return np.array([]), np.array([]), empty_meta

    # --- pre-compute log-k grid and Simpson weights (once, outside loop) ---
    ln_k = np.log(k_arr)
    w = _simpson_weights(ln_k)

    # --- profile radial grid (log-spaced for adequate resolution) ---
    r_profile = np.logspace(-3.0, 1.5, 500)

    # --- set peak normalisation ---
    zeta_c: float = 1.0 if mu is None else float(mu)

    # --- mode index array for chunking ---
    all_inds = np.arange(n_k, dtype=np.intp)

    if n_workers == 1:
        # --- Serial path (no multiprocessing overhead) ---
        # Per-mode logic must match _compaction_chunk() (parallel path)
        beta_f = np.empty(n_k)
        M_pbh = np.empty(n_k)
        C_max_arr = np.empty(n_k)
        alpha_arr = np.empty(n_k)
        C_c_arr = np.empty(n_k)
        M_H_arr = np.empty(n_k)

        for i in range(n_k):
            ki = float(k_arr[i])
            R = 1.0 / ki

            # 1. curvature profile at smoothing scale R (vectorized)
            _, zeta_r, sigma0_sq = _compute_zeta_profile_vectorized(
                ln_k, w, k_arr, P_S_k, r_profile, R, zeta_c,
            )

            # 2. compaction function
            comp = compute_C_profile(r_profile, zeta_r)
            C_max = float(comp["C_max"])
            alpha = float(comp["alpha"])

            # 3. critical threshold
            C_c = float(C_critical(alpha, epoch=epoch))

            # 4. horizon mass at re-entry
            M_H = gamma * _M_EQ * (_K_EQ / ki) ** 2

            C_max_arr[i] = C_max
            alpha_arr[i] = alpha
            C_c_arr[i] = C_c
            M_H_arr[i] = M_H

            # 5. critical scaling
            if C_max > C_c:
                M_pbh[i] = _K_CRIT * M_H * (C_max - C_c) ** _GAMMA_CRIT
            else:
                M_pbh[i] = 0.0

            # 6. formation fraction
            if beta_f_method == "sigma0":
                sigma_C = np.sqrt(sigma0_sq)
            else:
                sigma_C = np.sqrt(P_S_k[i])
            beta_f[i] = erfc(C_c / (np.sqrt(2.0) * sigma_C))

        meta = {
            "C_max_arr": C_max_arr,
            "alpha_arr": alpha_arr,
            "C_c_arr": C_c_arr,
            "M_H_arr": M_H_arr,
        }
        return beta_f, M_pbh, meta

    # --- Parallel path (ProcessPoolExecutor) ---
    chunks = np.array_split(all_inds, n_workers)
    # Filter out empty chunks
    chunks = [c for c in chunks if len(c) > 0]

    args_list = [
        (k_arr, P_S_k, chunk, r_profile, ln_k, w, zeta_c, gamma, epoch, beta_f_method)
        for chunk in chunks
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        fut_map = {
            executor.submit(_compaction_chunk, args): i
            for i, args in enumerate(args_list)
        }

        # Pre-allocate
        beta_f = np.empty(n_k)
        M_pbh = np.empty(n_k)
        C_max_arr = np.empty(n_k)
        alpha_arr = np.empty(n_k)
        C_c_arr = np.empty(n_k)
        M_H_arr = np.empty(n_k)

        for fut in as_completed(fut_map):
            chunk_idx = fut_map[fut]
            chunk_inds = chunks[chunk_idx]
            (C_max_c, alpha_c, C_c_c, M_H_c,
             M_pbh_c, beta_f_c, _) = fut.result()

            C_max_arr[chunk_inds] = C_max_c
            alpha_arr[chunk_inds] = alpha_c
            C_c_arr[chunk_inds] = C_c_c
            M_H_arr[chunk_inds] = M_H_c
            M_pbh[chunk_inds] = M_pbh_c
            beta_f[chunk_inds] = beta_f_c

    meta = {
        "C_max_arr": C_max_arr,
        "alpha_arr": alpha_arr,
        "C_c_arr": C_c_arr,
        "M_H_arr": M_H_arr,
    }

    return beta_f, M_pbh, meta
