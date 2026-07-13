"""
PBH accretion models: Bondi-Hoyle-Lyttleton (Ricotti+2007), Park-Ricotti
with radiative feedback (Agius+2024), and Chisholm legacy 3×10⁷ factor.

All internal computations in cgs; public methods accept/return masses in
solar masses.  NumPy/SciPy only — no astropy dependency.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Final, TypedDict

from scripts.constants import (
    CAMB_COSMOLOGY, T_cmb, ACCRETION,
    G, M_SUN, MPC_CM, K_B, M_P, KM_PER_S, C_LIGHT, SIGMA_SB, A_R, GAMMA_B, MU, N_EFF
)

DEFAULT_COSMOLOGY: Final[dict[str, float]] = {
    'H0': CAMB_COSMOLOGY['H0'],
    'ombh2': CAMB_COSMOLOGY['ombh2'],
    'omch2': CAMB_COSMOLOGY['omch2'],
    'T_cmb': T_cmb,
}


class CosmologyArrays(TypedDict):
    """Vectorised cosmology quantities at a set of redshifts (cgs)."""
    rho_gas: np.ndarray
    c_s: np.ndarray
    v_PBH: np.ndarray
    H: np.ndarray
    rho_crit: np.ndarray


class PBHAccretionError(Exception):
    """Raised on invalid accretion parameters or integration failure."""
    def __init__(self, message: str) -> None:
        self.message: str = message
        super().__init__(message)


# ── Cosmology helper ──────────────────────────────────────────────────────


def _cosmology_arrays(z_arr: np.ndarray, cosmology: dict[str, float]) -> CosmologyArrays:
    """Redshift-dependent cosmology quantities in cgs units.

    Parameters
    ----------
    z_arr : np.ndarray
        Redshift array.
    cosmology : dict
        Keys ``H0`` (km/s/Mpc), ``ombh2``, ``omch2``, ``T_cmb`` (K).

    Returns
    -------
    CosmologyArrays
        ``rho_gas``, ``c_s``, ``v_PBH``, ``H``, ``rho_crit``.
    """
    H0 = cosmology.get('H0', CAMB_COSMOLOGY['H0'])
    ombh2 = cosmology.get('ombh2', CAMB_COSMOLOGY['ombh2'])
    omch2 = cosmology.get('omch2', CAMB_COSMOLOGY['omch2'])
    T_cmb_val = cosmology.get('T_cmb', T_cmb)

    h = H0 / 100.0
    H0_cgs = H0 * KM_PER_S / MPC_CM
    Omega_m = (ombh2 + omch2) / h**2
    Omega_b = ombh2 / h**2

    rho_crit_0 = 3.0 * H0_cgs**2 / (8.0 * np.pi * G)
    Omega_gamma = (A_R * T_cmb_val**4 / C_LIGHT**2) / rho_crit_0
    Omega_r = Omega_gamma * (1.0 + 0.2271 * N_EFF)
    Omega_L = 1.0 - Omega_m - Omega_r

    z = np.asarray(z_arr, dtype=np.float64)
    H_z = H0_cgs * np.sqrt(Omega_m * (1.0 + z)**3 + Omega_r * (1.0 + z)**4 + Omega_L)
    rho_gas = Omega_b * rho_crit_0 * (1.0 + z)**3
    T_gas = T_cmb_val * (1.0 + z)
    c_s = np.sqrt(GAMMA_B * K_B * T_gas / (MU * M_P))
    v_PBH = 10.0 * (1.0 + z / 1000.0)**0.5 * KM_PER_S

    return CosmologyArrays(
        rho_gas=rho_gas, c_s=c_s, v_PBH=v_PBH, H=H_z,
        rho_crit=3.0 * H_z**2 / (8.0 * np.pi * G),
    )


# ── BHL accretion ─────────────────────────────────────────────────────────


class PBHAccretion:
    """Bondi-Hoyle-Lyttleton accretion (Ricotti, Ostriker & Mack 2007).

    .. math:: \\dot{M} = 4\\pi \\lambda G^2 M^2 \\rho_\\mathrm{gas} / v_\\mathrm{eff}^3

    with :math:`v_\\mathrm{eff}^2 = c_s^2 + v_\\mathrm{PBH}^2`.

    Parameters
    ----------
    model : str
        Label (default ``'BHL'``).
    lambda_acc : float
        Efficiency λ ∈ (0, 1].
    z_cutoff : float
        Redshift below which accretion is quenched.
    cosmology : dict or None
        Planck 2018 overrides (keys ``H0``, ``ombh2``, ``omch2``, ``T_cmb``).
    """

    def __init__(  # noqa: PLR0913
        self,
        model: str = 'BHL',
        lambda_acc: float = 0.1,
        z_cutoff: float = 10.0,
        cosmology: dict[str, float] | None = None,
    ) -> None:
        if not 0.0 < lambda_acc <= 1.0:
            raise PBHAccretionError(f'lambda_acc must be in (0, 1], got {lambda_acc}')
        if z_cutoff < 0.0:
            raise PBHAccretionError(f'z_cutoff must be >= 0, got {z_cutoff}')
        self.model = model
        self._lambda = lambda_acc
        self._z_cutoff = z_cutoff
        self._cosmology = {**DEFAULT_COSMOLOGY, **(cosmology or {})}

    def accretion_rate(self, M: float, z: float) -> float:
        """BHL :math:`\\dot{M}` [g/s] at mass *M* [M_⊙] and redshift *z*."""
        if M <= 0.0:
            return 0.0
        cosmo = _cosmology_arrays(np.array([z]), self._cosmology)
        v_eff = np.sqrt(cosmo['c_s'][0]**2 + cosmo['v_PBH'][0]**2)
        return 4.0 * np.pi * self._lambda * G**2 * (M * M_SUN)**2 * cosmo['rho_gas'][0] / v_eff**3  # noqa: E501

    def _accretion_ode(self, z: float, M: float) -> float:
        """dM/dz = -(Ṁ / M_⊙) / ((1+z) H)."""
        if z < self._z_cutoff:
            return 0.0
        return -(self.accretion_rate(M, z) / M_SUN) / ((1.0 + z) * float(
            _cosmology_arrays(np.array([z]), self._cosmology)['H'][0]
        ))

    def M_of_redshift(  # noqa: PLR0913
        self, M_form: float, z_obs: float, z_form: float | None = None,
    ) -> float:
        """Mass [M_⊙] at *z_obs* given formation mass at *z_form*.

        Integrates BHL rate from *z_form* to *z_obs*.  Below *z_cutoff*
        accretion is quenched and mass stays constant.

        Parameters
        ----------
        M_form : float
            Formation mass [M_⊙] (> 0).
        z_obs : float
            Observation redshift (<= *z_form*).
        z_form : float or None
            Formation redshift (default ~3400).

        Returns
        -------
        float
            Mass at *z_obs* [M_⊙].
        """
        if M_form <= 0.0:
            raise PBHAccretionError(f'M_form must be > 0, got {M_form}')
        if z_form is None:
            z_form = 3400.0
        if z_obs > z_form:
            raise PBHAccretionError(f'z_obs ({z_obs}) must be <= z_form ({z_form})')
        if np.isclose(z_obs, z_form):
            return M_form
        sol = solve_ivp(
            self._accretion_ode, (z_form, z_obs), np.array([M_form]),
            method='LSODA', atol=1e-12, rtol=1e-6, max_step=50.0,
        )
        if not sol.success:
            raise PBHAccretionError(f'M_of_redshift failed: {sol.message}')
        return float(sol.y[0, -1])

    def evolve(  # noqa: PLR0913
        self, M_form: float, z_form: float | None = None,
        z_final: float = 0.0, n_steps: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Full mass evolution from *z_form* to *z_final*.

        Parameters
        ----------
        M_form : float
            Formation mass [M_⊙] (> 0).
        z_form : float or None
            Formation redshift.
        z_final : float
            Final redshift (default 0).
        n_steps : int
            Output points (> 2).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(M_history [M_⊙], z_history)``.
        """
        if M_form <= 0.0:
            raise PBHAccretionError(f'M_form must be > 0, got {M_form}')
        if n_steps <= 2:
            raise PBHAccretionError(f'n_steps must be > 2, got {n_steps}')
        if z_form is None:
            z_form = 3400.0
        if z_final >= z_form:
            return (np.full(n_steps, M_form), np.linspace(z_form, z_final, n_steps))
        z_span = np.linspace(z_form, z_final, n_steps)
        sol = solve_ivp(
            self._accretion_ode, (z_form, z_final), np.array([M_form]),
            t_eval=z_span, method='LSODA', atol=1e-12, rtol=1e-6, max_step=50.0,
        )
        if not sol.success:
            raise PBHAccretionError(f'evolve failed: {sol.message}')
        return np.asarray(sol.y[0]), np.asarray(sol.t)

    def mass_growth_factor(self, M_form: float, z_obs: float,
                           z_form: float | None = None) -> float:
        """Ratio *M(z_obs) / M_form*."""
        return self.M_of_redshift(M_form, z_obs, z_form) / M_form


# ── Park-Ricotti (radiative feedback) ──────────────────────────────────────


class PR_Accretion(PBHAccretion):
    """Park-Ricotti accretion with photo-ionisation feedback.

    Replaces :math:`c_s \\to \\max(c_s, c_s^\\mathrm{ion})` where
    :math:`c_s^\\mathrm{ion} \\approx 23` km/s (T_ion ≈ 10⁴ K).

    Parameters
    ----------
    model : str
        Label (default ``'PR'``).
    lambda_acc : float
        Efficiency.
    z_cutoff : float
        Redshift cutoff.
    cosmology : dict or None
        Cosmology parameters.
    c_s_ionized : float
        Ionised sound speed [cm/s] (default 23 km/s).
    """

    def __init__(  # noqa: PLR0913
        self, model: str = 'PR', lambda_acc: float = 0.1,
        z_cutoff: float = 10.0, cosmology: dict[str, float] | None = None,
        c_s_ionized: float = 23.0e5,
    ) -> None:
        super().__init__(model=model, lambda_acc=lambda_acc, z_cutoff=z_cutoff, cosmology=cosmology)
        self._c_s_ionized = c_s_ionized

    def accretion_rate(self, M: float, z: float) -> float:
        """PR rate with radiative-feedback-suppressed sound speed."""
        if M <= 0.0:
            return 0.0
        cosmo = _cosmology_arrays(np.array([z]), self._cosmology)
        c_s_eff = max(float(cosmo['c_s'][0]), self._c_s_ionized)
        v_eff = np.sqrt(c_s_eff**2 + cosmo['v_PBH'][0]**2)
        return 4.0 * np.pi * self._lambda * G**2 * (M * M_SUN)**2 * cosmo['rho_gas'][0] / v_eff**3  # noqa: E501


# ── Chisholm legacy model ──────────────────────────────────────────────────


class ChisholmAccretion(PBHAccretion):
    """Legacy constant-factor accretion (Chisholm 2006, 3×10⁷).

    At z=0 mass is multiplied by 3×10⁷; elsewhere no growth.
    """

    _CHISHOLM_FACTOR: Final[float] = ACCRETION

    def M_of_redshift(  # noqa: PLR0913
        self, M_form: float, z_obs: float, z_form: float | None = None,
    ) -> float:
        if M_form <= 0.0:
            raise PBHAccretionError(f'M_form must be > 0, got {M_form}')
        return M_form * self._CHISHOLM_FACTOR if z_obs == 0.0 else M_form

    def evolve(  # noqa: PLR0913
        self, M_form: float, z_form: float | None = None,
        z_final: float = 0.0, n_steps: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        if M_form <= 0.0:
            raise PBHAccretionError(f'M_form must be > 0, got {M_form}')
        if z_form is None:
            z_form = 3400.0
        if z_final == 0.0:
            return (np.array([M_form, M_form * self._CHISHOLM_FACTOR]),
                    np.array([z_form, 0.0]))
        return (np.array([M_form, M_form]), np.array([z_form, z_final]))

    def mass_growth_factor(self, M_form: float, z_obs: float,
                           z_form: float | None = None) -> float:
        return self._CHISHOLM_FACTOR if z_obs == 0.0 else 1.0


# ── Eddington-limited accretion (DM halo enhanced) ────────────────────────


class EddingtonAccretion(PBHAccretion):
    """Eddington-limited BHL accretion with DM halo density enhancement.

    The accretion rate is the minimum of the Bondi-Hoyle rate (with a
    mass-dependent halo density boost) and the Eddington limit:

    .. math::

        \\dot{M} = \\min\\bigl(
            f_{\\rm boost}\\,\\dot{M}_{\\rm BHL},\\;
            \\dot{M}_{\\rm Edd}
        \\bigr)

    The halo boost factor models the enhanced gas density at the Bondi
    radius due to a dark-matter halo around the PBH (Ricotti+2007):

    .. math::

        f_{\\rm boost} = f_0\\, (M / M_{\\rm pivot})^{\\alpha}

    so larger PBHs have proportionally denser halos and faster accretion.
    The Eddington limit uses radiative efficiency ε.

    Parameters
    ----------
    model : str
        Label (default ``'Eddington'``).
    lambda_acc : float
        BHL efficiency λ ∈ (0, 1] (default 0.1).
    z_cutoff : float
        Redshift below which accretion is quenched (default 10.0).
    cosmology : dict or None
        Cosmology parameters.
    f_boost : float
        Halo density enhancement normalisation (default 1e4).
        With default ``f_boost=1e4`` the boosted-BHL rate exceeds the
        Eddington limit for all realistic PBH masses, so the model is
        effectively Eddington-limited (``Ṁ ∝ M``).  Set ``f_boost=1``
        for a pure Eddington model without halo enhancement.
    boost_index : float
        Mass exponent α for the boost factor (default 0.5).
    M_pivot : float
        Pivot mass [M_⊙] for the boost scaling (default 1.0).
    epsilon_rad : float
        Radiative efficiency for the Eddington limit (default 0.1).
    """

    def __init__(  # noqa: PLR0913
        self, model: str = 'Eddington', lambda_acc: float = 0.1,
        z_cutoff: float = 10.0, cosmology: dict[str, float] | None = None,
        f_boost: float = 1e4, boost_index: float = 0.5,
        M_pivot: float = 1.0, epsilon_rad: float = 0.1,
    ) -> None:
        super().__init__(
            model=model, lambda_acc=lambda_acc,
            z_cutoff=z_cutoff, cosmology=cosmology,
        )
        if f_boost <= 0.0:
            raise PBHAccretionError(f'f_boost must be > 0, got {f_boost}')
        if epsilon_rad <= 0.0 or epsilon_rad > 1.0:
            raise PBHAccretionError(
                f'epsilon_rad must be in (0, 1], got {epsilon_rad}'
            )
        self._f_boost = f_boost
        self._boost_index = boost_index
        self._M_pivot = M_pivot
        self._epsilon_rad = epsilon_rad

    def _eddington_rate(self, M: float) -> float:
        """Eddington accretion rate [g/s] for mass *M* [M_⊙].

        Ṁ_Edd = L_Edd / (ε c²) = 4π G M m_p / (ε σ_T c)
        """
        sigma_T = 6.652458732e-25  # cm², Thomson cross-section
        M_val = M.item() if hasattr(M, 'item') else float(M)
        return float(
            4.0 * np.pi * G * M_val * M_SUN * M_P
            / (self._epsilon_rad * sigma_T * C_LIGHT)
        )

    def accretion_rate(self, M: float, z: float) -> float:
        """Eddington-limited BHL rate with halo boost [g/s]."""
        if M <= 0.0:
            return 0.0
        # ODE solver may pass 1-D arrays; extract scalar
        M_val = M.item() if hasattr(M, 'item') else float(M)
        z_val = z.item() if hasattr(z, 'item') else float(z)

        cosmo = _cosmology_arrays(np.array([z_val]), self._cosmology)
        rho_gas = float(cosmo['rho_gas'][0])
        c_s = float(cosmo['c_s'][0])
        v_pbh = float(cosmo['v_PBH'][0])
        v_eff = float(np.sqrt(c_s**2 + v_pbh**2))
        M_bhl = (
            4.0 * np.pi * self._lambda * G**2 * (M_val * M_SUN)**2
            * rho_gas / v_eff**3
        )

        # Halo density enhancement (mass-dependent)
        boost = self._f_boost * (M_val / self._M_pivot) ** self._boost_index
        M_boosted = M_bhl * max(boost, 1.0)

        # Eddington cap
        M_edd = self._eddington_rate(M_val)

        return min(M_boosted, M_edd)

    def mass_growth_factor(self, M_form: float, z_obs: float,
                           z_form: float | None = None) -> float:
        """Growth factor *M(z_obs) / M_form*."""
        return self.M_of_redshift(M_form, z_obs, z_form) / M_form


# ── Merger history contribution ─────────────────────────────────────────────


class MergerAccretion(PBHAccretion):
    """Hierarchical PBH merger mass growth (simplified model).

    Accounts for mass growth through PBH-PBH binary mergers over cosmic
    history.  More massive PBHs have larger binary-formation cross-sections
    and undergo proportionally more mergers:

    .. math::

        M_{\\rm final} = M_{\\rm initial} \\times
        \\bigl(1 + A\\,(M_{\\rm initial} / M_{\\rm pivot})^{\\gamma}\\bigr)

    where the constants *A*, *γ* are calibrated to PBH binary formation
    models (Raidal+2019, Vaskonen+2020).  The growth is applied
    instantaneously at :math:`z = z_{\\rm form}` (no ODE integration).

    Parameters
    ----------
    model : str
        Label (default ``'Merger'``).
    A : float
        Merger amplitude (default 0.5).  The average number of mergers
        experienced by a ``M_pivot`` PBH.
    gamma : float
        Mass exponent (default 0.5).  Larger values → more merger growth
        for high-mass PBHs.
    M_pivot : float
        Pivot mass [M_⊙] for the merger scaling (default 1.0).
    """

    def __init__(  # noqa: PLR0913
        self, model: str = 'Merger',
        A: float = 0.5, gamma: float = 0.5,
        M_pivot: float = 1.0, **kwargs: object,
    ) -> None:
        super().__init__(model=model, lambda_acc=1.0, **kwargs)
        if A < 0.0:
            raise PBHAccretionError(f'A must be >= 0, got {A}')
        if gamma < 0.0:
            raise PBHAccretionError(f'gamma must be >= 0, got {gamma}')
        self._A = A
        self._gamma = gamma
        self._M_pivot = M_pivot

    def growth_factor(self, M: float) -> float:
        """Merger growth factor ``1 + A * (M / M_pivot)^gamma``."""
        return 1.0 + self._A * (M / self._M_pivot) ** self._gamma

    def M_of_redshift(  # noqa: PLR0913
        self, M_form: float, z_obs: float, z_form: float | None = None,
    ) -> float:
        if M_form <= 0.0:
            raise PBHAccretionError(f'M_form must be > 0, got {M_form}')
        # Merger growth is instantaneous at formation and persists to z=0
        return M_form * self.growth_factor(M_form)

    def evolve(  # noqa: PLR0913
        self, M_form: float, z_form: float | None = None,
        z_final: float = 0.0, n_steps: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        if M_form <= 0.0:
            raise PBHAccretionError(f'M_form must be > 0, got {M_form}')
        if z_form is None:
            z_form = 3400.0
        M_merged = M_form * self.growth_factor(M_form)
        if z_final < z_form:
            return (np.full(n_steps, M_merged),
                    np.linspace(z_form, z_final, n_steps))
        return (np.full(n_steps, M_form),
                np.linspace(z_form, z_final, n_steps))

    def mass_growth_factor(self, M_form: float, z_obs: float,
                           z_form: float | None = None) -> float:
        return self.growth_factor(M_form) if z_obs < self._z_cutoff else 1.0
