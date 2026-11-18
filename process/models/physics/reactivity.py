"""Thermal fusion reactivity backends.

PROCESS historically evaluates ⟨σv⟩(T) with closed-form Bosch–Hale fits for
D–T / D–³He / D–D. Reactions with resonant or otherwise structured
cross-sections (notably p–¹¹B) need an explicit Maxwellian average of σ(E)
against the relative-speed distribution at a given ion temperature.

This module provides:

* :class:`ReactivityModel` — common callable shape ``T [keV] → ⟨σv⟩ [m³/s]``
* :func:`maxwellian_reactivity` — Gauss–Laguerre ∫ σ(E) f_Maxw(E; T) dE
* :func:`tabulated_reactivity` — log–log table + interpolation wrapper
* :func:`profile_reactivity_integrand` — shared ``2 ρ ⟨σv⟩ (n/⟨n⟩)²`` profile weight

Bosch–Hale itself stays in :mod:`fusion_reactions` (regression-critical); wrap
it with :func:`as_reactivity_model` when a uniform interface is needed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from process.core import constants

# Centre-of-mass energy [MeV] → cross-section [m²]
CrossSectionFn = Callable[[np.ndarray], np.ndarray]


class ReactivityModel(Protocol):
    """Maps ion temperature [keV] to volumetric reactivity ⟨σv⟩ [m³/s]."""

    def __call__(self, t_kev: float | np.ndarray) -> np.ndarray: ...


def as_reactivity_model(
    reactivity_fn: Callable[[np.ndarray], np.ndarray],
) -> ReactivityModel:
    """Adapt an array-valued ⟨σv⟩(T) function to the :class:`ReactivityModel` shape."""

    def _model(t_kev: float | np.ndarray) -> np.ndarray:
        return np.asarray(reactivity_fn(np.asarray(t_kev, dtype=float)), dtype=float)

    return _model


def maxwellian_reactivity(
    t_kev: float | np.ndarray,
    cross_section: CrossSectionFn,
    reduced_mass_mev: float,
    *,
    n_quad: int = 160,
) -> np.ndarray:
    """Maxwellian reactivity ⟨σv⟩ [m³/s] from σ(E) at ion temperature(s) T.

    Evaluates

    .. math::

        \\langle\\sigma v\\rangle
        = \\sqrt{\\frac{8 kT}{\\pi \\mu}}
          \\int_0^\\infty \\sigma(x kT)\\, x\\, e^{-x}\\,\\mathrm{d}x

    with :math:`x = E / kT` by ``n_quad``-point Gauss–Laguerre quadrature.
    ``cross_section`` must accept centre-of-mass energy in **MeV** and return
    σ in **m²**. ``reduced_mass_mev`` is :math:`\\mu c^2` in MeV.

    Parameters
    ----------
    t_kev :
        Ion temperature [keV], scalar or array.
    cross_section :
        σ(E_cm [MeV]) → [m²].
    reduced_mass_mev :
        Reduced mass energy μc² [MeV].
    n_quad :
        Gauss–Laguerre order (default 160; matches the legacy p–¹¹B path).

    Returns
    -------
    :
        ⟨σv⟩ [m³/s], same shape as ``t_kev`` (0-d array if scalar input).
    """
    temperatures = np.atleast_1d(np.asarray(t_kev, dtype=float))
    mu_kg = (
        reduced_mass_mev
        * 1.0e6
        * constants.ELECTRON_VOLT
        / constants.SPEED_LIGHT**2
    )
    x_nodes, x_weights = np.polynomial.laguerre.laggauss(n_quad)
    kt_j = temperatures * 1.0e3 * constants.ELECTRON_VOLT

    # Centre-of-mass energies [MeV] for each temperature × Laguerre node
    energy_mev = (temperatures[:, None] * x_nodes[None, :]) / 1.0e3
    sigma = cross_section(energy_mev)
    integral = np.sum(x_weights[None, :] * sigma * x_nodes[None, :], axis=1)

    result = np.sqrt(8.0 * kt_j / (np.pi * mu_kg)) * integral
    if np.ndim(t_kev) == 0:
        return result.reshape(())
    return result


def tabulate_reactivity(
    cross_section: CrossSectionFn,
    reduced_mass_mev: float,
    t_min_kev: float,
    t_max_kev: float,
    n_points: int = 400,
    *,
    n_quad: int = 160,
    floor: float = 1.0e-60,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a geometric (T, ⟨σv⟩) table for log–log interpolation.

    Returns
    -------
    temperatures, reactivities :
        Length-``n_points`` arrays; reactivities floored at ``floor`` so
        ``log`` interpolation remains defined at low T.
    """
    temperatures = np.geomspace(t_min_kev, t_max_kev, n_points)
    reactivities = maxwellian_reactivity(
        temperatures,
        cross_section,
        reduced_mass_mev,
        n_quad=n_quad,
    )
    reactivities = np.maximum(np.asarray(reactivities, dtype=float), floor)
    return temperatures, reactivities


def interpolate_reactivity_loglog(
    t_kev: float | np.ndarray,
    temperatures: np.ndarray,
    reactivities: np.ndarray,
) -> np.ndarray:
    """Log–log interpolate a reactivity table; clip queries to the table domain."""
    scalar = np.ndim(t_kev) == 0
    query = np.clip(
        np.atleast_1d(np.asarray(t_kev, dtype=float)),
        temperatures[0],
        temperatures[-1],
    )
    result = np.exp(
        np.interp(np.log(query), np.log(temperatures), np.log(reactivities))
    )
    if scalar:
        return result.reshape(())
    return result


def make_tabulated_cross_section_reactivity(
    cross_section: CrossSectionFn,
    reduced_mass_mev: float,
    t_min_kev: float,
    t_max_kev: float,
    n_points: int = 400,
    *,
    n_quad: int = 160,
) -> ReactivityModel:
    """Return a :class:`ReactivityModel` backed by a cached Maxwellian σ×f table."""
    temperatures, reactivities = tabulate_reactivity(
        cross_section,
        reduced_mass_mev,
        t_min_kev,
        t_max_kev,
        n_points,
        n_quad=n_quad,
    )

    def _model(t_kev: float | np.ndarray) -> np.ndarray:
        return interpolate_reactivity_loglog(t_kev, temperatures, reactivities)

    return _model


def profile_reactivity_integrand(
    rho: np.ndarray,
    ion_temperature_profile: np.ndarray,
    density_profile_normalised: np.ndarray,
    reactivity: ReactivityModel,
) -> np.ndarray:
    """Volume-average weight ``2 ρ ⟨σv⟩(T_i) (n/⟨n⟩)²`` used by thermal channels.

    Integrating this with Simpson's rule over ``rho ∈ [0, 1]`` yields a
    density-normalised ⟨σv⟩ that can be scaled by reactant densities later.
    """
    sigv = np.asarray(reactivity(ion_temperature_profile), dtype=float)
    return 2.0 * rho * sigv * density_profile_normalised**2
