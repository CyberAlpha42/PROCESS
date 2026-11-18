"""Plugin registry for p–¹¹B astrophysical S-factor / cross-section models.

Each registered model exposes ``s_factor(E_cm [MeV]) → [MeV b]`` and is converted
to σ via the standard Gamow relation. Thermal reactivity then uses
:func:`process.models.physics.reactivity.maxwellian_reactivity`.

Models
------
0 : Wang et al. (2026) — default; Mazzucconi + Becker + Buck, includes 4.7 MeV
1 : Nevins & Swain (2000) — historical reference (Becker + Segel, E ≤ 3.5 MeV)
2 : Tentori & Belloni (2023) — analytic fit to Sikora–Weller + Buck
3 : Sikora–Weller lower envelope — Tentori S-factor × 0.9 (model-dependent
    absolute normalisation; **not** a design baseline — see Wang et al.)

Select with input ``i_p11b_xsec_model`` (default 0). Reactivity tables are
cached **per model index** so switching models does not reuse a stale table.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

P11B_GAMOW_ENERGY_MEV = 22.589
P11B_REDUCED_MASS_MEV = 859.526
_BARN_TO_M2 = 1.0e-28

# i_p11b_xsec_model choices
P11B_XSEC_WANG_2026 = 0
P11B_XSEC_NEVINS_SWAIN_2000 = 1
P11B_XSEC_TENTORI_BELLONI_2023 = 2
P11B_XSEC_SIKORA_WELLER_LOWER = 3

_OUT_OF_RANGE_WARNED: set[str] = set()


def _breit_wigner_sum(
    energy_kev: np.ndarray,
    amplitudes: np.ndarray,
    resonances_kev: np.ndarray,
    widths_kev: np.ndarray,
) -> np.ndarray:
    """Σ A_k / [(E − E_k)² + (δE_k)²] with E, E_k, δE_k in keV."""
    return np.sum(
        amplitudes[None, :]
        / (
            (energy_kev[:, None] - resonances_kev[None, :]) ** 2
            + widths_kev[None, :] ** 2
        ),
        axis=1,
    )


@dataclass(frozen=True)
class PiecewiseSFactor:
    """Piecewise S(E) used by Nevins / Tentori / Wang-style fits.

    Segment 1 (E ≤ e1_mev): polynomial in E[keV] + optional 148 keV BW.
    Segment 2 (e1 < E ≤ e2): polynomial in x = (E − e1) / 0.1 MeV.
    Segment 3 (e2 < E ≤ e_max): background + Σ Breit–Wigner (E in keV).
    """

    e1_mev: float
    e2_mev: float
    e_max_mev: float
    c0: float
    c1: float
    c2: float
    a_l: float | None
    e_l_kev: float
    delta_e_l_kev: float
    d0: float
    d1: float
    d2: float
    d5: float
    background: float
    amplitudes: tuple[float, ...]
    resonances_kev: tuple[float, ...]
    widths_kev: tuple[float, ...]
    s_scale: float = 1.0

    def __call__(self, energy_mev: np.ndarray) -> np.ndarray:
        energy = np.asarray(energy_mev, dtype=float)
        s_value = np.zeros_like(energy)

        mask1 = (energy > 0.0) & (energy <= self.e1_mev)
        e_kev = energy[mask1] * 1.0e3
        s1 = self.c0 + self.c1 * e_kev + self.c2 * e_kev**2
        if self.a_l is not None:
            s1 = s1 + self.a_l / (
                (e_kev - self.e_l_kev) ** 2 + self.delta_e_l_kev**2
            )
        s_value[mask1] = s1

        mask2 = (energy > self.e1_mev) & (energy <= self.e2_mev)
        x = (energy[mask2] - self.e1_mev) / 0.1
        s_value[mask2] = (
            self.d0 + self.d1 * x + self.d2 * x**2 + self.d5 * x**5
        )

        mask3 = (energy > self.e2_mev) & (energy <= self.e_max_mev)
        if np.any(mask3):
            amps = np.asarray(self.amplitudes, dtype=float)
            res = np.asarray(self.resonances_kev, dtype=float)
            wid = np.asarray(self.widths_kev, dtype=float)
            s_value[mask3] = self.background + _breit_wigner_sum(
                energy[mask3] * 1.0e3, amps, res, wid
            )

        if self.s_scale != 1.0:
            s_value = s_value * self.s_scale
        return s_value


def _wang_2026_s_factor(energy_mev: np.ndarray) -> np.ndarray:
    """Wang et al. (2026) piecewise S-factor [MeV b] (legacy PROCESS fit)."""
    energy = np.asarray(energy_mev, dtype=float)
    s_value = np.zeros_like(energy)

    mask1 = (energy > 0.0) & (energy <= 0.4)
    e_kev = energy[mask1] * 1.0e3
    s_value[mask1] = (
        197.0
        + 0.240 * e_kev
        + 2.31e-4 * e_kev**2
        + 1.82e4 / ((e_kev - 148.0) ** 2 + 2.35**2)
    )

    mask2 = (energy > 0.4) & (energy <= 0.7)
    y = (energy[mask2] - 0.4) / 0.1
    s_value[mask2] = 330.2 + 102.436 * y - 58.481 * y**2 + 0.0933 * y**5

    mask3 = (energy > 0.7) & (energy <= 10.0)
    if np.any(mask3):
        amplitudes = np.array(
            [2.0235e6, 4.0102e6, 1.3220e6, 4.9451e6, 4.3430e5]
        )
        resonances_mev = np.array([0.6222, 1.3884, 2.4924, 3.5286, 4.7036])
        widths_mev = np.array([0.0996, 0.4499, 0.2386, 0.3985, 0.1525])
        s_value[mask3] = 0.209689 + _breit_wigner_sum(
            energy[mask3] * 1.0e3,
            amplitudes,
            resonances_mev * 1.0e3,
            widths_mev * 1.0e3,
        )
    return s_value


# Nevins & Swain (2000) — coefficients as tabulated by Tentori & Belloni (2023)
_NEVINS_SWAIN = PiecewiseSFactor(
    e1_mev=0.400,
    e2_mev=0.642,
    e_max_mev=3.50,
    c0=197.0,
    c1=0.240,
    c2=2.31e-4,
    a_l=1.82e4,
    e_l_kev=148.0,
    delta_e_l_kev=2.35,
    d0=330.0,
    d1=66.1,
    d2=-20.3,
    d5=-1.58,
    background=4.38,
    amplitudes=(2.57e6, 5.67e5, 1.34e5, 5.68e5),
    resonances_kev=(581.3, 1083.0, 2405.0, 3344.0),
    widths_kev=(85.7, 234.0, 138.0, 309.0),
)

# Tentori & Belloni (2023) — fit to Sikora–Weller + Buck; 148 keV BW from NS
_TENTORI_BELLONI = PiecewiseSFactor(
    e1_mev=0.400,
    e2_mev=0.668,
    e_max_mev=9.76,
    c0=197.0,
    c1=0.269,
    c2=2.54e-4,
    a_l=1.82e4,  # taken from Nevins (SW dataset misses 148 keV resonance)
    e_l_kev=148.0,
    delta_e_l_kev=2.35,
    d0=346.0,
    d1=150.0,
    d2=-59.9,
    d5=-0.460,
    background=0.381,
    amplitudes=(1.98e6, 3.89e6, 1.36e6, 3.71e6),
    resonances_kev=(640.9, 1211.0, 2340.0, 3294.0),
    widths_kev=(85.5, 414.0, 221.0, 351.0),
)

# Sikora–Weller lower envelope: Tentori S × 0.9 (absolute-norm uncertainty)
_SIKORA_WELLER_LOWER = PiecewiseSFactor(
    e1_mev=_TENTORI_BELLONI.e1_mev,
    e2_mev=_TENTORI_BELLONI.e2_mev,
    e_max_mev=_TENTORI_BELLONI.e_max_mev,
    c0=_TENTORI_BELLONI.c0,
    c1=_TENTORI_BELLONI.c1,
    c2=_TENTORI_BELLONI.c2,
    a_l=_TENTORI_BELLONI.a_l,
    e_l_kev=_TENTORI_BELLONI.e_l_kev,
    delta_e_l_kev=_TENTORI_BELLONI.delta_e_l_kev,
    d0=_TENTORI_BELLONI.d0,
    d1=_TENTORI_BELLONI.d1,
    d2=_TENTORI_BELLONI.d2,
    d5=_TENTORI_BELLONI.d5,
    background=_TENTORI_BELLONI.background,
    amplitudes=_TENTORI_BELLONI.amplitudes,
    resonances_kev=_TENTORI_BELLONI.resonances_kev,
    widths_kev=_TENTORI_BELLONI.widths_kev,
    s_scale=0.9,
)


@dataclass(frozen=True)
class P11BCrossSectionModel:
    """One registered p–¹¹B cross-section / S-factor model."""

    index: int
    key: str
    name: str
    reference: str
    e_min_mev: float
    e_max_mev: float
    s_factor_fn: Callable[[np.ndarray], np.ndarray]
    gamow_energy_mev: float = P11B_GAMOW_ENERGY_MEV
    reduced_mass_mev: float = P11B_REDUCED_MASS_MEV
    notes: str = ""

    def s_factor(self, energy_mev: float | np.ndarray) -> np.ndarray:
        """Astrophysical S-factor [MeV b]; zero outside the fitted energy window."""
        return self.s_factor_fn(np.asarray(energy_mev, dtype=float))

    def cross_section(
        self,
        energy_mev: float | np.ndarray,
        *,
        warn_out_of_range: bool = True,
    ) -> np.ndarray:
        """Fusion cross-section [m²] from S(E) via the Gamow relation.

        Energies outside ``(0, e_max_mev]`` return 0. When
        ``warn_out_of_range`` is true, the first out-of-range query for this
        model logs a warning (not silent extrapolation).
        """
        energy = np.asarray(energy_mev, dtype=float)
        cross_section = np.zeros_like(energy, dtype=float)
        valid = (energy > 0.0) & (energy <= self.e_max_mev)
        if warn_out_of_range and np.any(~valid & (energy > 0.0)):
            warn_key = f"{self.key}:e_max"
            if warn_key not in _OUT_OF_RANGE_WARNED:
                _OUT_OF_RANGE_WARNED.add(warn_key)
                logger.warning(
                    "p-11B cross-section model %s (%s): centre-of-mass energy "
                    "exceeds fitted range E_max=%.3f MeV; σ set to 0 outside "
                    "the range (no silent extrapolation).",
                    self.index,
                    self.key,
                    self.e_max_mev,
                )
        if np.any(valid):
            e_valid = energy[valid]
            cross_section[valid] = (
                self.s_factor_fn(e_valid)
                / e_valid
                * np.exp(-np.sqrt(self.gamow_energy_mev / e_valid))
                * _BARN_TO_M2
            )
        return cross_section


def _build_registry() -> dict[int, P11BCrossSectionModel]:
    return {
        P11B_XSEC_WANG_2026: P11BCrossSectionModel(
            index=P11B_XSEC_WANG_2026,
            key="wang_2026",
            name="Wang et al. 2026",
            reference="arXiv:2601.00241",
            e_min_mev=0.0,
            e_max_mev=10.0,
            s_factor_fn=_wang_2026_s_factor,
            notes="Default. Includes 4.7 MeV resonance (Mazzucconi et al.).",
        ),
        P11B_XSEC_NEVINS_SWAIN_2000: P11BCrossSectionModel(
            index=P11B_XSEC_NEVINS_SWAIN_2000,
            key="nevins_swain_2000",
            name="Nevins & Swain 2000",
            reference="Nucl. Fusion 40, 865 (2000)",
            e_min_mev=0.0,
            e_max_mev=3.50,
            s_factor_fn=_NEVINS_SWAIN,
            notes="Historical reference; E ≤ 3.5 MeV (no 4.7 MeV resonance).",
        ),
        P11B_XSEC_TENTORI_BELLONI_2023: P11BCrossSectionModel(
            index=P11B_XSEC_TENTORI_BELLONI_2023,
            key="tentori_belloni_2023",
            name="Tentori & Belloni 2023",
            reference="Nucl. Fusion 63, 086001 (2023)",
            e_min_mev=0.0,
            e_max_mev=9.76,
            s_factor_fn=_TENTORI_BELLONI,
            notes="Analytic fit to Sikora–Weller + Buck; 148 keV BW from Nevins.",
        ),
        P11B_XSEC_SIKORA_WELLER_LOWER: P11BCrossSectionModel(
            index=P11B_XSEC_SIKORA_WELLER_LOWER,
            key="sikora_weller_lower",
            name="Sikora & Weller 2016 (lower envelope)",
            reference="J. Fusion Energy 35, 538 (2016); envelope via Tentori×0.9",
            e_min_mev=0.0,
            e_max_mev=9.76,
            s_factor_fn=_SIKORA_WELLER_LOWER,
            notes=(
                "Model-dependent absolute normalisation (ℓ of primary α). "
                "Implemented as 0.9× Tentori S-factor for a ~10% lower envelope. "
                "Not recommended as a design baseline (Wang et al. 2026)."
            ),
        ),
    }


P11B_XSEC_MODELS: dict[int, P11BCrossSectionModel] = _build_registry()


def list_p11b_cross_section_models() -> list[P11BCrossSectionModel]:
    """Return registered models sorted by ``i_p11b_xsec_model`` index."""
    return [P11B_XSEC_MODELS[i] for i in sorted(P11B_XSEC_MODELS)]


def get_p11b_cross_section_model(
    model: int | str | P11BCrossSectionModel | None = None,
) -> P11BCrossSectionModel:
    """Resolve a model by index, key, or instance (default = Wang 2026)."""
    if model is None:
        return P11B_XSEC_MODELS[P11B_XSEC_WANG_2026]
    if isinstance(model, P11BCrossSectionModel):
        return model
    if isinstance(model, str):
        for entry in P11B_XSEC_MODELS.values():
            if entry.key == model:
                return entry
        raise KeyError(
            f"Unknown p-11B cross-section model key {model!r}; "
            f"known: {[m.key for m in list_p11b_cross_section_models()]}"
        )
    if model not in P11B_XSEC_MODELS:
        raise KeyError(
            f"Unknown i_p11b_xsec_model={model}; "
            f"choices={sorted(P11B_XSEC_MODELS)}"
        )
    return P11B_XSEC_MODELS[model]


def clear_out_of_range_warnings() -> None:
    """Reset the one-shot out-of-range warning set (for unit tests)."""
    _OUT_OF_RANGE_WARNED.clear()
