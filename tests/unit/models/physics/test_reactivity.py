"""Unit tests for the thermal reactivity interface (Bosch–Hale + σ×Maxwellian)."""

import numpy as np
import pytest

from process.models.physics import fusion_reactions as fr
from process.models.physics import reactivity as reac


class TestMaxwellianReactivity:
    def test_matches_legacy_p11b_direct(self):
        """General integrator must reproduce the former p–¹¹B quadrature."""
        for t_kev in (50.0, 100.0, 300.0, 600.0):
            general = reac.maxwellian_reactivity(
                t_kev, fr.p11b_cross_section, fr.P11B_REDUCED_MASS_MEV
            )
            legacy = fr._p11b_reactivity_direct(t_kev)
            assert float(general) == pytest.approx(float(legacy), rel=1e-12)

    def test_vectorised(self):
        temps = np.array([20.0, 100.0, 400.0])
        values = reac.maxwellian_reactivity(
            temps, fr.p11b_cross_section, fr.P11B_REDUCED_MASS_MEV
        )
        assert values.shape == temps.shape
        assert np.all(np.diff(values) > 0.0)

    def test_zero_cross_section_gives_zero(self):
        def sigma_zero(energy_mev):
            return np.zeros_like(np.asarray(energy_mev, dtype=float))

        assert float(
            reac.maxwellian_reactivity(100.0, sigma_zero, fr.P11B_REDUCED_MASS_MEV)
        ) == pytest.approx(0.0)


class TestTabulatedReactivity:
    def test_p11b_table_matches_direct(self):
        """Tabulated log–log path agrees with direct quadrature (~2%)."""
        model = reac.make_tabulated_cross_section_reactivity(
            fr.p11b_cross_section,
            fr.P11B_REDUCED_MASS_MEV,
            0.5,
            3000.0,
            n_points=400,
        )
        for t_kev in (50.0, 100.0, 300.0):
            assert float(model(t_kev)) == pytest.approx(
                float(fr._p11b_reactivity_direct(t_kev)), rel=0.02
            )

    def test_public_p11b_reactivity_uses_interface(self):
        """fusion_reactions.p11b_reactivity stays consistent with the table API."""
        temps, rates = reac.tabulate_reactivity(
            fr.p11b_cross_section,
            fr.P11B_REDUCED_MASS_MEV,
            0.5,
            3000.0,
            n_points=400,
        )
        for t_kev in (80.0, 200.0):
            assert float(fr.p11b_reactivity(t_kev)) == pytest.approx(
                float(reac.interpolate_reactivity_loglog(t_kev, temps, rates)),
                rel=1e-10,
            )


class TestBoschHaleViaInterface:
    def test_as_reactivity_model_preserves_bosch_hale(self):
        """Wrapping Bosch–Hale must be bit-identical to the bare call."""
        constants = fr.BoschHaleConstants(**fr.REACTION_CONSTANTS_DT)
        model = reac.as_reactivity_model(
            lambda t: fr.bosch_hale_reactivity(t, constants)
        )
        t = np.array([10.0, 20.0, 55.73])
        assert np.allclose(
            model(t), fr.bosch_hale_reactivity(t, constants), rtol=0.0, atol=0.0
        )


class TestProfileIntegrand:
    def test_uniform_profile_recovers_reactivity(self):
        """Flat n and T → integrand integrates to ⟨σv⟩ itself."""
        n_points = 21
        rho = np.linspace(0.0, 1.0, n_points)
        ti = np.full(n_points, 100.0)
        dens_norm = np.ones(n_points)
        model = reac.as_reactivity_model(fr.p11b_reactivity)
        integrand = reac.profile_reactivity_integrand(rho, ti, dens_norm, model)
        # ∫_0^1 2ρ ⟨σv⟩ dρ = ⟨σv⟩ for constant ⟨σv⟩
        from scipy import integrate

        averaged = integrate.simpson(integrand, x=rho)
        assert averaged == pytest.approx(float(fr.p11b_reactivity(100.0)), rel=1e-6)
