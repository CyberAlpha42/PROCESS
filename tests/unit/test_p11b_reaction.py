"""Unit tests for the p-11B fusion reaction model."""

import numpy as np
import pytest
from scipy import integrate

from process.core import constants
from process.models.physics import fusion_reactions as fr


class TestP11BReactivity:
    def test_s_factor_continuity(self):
        """The piecewise S-factor fit should be continuous at the segment edges."""
        left = fr.p11b_s_factor(np.array([0.4 - 1e-6, 0.7 - 1e-6]))
        right = fr.p11b_s_factor(np.array([0.4 + 1e-6, 0.7 + 1e-6]))
        assert np.allclose(left, right, rtol=0.05)

    def test_cross_section_zero_outside_fit_range(self):
        sigma = fr.p11b_cross_section(np.array([-1.0, 0.0, 11.0, 20.0]))
        assert np.all(sigma == 0.0)
        assert fr.p11b_cross_section(np.array([0.6]))[0] > 0.0

    @pytest.mark.parametrize("temperature_kev", [50.0, 100.0, 300.0, 600.0])
    def test_reactivity_matches_direct_integration(self, temperature_kev):
        """Tabulated/interpolated reactivity should match a direct quadrature."""
        mu = (
            fr.P11B_REDUCED_MASS_MEV
            * 1.0e6
            * constants.ELECTRON_VOLT
            / constants.SPEED_LIGHT**2
        )
        kt = temperature_kev * 1.0e3 * constants.ELECTRON_VOLT

        def integrand(energy_mev):
            return fr.p11b_cross_section(energy_mev) * energy_mev * np.exp(
                -energy_mev * 1.0e6 * constants.ELECTRON_VOLT / kt
            )

        integral, _ = integrate.quad(integrand, 1.0e-4, 10.0, limit=200)
        # Convert the energy integral from MeV to joules: dE_J = A dE_MeV and
        # the extra energy factor E_J = A E_MeV gives a factor A**2.
        joule_per_mev = 1.0e6 * constants.ELECTRON_VOLT
        expected = (
            np.sqrt(8.0 / (np.pi * mu))
            * kt**-1.5
            * integral
            * joule_per_mev**2
        )
        assert fr.p11b_reactivity(temperature_kev) == pytest.approx(expected, rel=0.02)

    def test_reactivity_increases_with_temperature(self):
        values = fr.p11b_reactivity(np.array([20.0, 50.0, 100.0, 300.0, 600.0]))
        assert np.all(np.diff(values) > 0.0)

    def test_reactivity_is_scalar_safe(self):
        assert np.isscalar(fr.p11b_reactivity(100.0)) or np.ndim(
            fr.p11b_reactivity(100.0)
        ) == 0


class TestP11BReaction:
    def test_reaction_returns_early_without_fuel(self):
        """No boron (or no protons) must not change any fusion power density."""

        class DummyPlasmaProfile:
            pass

        reaction = fr.FusionReactionRate.__new__(fr.FusionReactionRate)
        reaction.plasma_profile = DummyPlasmaProfile()
        reaction.data = type(
            "D",
            (),
            {
                "physics": type(
                    "P",
                    (),
                    {
                        "nd_plasma_protons_vol_avg": 1.0e19,
                        "nd_plasma_boron11_vol_avg": 0.0,
                    },
                )()
            },
        )()
        reaction.p11b_power_density = 0.0
        reaction.p11b_rate_density = 0.0
        reaction.p11b_reaction()
        assert reaction.p11b_power_density == 0.0


class _UniformProfile:
    """Minimal stand-in for a PROCESS profile object."""

    def __init__(self, n_points=11):
        self.profile_x = np.linspace(0.0, 1.0, n_points)
        self.profile_y = np.ones(n_points)
        self.profile_dx = 1.0 / (n_points - 1)


class TestP11BAlphaContribution:
    def test_alpha_constants(self):
        """The p-11B alpha constants should mirror the D-T ones."""
        assert constants.P_B11_ALPHA_COUNT == 3.0
        assert constants.P_B11_NEUTRON_ENERGY_FRACTION == 0.0
        assert constants.P_B11_ALPHA_ENERGY_FRACTION == 1.0
        assert constants.P_B11_ALPHA_ENERGY == pytest.approx(
            constants.P_B11_ENERGY / 3.0, rel=1e-12
        )
        # ~2.894 MeV per alpha
        assert constants.P_B11_ALPHA_ENERGY / constants.ELECTRON_VOLT == pytest.approx(
            2.894e6, rel=2e-3
        )

    def test_alpha_power_and_rate_are_credited(self):
        """All p-11B power must go to alphas, with 3 alphas per reaction."""
        n_points = 11
        te_profile = _UniformProfile(n_points)
        ne_profile = _UniformProfile(n_points)
        # A 100 keV, 1e20 m^-3 plasma with 10% protons and 10% boron-11
        te_profile.profile_y = np.full(n_points, 100.0)

        reaction = fr.FusionReactionRate.__new__(fr.FusionReactionRate)
        reaction.plasma_profile = type(
            "Profile", (), {"teprofile": te_profile, "neprofile": ne_profile}
        )()
        reaction.data = type(
            "D",
            (),
            {
                "physics": type(
                    "P",
                    (),
                    {
                        "nd_plasma_protons_vol_avg": 1.0e19,
                        "nd_plasma_boron11_vol_avg": 1.0e19,
                        "nd_plasma_electrons_vol_avg": 1.0e20,
                        "temp_plasma_ion_vol_avg_kev": 100.0,
                        "temp_plasma_electron_vol_avg_kev": 100.0,
                        "i_p11b_xsec_model": 0,
                    },
                )()
            },
        )()

        # Cumulative counters used by sum_fusion_rates()
        reaction.alpha_power_density = 0.0
        reaction.pden_non_alpha_charged_mw = 0.0
        reaction.neutron_power_density = 0.0
        reaction.fusion_rate_density = 0.0
        reaction.alpha_rate_density = 0.0
        reaction.proton_rate_density = 0.0
        reaction.p11b_power_density = 0.0
        reaction.p11b_rate_density = 0.0
        reaction.p11b_alpha_rate_density = 0.0

        reaction.p11b_reaction()

        assert reaction.p11b_power_density > 0.0
        # All power is alpha power
        assert reaction.alpha_power_density == pytest.approx(
            reaction.p11b_power_density, rel=1e-12
        )
        # Three alphas per reaction
        assert reaction.p11b_alpha_rate_density == pytest.approx(
            3.0 * reaction.p11b_rate_density, rel=1e-12
        )
        assert reaction.alpha_rate_density == pytest.approx(
            3.0 * reaction.p11b_rate_density, rel=1e-12
        )
        # Nothing in the non-alpha charged or neutron buckets
        assert reaction.pden_non_alpha_charged_mw == 0.0
        assert reaction.neutron_power_density == 0.0
