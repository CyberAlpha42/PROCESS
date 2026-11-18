"""Unit tests for 5-species fuel fraction accounting (p-11B fuel composition)."""

from __future__ import annotations

import pytest

from process.core import constants
from process.core.exceptions import ProcessValidationError
from process.core.init import check_process
from process.core.model import DataStructure
from process.data_structure.impurity_radiation_variables import N_IMPURITIES


def _fuel_z_bar(f_d, f_t, f_he3, f_p, f_b):
    return f_d + f_t + 2.0 * f_he3 + f_p + 5.0 * f_b


def _m_fuel(f_d, f_t, f_he3, f_p, f_b):
    return (
        constants.M_DEUTERON_AMU * f_d
        + constants.M_TRITON_AMU * f_t
        + constants.M_HELION_AMU * f_he3
        + constants.M_PROTON_AMU * f_p
        + constants.M_BORON11_AMU * f_b
    )


def _data_ready_for_fuel_checks() -> DataStructure:
    """Minimal DataStructure that reaches the fuel-fraction checks in check_process."""
    data = DataStructure()
    data.numerics.n_iteration_variables = 0
    data.numerics.n_equality_constraints = 0
    data.numerics.n_inequality_constraints = 0
    # He impurity slot must stay at the required default
    data.impurity_radiation.f_nd_impurity_electrons[1] = 0.1
    for i in range(N_IMPURITIES):
        if i != 1:
            data.impurity_radiation.f_nd_impurity_electrons[i] = 0.0
    data.physics.f_nd_alpha_thermal_electron = 0.01
    data.physics.f_nd_beam_electron = 0.0
    data.physics.f_nd_protium_electrons = 0.0
    data.physics.f_nd_boron11_electrons = 0.0
    return data


class TestFuelAccountingFormulas:
    """Pure-function checks of the n_fuel / m_fuel formulas (no solver)."""

    def test_dt_degeneracy_matches_legacy_znfuel(self):
        """f_p=f_B=0 → n_fuel = znfuel/(1+f_He3) bit-for-bit with legacy."""
        n_e = 1.0e20
        f_alpha = 0.1
        f_beam = 0.0
        f_d, f_t, f_he3 = 0.5, 0.5, 0.0
        f_p = f_b = 0.0
        n_p_ash = 0.0
        n_b_legacy = 0.0
        znimp = 0.0

        q_avail = (
            n_e
            - 2.0 * f_alpha * n_e
            - n_p_ash
            - f_beam * n_e
            - 5.0 * n_b_legacy
            - znimp
        )
        z_bar = _fuel_z_bar(f_d, f_t, f_he3, f_p, f_b)
        n_fuel = q_avail / z_bar

        znfuel_legacy = q_avail
        n_fuel_legacy = znfuel_legacy / (1.0 + f_he3)

        assert z_bar == pytest.approx(1.0 + f_he3)
        assert n_fuel == n_fuel_legacy

    def test_pure_pb_analytic(self):
        """Pure pB f_p=5/6, f_B=1/6 with ash/imp≈0 → analytic n_p, n_B, m_fuel."""
        n_e = 1.0e20
        f_alpha = 0.01
        f_p, f_b = 5.0 / 6.0, 1.0 / 6.0
        f_d = f_t = f_he3 = 0.0

        q_avail = n_e - 2.0 * f_alpha * n_e
        z_bar = _fuel_z_bar(f_d, f_t, f_he3, f_p, f_b)
        n_fuel = q_avail / z_bar
        n_p = f_p * n_fuel
        n_b = f_b * n_fuel
        m_fuel = _m_fuel(f_d, f_t, f_he3, f_p, f_b)

        assert z_bar == pytest.approx(1.0 * f_p + 5.0 * f_b)
        assert n_p + 5.0 * n_b + 2.0 * f_alpha * n_e == pytest.approx(n_e, rel=1e-12)
        assert n_p / n_b == pytest.approx(5.0, rel=1e-12)
        assert m_fuel == pytest.approx(
            constants.M_PROTON_AMU * f_p + constants.M_BORON11_AMU * f_b,
            rel=1e-12,
        )
        assert n_fuel > 0.0
        assert m_fuel > 0.0


class TestFuelValidation:
    def test_five_species_sum_required(self):
        data = _data_ready_for_fuel_checks()
        data.physics.f_plasma_fuel_deuterium = 0.0
        data.physics.f_plasma_fuel_tritium = 0.0
        data.physics.f_plasma_fuel_helium3 = 0.0
        data.physics.f_plasma_fuel_protium = 0.0
        data.physics.f_plasma_fuel_boron11 = 0.0
        with pytest.raises(ProcessValidationError, match="Fuel ion fractions"):
            check_process({}, data)

    def test_mutual_exclusion_fuel_vs_legacy(self):
        data = _data_ready_for_fuel_checks()
        data.physics.f_plasma_fuel_deuterium = 0.0
        data.physics.f_plasma_fuel_tritium = 0.0
        data.physics.f_plasma_fuel_helium3 = 0.0
        data.physics.f_plasma_fuel_protium = 5.0 / 6.0
        data.physics.f_plasma_fuel_boron11 = 1.0 / 6.0
        data.physics.f_nd_protium_electrons = 0.1
        with pytest.raises(ProcessValidationError, match="choose one path"):
            check_process({}, data)

    def test_q_avail_precheck_readable(self):
        data = _data_ready_for_fuel_checks()
        data.physics.f_plasma_fuel_deuterium = 0.5
        data.physics.f_plasma_fuel_tritium = 0.5
        data.physics.f_plasma_fuel_helium3 = 0.0
        data.physics.f_plasma_fuel_protium = 0.0
        data.physics.f_plasma_fuel_boron11 = 0.0
        data.physics.f_nd_alpha_thermal_electron = 0.6  # 2*0.6 > 1
        with pytest.raises(ProcessValidationError, match="q_avail"):
            check_process({}, data)

    def test_pure_pb_fuel_section_accepted(self):
        data = _data_ready_for_fuel_checks()
        data.physics.f_plasma_fuel_deuterium = 0.0
        data.physics.f_plasma_fuel_tritium = 0.0
        data.physics.f_plasma_fuel_helium3 = 0.0
        data.physics.f_plasma_fuel_protium = 5.0 / 6.0
        data.physics.f_plasma_fuel_boron11 = 1.0 / 6.0
        try:
            check_process({}, data)
        except ProcessValidationError as exc:
            msg = str(exc)
            assert "Fuel ion fractions" not in msg
            assert "choose one path" not in msg
            assert "q_avail" not in msg
