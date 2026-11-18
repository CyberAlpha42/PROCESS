"""Unit tests for the p–¹¹B cross-section plugin registry."""

import numpy as np
import pytest

from process.models.physics import fusion_reactions as fr
from process.models.physics import p11b_cross_sections as xsec


@pytest.fixture(autouse=True)
def _clear_caches():
    fr.clear_p11b_reactivity_tables()
    xsec.clear_out_of_range_warnings()
    yield
    fr.clear_p11b_reactivity_tables()
    xsec.clear_out_of_range_warnings()


class TestRegistry:
    def test_four_models_registered(self):
        models = xsec.list_p11b_cross_section_models()
        assert [m.index for m in models] == [0, 1, 2, 3]
        assert models[0].key == "wang_2026"

    def test_resolve_by_key_and_index(self):
        assert xsec.get_p11b_cross_section_model(1).key == "nevins_swain_2000"
        assert (
            xsec.get_p11b_cross_section_model("tentori_belloni_2023").index == 2
        )

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            xsec.get_p11b_cross_section_model(99)
        with pytest.raises(KeyError):
            xsec.get_p11b_cross_section_model("no_such_model")


class TestWangBackwardCompat:
    def test_legacy_wrappers_match_registry_default(self):
        e = np.array([0.2, 0.6, 1.5, 5.0])
        model = xsec.get_p11b_cross_section_model(0)
        assert np.allclose(fr.p11b_s_factor(e), model.s_factor(e))
        assert np.allclose(
            fr.p11b_cross_section(e, warn_out_of_range=False),
            model.cross_section(e, warn_out_of_range=False),
        )


class TestModelPhysics:
    @pytest.mark.parametrize("model_id", [0, 1, 2, 3])
    def test_s_factor_nonnegative_in_range(self, model_id):
        model = xsec.get_p11b_cross_section_model(model_id)
        e = np.geomspace(0.05, min(model.e_max_mev, 3.0), 40)
        assert np.all(model.s_factor(e) >= 0.0)

    def test_nevins_zero_above_3p5_mev(self):
        model = xsec.get_p11b_cross_section_model(1)
        assert model.cross_section(4.0, warn_out_of_range=False) == 0.0

    def test_out_of_range_warns_once(self, caplog):
        model = xsec.get_p11b_cross_section_model(1)
        with caplog.at_level("WARNING"):
            model.cross_section(np.array([4.0, 5.0]))
            model.cross_section(np.array([6.0]))
        warnings = [r for r in caplog.records if "exceeds fitted range" in r.message]
        assert len(warnings) == 1

    def test_sikora_is_tentori_scaled(self):
        tentori = xsec.get_p11b_cross_section_model(2)
        sikora = xsec.get_p11b_cross_section_model(3)
        e = np.array([0.3, 0.6, 1.2, 3.0])
        assert np.allclose(sikora.s_factor(e), 0.9 * tentori.s_factor(e), rtol=1e-12)


class TestReactivityAcrossModels:
    @pytest.mark.parametrize("t_kev", [100.0, 200.0, 300.0])
    def test_models_agree_within_literature_band(self, t_kev):
        """⟨σv⟩ spread among models at thermonuclear T is O(10%), not factors."""
        values = {
            m.index: float(fr.p11b_reactivity(t_kev, model=m.index))
            for m in xsec.list_p11b_cross_section_models()
        }
        assert all(v > 0.0 for v in values.values())
        # Relative to Wang default
        wang = values[0]
        for model_id, value in values.items():
            assert value == pytest.approx(wang, rel=0.25), (
                f"model {model_id} at T={t_kev} keV differs from Wang by "
                f"{abs(value / wang - 1):.1%}"
            )

    def test_per_model_table_cache_isolation(self):
        """Switching models must not reuse another model's table."""
        r0 = float(fr.p11b_reactivity(150.0, model=0))
        r1 = float(fr.p11b_reactivity(150.0, model=1))
        r0_again = float(fr.p11b_reactivity(150.0, model=0))
        assert r0 == pytest.approx(r0_again, rel=0.0, abs=0.0)
        # Nevins vs Wang differ at 150 keV (high-energy content).
        # abs=0: default abs=1e-12 would treat ~1e-22 rates as equal.
        assert r0 != pytest.approx(r1, rel=1e-6, abs=0)

    def test_temperature_clip_warns(self, caplog):
        with caplog.at_level("WARNING"):
            fr.p11b_reactivity(5000.0, model=0)
            fr.p11b_reactivity(6000.0, model=0)
        warnings = [r for r in caplog.records if "clipped" in r.message]
        assert len(warnings) == 1
