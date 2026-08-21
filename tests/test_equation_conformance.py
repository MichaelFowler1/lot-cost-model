"""The projections must satisfy the equation the tool prints.

This file exists because a 36% overstatement lived in the tool while every
other test passed. The suite was full of shape tests ("unit cost falls across
the buy") and internal-consistency tests ("the risk total matches the
projections sheet"), and none of them can see a wrong formula: a monotone
curve is still monotone when it is scaled, and two numbers derived from the
same wrong projection agree with each other perfectly.

So these retype the fitted equation from the regression and evaluate it
independently. Nothing here reads a projected cost to decide what the answer
should be.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lot_cost_model as M

# Invented data, chosen because it selects LC+Rate and so exercises both the
# learning and the rate term.
RATE_QTY = [22.0, 18.0, 25.0, 30.0, 30.0, 36.0]
RATE_AUC = [4400.0, 3900.0, 3600.0, 3350.0, 3200.0, 3100.0]
RATE_FY = [2018, 2019, 2020, 2021, 2022, 2023]

#: The lots priced back against themselves cost this, by construction.
RATE_ACTUAL_TOTAL_K = float(np.sum(np.array(RATE_QTY) * np.array(RATE_AUC)))

#: Unit costs are rounded to the cent for display, so compare to the cent
#: rather than to a tolerance picked by feel.
CENT = 0.005


@pytest.fixture(scope="module")
def rate_analogy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Lot": range(1, 7),
            "Lot FY": RATE_FY,
            "Qty": RATE_QTY,
            "AUC ($K)": RATE_AUC,
        }
    )


@pytest.fixture(scope="module")
def rate_backcast() -> pd.DataFrame:
    """The analogy lots priced as the estimate, so the answer is known."""
    return pd.DataFrame(
        {
            "Lot": range(1, 7),
            "Lot FY": RATE_FY,
            "Qty": RATE_QTY,
            "Complexity": [1.0] * 6,
        }
    )


def midpoints(quantities, b, prior=0):
    """Recompute midpoints rather than reading the rounded display column."""
    spans = M.track_units(np.array(quantities, dtype=float), prior)
    return np.array(
        [M.lmp_func(s["S"], s["E"], q, b) for s, q in zip(spans, quantities)]
    )


def equation_unit_costs(ctx, projections, model):
    """Evaluate the printed equation for every projected lot.

    LC       cost = t1 * midpoint**b
    Rate     cost = t1 * qty**b            (fitted against lot quantity)
    LC+Rate  cost = t1 * midpoint**b * qty**c
    """
    qty = projections["Lot Quantity"].to_numpy(dtype=float)
    prior = int(ctx["cfg"].get("FcstPriorUnits", 0) or 0)
    scale = float(ctx["cfg"]["CostUnitScale"])

    if model == "LC":
        return ctx["t1_lc"] * midpoints(qty, ctx["b_lc"], prior) ** ctx[
            "b_lc"
        ] * scale
    if model == "Rate":
        return ctx["t1_rt"] * qty ** ctx["b_rt"] * scale
    if model == "LC+Rate":
        mid = midpoints(qty, ctx["b_br"], prior)
        return (
            ctx["t1_br"] * mid ** ctx["b_br"] * qty ** ctx["c_br"] * scale
        )
    raise AssertionError(f"unknown model {model!r}")


UNIT_COL = {
    "LC": "LC Unit Cost ($K)",
    "Rate": "Rate Unit Cost ($K)",
    "LC+Rate": "LC+Rate Unit Cost ($K)",
}


class TestProjectionsSatisfyTheEquation:
    """Every model, whether or not it was the one selected."""

    @pytest.mark.parametrize("model", ["LC", "Rate", "LC+Rate"])
    def test_on_the_rate_fixture(self, rate_analogy, rate_backcast, model):
        proj, ctx = M.run_lot_cost_model(rate_analogy, rate_backcast)
        expected = equation_unit_costs(ctx, proj, model)
        actual = proj[UNIT_COL[model]].to_numpy(dtype=float)
        assert np.all(np.isfinite(expected)), f"{model} did not fit"
        np.testing.assert_allclose(actual, expected, atol=CENT, rtol=0)

    @pytest.mark.parametrize("model", ["LC", "Rate", "LC+Rate"])
    def test_on_the_bundled_example(self, analogy_df, estimate_df, model):
        proj, ctx = M.run_lot_cost_model(analogy_df, estimate_df)
        expected = equation_unit_costs(ctx, proj, model)
        actual = proj[UNIT_COL[model]].to_numpy(dtype=float)
        np.testing.assert_allclose(actual, expected, atol=CENT, rtol=0)

    @pytest.mark.parametrize("model", ["LC", "Rate", "LC+Rate"])
    def test_still_holds_when_the_forecast_continues_production(
        self, rate_analogy, rate_backcast, model
    ):
        # Prior units move the midpoints, so this catches an equation check
        # that only works from unit 1.
        proj, ctx = M.run_lot_cost_model(
            rate_analogy, rate_backcast, {"FcstPriorUnits": 161}
        )
        expected = equation_unit_costs(ctx, proj, model)
        actual = proj[UNIT_COL[model]].to_numpy(dtype=float)
        np.testing.assert_allclose(actual, expected, atol=CENT, rtol=0)

    def test_lot_cost_is_unit_cost_times_quantity(
        self, rate_analogy, rate_backcast
    ):
        proj, ctx = M.run_lot_cost_model(rate_analogy, rate_backcast)
        scale = float(ctx["cfg"]["TotalScale"])
        qty = proj["Lot Quantity"].to_numpy(dtype=float)
        # The lot cost is built from the unrounded unit cost and only then
        # rounded, so the displayed unit cost can be out by up to half a cent.
        # Carry that through rather than loosening the tolerance by feel.
        slack = qty * scale * CENT
        for model, unit_col in UNIT_COL.items():
            before = proj[
                f"{model} Lot Cost Before Complexity ($)"
            ].to_numpy(dtype=float)
            expected = proj[unit_col].to_numpy(dtype=float) * qty * scale
            assert np.all(np.abs(before - expected) <= slack + 0.01), (
                f"{model} lot cost is not its unit cost times quantity"
            )


class TestBackCast:
    """Pricing the analogy lots as the estimate recovers what they cost."""

    def test_selected_model_recovers_the_known_total(
        self, rate_analogy, rate_backcast
    ):
        proj, ctx = M.run_lot_cost_model(rate_analogy, rate_backcast)
        summary = M.generate_analyst_summary(ctx, {"Program": "TEST"})
        selected = summary.loc[summary["Item"] == "SELECTED"].iloc[0]
        model = [c for c in ("LC", "Rate", "LC+Rate")
                 if selected[c] == "YES"][0]
        assert model == "LC+Rate", "fixture no longer exercises the rate term"

        total_k = (
            proj[f"{model} Lot Cost After Complexity ($)"].sum()
            / float(ctx["cfg"]["TotalScale"])
        )
        # A good fit, so within a fraction of a percent of the real total.
        assert total_k == pytest.approx(RATE_ACTUAL_TOTAL_K, rel=0.005)

    def test_every_model_back_casts_close_to_the_actuals(
        self, rate_analogy, rate_backcast
    ):
        proj, ctx = M.run_lot_cost_model(rate_analogy, rate_backcast)
        scale = float(ctx["cfg"]["TotalScale"])
        for model in ("LC", "LC+Rate"):
            total_k = (
                proj[f"{model} Lot Cost After Complexity ($)"].sum() / scale
            )
            assert total_k == pytest.approx(RATE_ACTUAL_TOTAL_K, rel=0.02), (
                f"{model} back-cast is {total_k:,.0f} against an actual "
                f"{RATE_ACTUAL_TOTAL_K:,.0f}"
            )


class TestLegacySwitch:
    """The old behaviour is still reachable, and still wrong."""

    def test_legacy_lcr_is_always_higher_never_lower(
        self, rate_analogy, rate_backcast
    ):
        # The rate exponent is negative and lot quantities exceed one, so
        # dropping qty**c can only push cost up.
        fixed, ctx = M.run_lot_cost_model(rate_analogy, rate_backcast)
        legacy, _ = M.run_lot_cost_model(
            rate_analogy, rate_backcast, {"LegacyRateOmission": True}
        )
        assert ctx["c_br"] < 0
        assert (
            legacy["LC+Rate Unit Cost ($K)"]
            > fixed["LC+Rate Unit Cost ($K)"]
        ).all()

    def test_legacy_overstates_this_fixture_by_about_a_third(
        self, rate_analogy, rate_backcast
    ):
        fixed, ctx = M.run_lot_cost_model(rate_analogy, rate_backcast)
        legacy, _ = M.run_lot_cost_model(
            rate_analogy, rate_backcast, {"LegacyRateOmission": True}
        )
        col = "LC+Rate Lot Cost After Complexity ($)"
        overstatement = legacy[col].sum() / fixed[col].sum() - 1.0
        assert overstatement == pytest.approx(0.361, abs=0.005)

    def test_legacy_lcr_drops_the_rate_term(
        self, rate_analogy, rate_backcast
    ):
        legacy, ctx = M.run_lot_cost_model(
            rate_analogy, rate_backcast, {"LegacyRateOmission": True}
        )
        qty = legacy["Lot Quantity"].to_numpy(dtype=float)
        mid = midpoints(qty, ctx["b_br"])
        without_rate = ctx["t1_br"] * mid ** ctx["b_br"]
        np.testing.assert_allclose(
            legacy["LC+Rate Unit Cost ($K)"].to_numpy(dtype=float),
            without_rate,
            atol=CENT,
            rtol=0,
        )

    def test_legacy_rate_projects_on_the_midpoint(
        self, rate_analogy, rate_backcast
    ):
        legacy, ctx = M.run_lot_cost_model(
            rate_analogy, rate_backcast, {"LegacyRateOmission": True}
        )
        qty = legacy["Lot Quantity"].to_numpy(dtype=float)
        mid = midpoints(qty, ctx["b_rt"])
        np.testing.assert_allclose(
            legacy["Rate Unit Cost ($K)"].to_numpy(dtype=float),
            ctx["t1_rt"] * mid ** ctx["b_rt"],
            atol=CENT,
            rtol=0,
        )

    def test_lc_is_untouched_by_the_switch(self, rate_analogy, rate_backcast):
        # LC has no rate term to drop, so old estimates that selected LC
        # were never affected.
        fixed, _ = M.run_lot_cost_model(rate_analogy, rate_backcast)
        legacy, _ = M.run_lot_cost_model(
            rate_analogy, rate_backcast, {"LegacyRateOmission": True}
        )
        pd.testing.assert_series_equal(
            fixed["LC Unit Cost ($K)"], legacy["LC Unit Cost ($K)"]
        )

    def test_corrected_is_the_default(self):
        assert SettingsDefault() is False

    def test_the_replaced_key_is_refused(self, rate_analogy, rate_backcast):
        # Honouring it silently would give a caller who asked for legacy
        # behaviour the corrected numbers without saying so.
        for value in (True, False):
            with pytest.raises(ValueError, match="LegacyRateOmission"):
                M.run_lot_cost_model(
                    rate_analogy,
                    rate_backcast,
                    {M.LEGACY_KEY: value},
                )


def SettingsDefault() -> bool:
    return M.SETTINGS["LegacyRateOmission"]
