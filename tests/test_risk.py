"""The cost_core bridge: intervals and the simulated buy total.

The bridge hands cost_core the objects run_lot_cost_model already produced,
so these tests care mostly about one thing: the risk numbers must describe
the same lots, the same selected model and the same point estimate as the
projections sheet. If they ever drift apart, the tool is reporting a
distribution around a number it is not showing anyone.
"""

from __future__ import annotations

import numpy as np
import pytest

import lot_cost_model as M
import risk as R

pytestmark = pytest.mark.skipif(
    not R.AVAILABLE, reason=f"cost_core not installed: {R.IMPORT_ERROR}"
)


def options(**kw):
    base = dict(n_iter=4000, seed=11)
    base.update(kw)
    return R.RiskOptions(**base)


@pytest.fixture(scope="module")
def fitted(analogy_df, estimate_df):
    proj, ctx = M.run_lot_cost_model(analogy_df, estimate_df)
    summary = M.generate_analyst_summary(ctx, {"Program": "TEST"})
    return ctx, proj, summary


@pytest.fixture(scope="module")
def result(fitted):
    ctx, proj, summary = fitted
    return R.run_risk(ctx, proj, summary, options(n_iter=20000))


class TestAgreementWithTheModel:
    """The whole point of enriching rather than refitting."""

    def test_uses_the_model_the_tool_selected(self, fitted, result):
        _, _, summary = fitted
        selected = summary.loc[summary["Item"] == "SELECTED"]
        picked = [c for c in ("LC", "Rate", "LC+Rate")
                  if selected.iloc[0][c] == "YES"]
        assert result.model == picked[0]

    def test_point_estimate_matches_the_projections_sheet(self, fitted, result):
        _, proj, _ = fitted
        col = f"{result.model} Lot Cost After Complexity ($)"
        assert result.total_point == pytest.approx(
            proj[col].sum(), rel=1e-6
        )

    def test_one_interval_row_per_forecast_lot(self, fitted, result):
        _, proj, _ = fitted
        assert len(result.intervals) == len(proj)

    def test_simulation_centres_on_the_point_estimate(self, result):
        assert 35 < result.point_percentile < 65


class TestIntervals:
    def test_interval_brackets_the_point_estimate(self, result):
        iv = result.intervals
        assert (iv["Unit Cost Lower"] <= iv["Unit Cost ($K)"]).all()
        assert (iv["Unit Cost ($K)"] <= iv["Unit Cost Upper"]).all()

    def test_total_is_the_sum_of_the_lots(self, result):
        assert result.total_point == pytest.approx(
            result.intervals["Lot Cost ($)"].sum(), rel=1e-9
        )
        assert result.total_lower < result.total_point < result.total_upper

    def test_the_band_tracks_the_selected_model(self, result):
        # The example selects LC+Rate, whose unit cost turns back up on the
        # small final lot, so this deliberately does not assert a falling
        # curve. What must hold is that the band follows the point estimate.
        iv = result.intervals
        point = iv["Unit Cost ($K)"].to_numpy()
        lower = iv["Unit Cost Lower"].to_numpy()
        upper = iv["Unit Cost Upper"].to_numpy()
        assert np.all(np.diff(np.sign(np.diff(point))) >= -1)  # one turn
        np.testing.assert_array_less(lower, point)
        np.testing.assert_array_less(point, upper)

    def test_a_wider_level_gives_a_wider_interval(self, fitted):
        ctx, proj, summary = fitted
        narrow = R.run_risk(
            ctx, proj, summary, options(level=0.50, simulate=False)
        )
        wide = R.run_risk(
            ctx, proj, summary, options(level=0.95, simulate=False)
        )
        assert (wide.total_upper - wide.total_lower) > (
            narrow.total_upper - narrow.total_lower
        )


class TestSimulation:
    def test_percentiles_are_ordered(self, result):
        assert result.p50 < result.p80 < result.p90

    def test_reserve_to_p80_is_the_gap_above_the_point(self, result):
        assert result.reserve_to_p80 == pytest.approx(
            result.p80 - result.total_point, rel=1e-6
        )

    def test_same_seed_reproduces_the_answer(self, fitted):
        ctx, proj, summary = fitted
        a = R.run_risk(ctx, proj, summary, options())
        b = R.run_risk(ctx, proj, summary, options())
        assert a.p80 == b.p80

    def test_a_different_seed_moves_the_answer(self, fitted):
        ctx, proj, summary = fitted
        a = R.run_risk(ctx, proj, summary, options(seed=1))
        b = R.run_risk(ctx, proj, summary, options(seed=2))
        assert a.p80 != b.p80

    def test_more_correlation_widens_the_buy(self, fitted):
        # Consecutive lots moving together stops the shocks cancelling.
        ctx, proj, summary = fitted
        low = R.run_risk(ctx, proj, summary, options(lot_correlation=0.0))
        high = R.run_risk(ctx, proj, summary, options(lot_correlation=0.9))
        assert high.sim_cv > low.sim_cv


class TestSCurve:
    def test_has_a_row_per_percentile(self, result):
        assert len(result.scurve) == 99

    def test_is_monotonic(self, result):
        vals = result.scurve["Buy Total ($)"].to_numpy()
        assert np.all(np.diff(vals) >= 0)

    def test_agrees_with_the_reported_percentiles(self, result):
        sc = result.scurve.set_index(
            (result.scurve["Percentile"] * 100).round().astype(int)
        )["Buy Total ($)"]
        assert sc[50] == pytest.approx(result.p50, rel=1e-6)
        assert sc[80] == pytest.approx(result.p80, rel=1e-6)
        assert sc[90] == pytest.approx(result.p90, rel=1e-6)


class TestComplexityFactor:
    def test_is_already_in_the_intervals(self, analogy_df, estimate_df):
        # The projections carry complexity, so the intervals inherit it and
        # doubling the factor doubles the whole distribution.
        doubled = estimate_df.copy()
        doubled["Complexity"] = estimate_df["Complexity"] * 2

        def total(est):
            proj, ctx = M.run_lot_cost_model(analogy_df, est)
            summary = M.generate_analyst_summary(ctx, {"Program": "TEST"})
            return R.run_risk(
                ctx, proj, summary, options(simulate=False)
            ).total_point

        assert total(doubled) == pytest.approx(2 * total(estimate_df), rel=1e-6)


class TestQuantityOnlyLots:
    def test_are_called_out_in_the_notes(self, analogy_df, estimate_df):
        gapped = analogy_df.copy()
        gapped.loc[3, "AUC ($K)"] = np.nan
        proj, ctx = M.run_lot_cost_model(gapped, estimate_df)
        summary = M.generate_analyst_summary(ctx, {"Program": "TEST"})
        res = R.run_risk(ctx, proj, summary, options(simulate=False))
        assert res.n_obs == 5
        assert any("quantity-only" in n for n in res.notes)


class TestBasis:
    def test_continuation_is_cheaper_than_pricing_from_unit_one(
        self, analogy_df, estimate_df
    ):
        def total(prior):
            proj, ctx = M.run_lot_cost_model(
                analogy_df, estimate_df, {"FcstPriorUnits": prior}
            )
            summary = M.generate_analyst_summary(ctx, {"Program": "TEST"})
            return R.run_risk(
                ctx, proj, summary, options(simulate=False)
            ).total_point

        assert total(110) < total(0)

    def test_the_analogy_caveat_is_stated_when_pricing_from_unit_one(
        self, result
    ):
        assert any("analogy" in n for n in result.notes)

    def test_continuation_says_so_instead(self, analogy_df, estimate_df):
        proj, ctx = M.run_lot_cost_model(
            analogy_df, estimate_df, {"FcstPriorUnits": 110}
        )
        summary = M.generate_analyst_summary(ctx, {"Program": "TEST"})
        res = R.run_risk(ctx, proj, summary, options(simulate=False))
        assert any("continue from unit 111" in n for n in res.notes)


class TestGuards:
    def test_refuses_when_no_model_was_selected(self, fitted):
        import pandas as pd

        ctx, proj, _ = fitted
        empty = pd.DataFrame({"Item": ["Program"], "Value": ["TEST"],
                              "LC": [""], "Rate": [""], "LC+Rate": [""]})
        with pytest.raises(RuntimeError, match="which model"):
            R.run_risk(ctx, proj, empty, options(simulate=False))


class TestSummaryFrame:
    def test_renders_without_a_simulation(self, fitted):
        ctx, proj, summary = fitted
        res = R.run_risk(ctx, proj, summary, options(simulate=False))
        frame = R.summary_frame(res)
        assert list(frame.columns) == ["Item", "Value"]
        assert (frame["Item"] == "T1 first unit cost ($K)").any()

    def test_includes_percentiles_after_a_simulation(self, result):
        frame = R.summary_frame(result)
        assert (frame["Item"] == "Simulated P80 ($)").any()
        assert (frame["Item"] == "Reserve to P80 ($)").any()
