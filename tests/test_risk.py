"""The cost_core bridge: intervals, simulation, and the guards around both.

The tests that matter most here are the ones in TestSpanContract. The bridge
redirects a private cost_core hook so a buy can be priced from unit 1, and a
silent change in that hook would mean simulating the wrong lots and reporting
confident numbers for them.
"""

from __future__ import annotations

import numpy as np
import pytest

import risk as R

pytestmark = pytest.mark.skipif(
    not R.AVAILABLE, reason=f"cost_core not installed: {R.IMPORT_ERROR}"
)

FCST_QTY = [8, 16, 16, 16, 12, 6]
CF = [1.15] * 6


def options(**kw):
    base = dict(dollar_year=2025, program="TEST", n_iter=4000, seed=11)
    base.update(kw)
    return R.RiskOptions(**base)


@pytest.fixture(scope="module")
def result(cfg):
    qty = [10.0, 20.0, 25.0, 25.0, 15.0, 15.0]
    auc = [800.61, 639.49, 563.66, 520.05, 502.98, 487.08]
    return R.run_risk(qty, auc, FCST_QTY, CF, cfg, options(n_iter=20000))


class TestSpans:
    def test_matches_the_main_module_unit_tracking(self):
        import lot_cost_model as M

        spans = R.unit_spans([9, 21, 22], 0)
        tracked = M.track_units(np.array([9, 21, 22]), 0)
        assert spans.tolist() == [[d["S"], d["E"]] for d in tracked]

    def test_prior_units_shift_the_series(self):
        assert R.unit_spans([5, 5], 100).tolist() == [[101, 105], [106, 110]]


class TestSpanContract:
    """cost_core's private span hook is load-bearing. Pin it."""

    def test_hook_still_exists_on_the_report(self):
        from cost_core.lots import LotFitReport

        assert hasattr(LotFitReport, R.SPAN_HOOK), (
            f"cost_core no longer defines {R.SPAN_HOOK}; the Monte Carlo "
            "can no longer be pinned to our lot positions."
        )

    def test_our_spans_reproduce_cost_cores_own_continuation(self, cfg):
        # When the forecast continues from the last unit built, cost_core
        # would derive exactly the spans we hand it. If this drifts, our
        # redirect is changing the answer rather than just relocating it.
        from cost_core.learning_curve import fit_curve
        from cost_core.lots import LotFitReport, LotSeries

        qty = np.array([10, 20, 25, 25, 15, 15], dtype=float)
        auc = np.array([800.61, 639.49, 563.66, 520.05, 502.98, 487.08])
        ranges = R.unit_spans(qty, 0)
        fit = fit_curve(
            theory="crawford", method="ols", lots=ranges, lot_costs=auc * qty
        )
        series = LotSeries(
            quantities=qty,
            costs=auc * qty,
            cost_basis="recurring",
            first_unit=1,
            dollar_year=2025,
        )
        report = LotFitReport(series=series, fit=fit)

        theirs = np.asarray(getattr(report, R.SPAN_HOOK)(FCST_QTY))
        ours = R.unit_spans(FCST_QTY, int(qty.sum()))
        assert theirs.tolist() == ours.tolist()

    def test_simulation_is_refused_if_the_hook_disappears(
        self, cfg, monkeypatch
    ):
        # Simulate a future cost_core that renamed the hook. The bridge must
        # refuse rather than silently simulate its own choice of lots.
        monkeypatch.setattr(R, "SPAN_HOOK", "_hook_that_does_not_exist")
        with pytest.raises(RuntimeError, match="could not be pinned"):
            R.run_risk(
                [10.0, 20.0, 25.0, 25.0, 15.0, 15.0],
                [800.61, 639.49, 563.66, 520.05, 502.98, 487.08],
                FCST_QTY,
                CF,
                cfg,
                options(),
            )

    def test_redirect_records_being_called(self):
        spans = R.unit_spans([4, 4], 0)
        redirect = R._SpanRedirect(spans)
        assert redirect.calls == 0
        assert redirect([4, 4]) is spans
        assert redirect.calls == 1


class TestIntervals:
    def test_interval_brackets_the_point_estimate(self, result):
        iv = result.intervals
        assert (iv["Unit Cost Low ($K)"] <= iv["Unit Cost ($K)"]).all()
        assert (iv["Unit Cost ($K)"] <= iv["Unit Cost High ($K)"]).all()

    def test_one_row_per_forecast_lot(self, result):
        assert len(result.intervals) == len(FCST_QTY)

    def test_total_is_the_sum_of_the_lots(self, result):
        assert result.total_point == pytest.approx(
            result.intervals["Lot Cost ($)"].sum(), rel=1e-9
        )

    def test_unit_cost_falls_across_the_buy(self, result):
        costs = result.intervals["Unit Cost ($K)"].to_numpy()
        assert np.all(np.diff(costs) < 0)

    def test_a_wider_level_gives_a_wider_interval(self, cfg):
        qty = [10.0, 20.0, 25.0, 25.0, 15.0, 15.0]
        auc = [800.61, 639.49, 563.66, 520.05, 502.98, 487.08]
        narrow = R.run_risk(
            qty, auc, FCST_QTY, CF, cfg, options(level=0.50, simulate=False)
        )
        wide = R.run_risk(
            qty, auc, FCST_QTY, CF, cfg, options(level=0.95, simulate=False)
        )
        assert (wide.total_upper - wide.total_lower) > (
            narrow.total_upper - narrow.total_lower
        )


class TestSimulation:
    def test_percentiles_are_ordered(self, result):
        assert result.p50 < result.p80 < result.p90

    def test_point_estimate_lands_near_the_middle(self, result):
        # The point estimate is the median of a roughly symmetric draw.
        assert 35 < result.point_percentile < 65

    def test_same_seed_reproduces_the_answer(self, cfg):
        qty = [10.0, 20.0, 25.0, 25.0, 15.0, 15.0]
        auc = [800.61, 639.49, 563.66, 520.05, 502.98, 487.08]
        a = R.run_risk(qty, auc, FCST_QTY, CF, cfg, options())
        b = R.run_risk(qty, auc, FCST_QTY, CF, cfg, options())
        assert a.p80 == b.p80

    def test_a_different_seed_moves_the_answer(self, cfg):
        qty = [10.0, 20.0, 25.0, 25.0, 15.0, 15.0]
        auc = [800.61, 639.49, 563.66, 520.05, 502.98, 487.08]
        a = R.run_risk(qty, auc, FCST_QTY, CF, cfg, options(seed=1))
        b = R.run_risk(qty, auc, FCST_QTY, CF, cfg, options(seed=2))
        assert a.p80 != b.p80

    def test_dropping_residual_scatter_narrows_the_spread(self, cfg):
        qty = [10.0, 20.0, 25.0, 25.0, 15.0, 15.0]
        auc = [800.61, 639.49, 563.66, 520.05, 502.98, 487.08]
        with_scatter = R.run_risk(
            qty, auc, FCST_QTY, CF, cfg, options(include_residual=True)
        )
        without = R.run_risk(
            qty, auc, FCST_QTY, CF, cfg, options(include_residual=False)
        )
        assert without.sim_cv < with_scatter.sim_cv


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
    def test_scales_the_total_linearly(self, cfg):
        qty = [10.0, 20.0, 25.0, 25.0, 15.0, 15.0]
        auc = [800.61, 639.49, 563.66, 520.05, 502.98, 487.08]
        one = R.run_risk(
            qty, auc, FCST_QTY, [1.0] * 6, cfg, options(simulate=False)
        )
        two = R.run_risk(
            qty, auc, FCST_QTY, [2.0] * 6, cfg, options(simulate=False)
        )
        assert two.total_point == pytest.approx(2 * one.total_point, rel=1e-9)

    def test_a_blank_factor_is_treated_as_one(self, cfg):
        qty = [10.0, 20.0, 25.0, 25.0, 15.0, 15.0]
        auc = [800.61, 639.49, 563.66, 520.05, 502.98, 487.08]
        blank = R.run_risk(
            qty, auc, FCST_QTY, [np.nan] * 6, cfg, options(simulate=False)
        )
        ones = R.run_risk(
            qty, auc, FCST_QTY, [1.0] * 6, cfg, options(simulate=False)
        )
        assert blank.total_point == pytest.approx(ones.total_point)


class TestQuantityOnlyLots:
    def test_units_are_kept_in_the_cumulative_count(self, cfg):
        # cost_core's own CSV path drops these rows, which would slide every
        # later lot along the curve. Ours must not.
        qty = [10.0, 20.0, 25.0, 25.0, 15.0, 15.0]
        auc = [800.61, 639.49, 563.66, np.nan, 502.98, 487.08]
        res = R.run_risk(
            qty, auc, FCST_QTY, CF, cfg, options(simulate=False)
        )
        assert res.n_obs == 5
        assert any("quantity-only" in n for n in res.notes)


class TestGuards:
    def test_base_year_is_required(self, cfg):
        with pytest.raises(ValueError, match="base year"):
            R.run_risk(
                [10.0, 20.0, 25.0],
                [800.0, 640.0, 560.0],
                FCST_QTY,
                CF,
                cfg,
                options(dollar_year=None),
            )

    def test_three_costed_lots_are_required(self, cfg):
        with pytest.raises(ValueError, match="at least 3"):
            R.run_risk(
                [10.0, 20.0],
                [800.0, 640.0],
                FCST_QTY,
                CF,
                cfg,
                options(),
            )

    def test_low_degrees_of_freedom_is_warned_about(self, cfg):
        res = R.run_risk(
            [10.0, 20.0, 25.0],
            [800.0, 640.0, 560.0],
            FCST_QTY,
            CF,
            cfg,
            options(simulate=False),
        )
        assert res.df <= 2
        assert any("degree" in w for w in res.warnings)


class TestBasis:
    def test_continuation_prices_later_units_than_unit_one(self, cfg):
        qty = [10.0, 20.0, 25.0, 25.0, 15.0, 15.0]
        auc = [800.61, 639.49, 563.66, 520.05, 502.98, 487.08]
        from_one = R.run_risk(
            qty, auc, FCST_QTY, CF, cfg, options(simulate=False)
        )
        cont = R.run_risk(
            qty,
            auc,
            FCST_QTY,
            CF,
            dict(cfg, FcstPriorUnits=110),
            options(simulate=False),
        )
        assert cont.intervals["First Unit in Lot"].iloc[0] == 111
        assert cont.total_point < from_one.total_point

    def test_the_analogy_caveat_is_stated_when_pricing_from_unit_one(
        self, result
    ):
        assert any("analogy" in n for n in result.notes)


class TestSummaryFrame:
    def test_renders_without_a_simulation(self, cfg):
        qty = [10.0, 20.0, 25.0, 25.0, 15.0, 15.0]
        auc = [800.61, 639.49, 563.66, 520.05, 502.98, 487.08]
        res = R.run_risk(
            qty, auc, FCST_QTY, CF, cfg, options(simulate=False)
        )
        frame = R.summary_frame(res)
        assert list(frame.columns) == ["Item", "Value"]
        assert (frame["Item"] == "T1 first unit cost ($K)").any()

    def test_includes_percentiles_after_a_simulation(self, result):
        frame = R.summary_frame(result)
        assert (frame["Item"] == "Simulated P80 ($)").any()
