"""The deterministic model: lot midpoints, unit tracking, and the fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lot_cost_model as M


class TestLotMidpoint:
    def test_single_unit_lot_is_its_own_midpoint(self):
        assert M.lmp_func(5, 5, 1, -0.15) == 5.0

    def test_zero_slope_gives_the_arithmetic_midpoint(self):
        # With no learning the midpoint is just the middle of the lot.
        assert M.lmp_func(1, 11, 11, 0.0) == pytest.approx(6.0)

    def test_midpoint_lies_inside_the_lot(self):
        mid = M.lmp_func(11, 30, 20, -0.152)
        assert 11 <= mid <= 30

    def test_midpoint_moves_earlier_as_learning_steepens(self):
        # A steeper curve weights the cheap late units less, pulling the
        # cost-average unit toward the start of the lot.
        shallow = M.lmp_func(11, 30, 20, -0.05)
        steep = M.lmp_func(11, 30, 20, -0.40)
        assert steep < shallow

    def test_unknown_slope_gives_nan(self):
        assert pd.isna(M.lmp_func(1, 10, 10, None))
        assert pd.isna(M.lmp_func(1, 10, 10, np.nan))

    def test_b_equals_minus_one_uses_the_log_branch(self):
        # b = -1 is a removable singularity; it must not divide by zero.
        val = M.lmp_func(1, 10, 10, -1.0)
        assert np.isfinite(val) and val > 0


class TestUnitTracking:
    def test_lots_are_contiguous_from_unit_one(self):
        se = M.track_units(np.array([9, 21, 22]), 0)
        assert [(d["S"], d["E"]) for d in se] == [(1, 9), (10, 30), (31, 52)]

    def test_prior_units_shift_the_whole_series(self):
        se = M.track_units(np.array([5, 5]), 100)
        assert [(d["S"], d["E"]) for d in se] == [(101, 105), (106, 110)]


class TestHelpers:
    def test_find_col_is_case_and_space_insensitive(self):
        cols = ["Lot FY", "AUC ($K)"]
        assert M.find_col(cols, ["auc ($k)"]) == "AUC ($K)"
        assert M.find_col(cols, ["nope", " lot fy "]) == "Lot FY"

    def test_find_col_returns_none_when_absent(self):
        assert M.find_col(["a"], ["b"]) is None

    def test_to_num_strips_currency_formatting(self):
        assert M.to_num("$1,234.50") == pytest.approx(1234.50)
        assert M.to_num(7) == 7.0
        assert pd.isna(M.to_num("not a number"))
        assert pd.isna(M.to_num(""))


class TestFit:
    def test_the_example_fit_has_not_drifted(self, analogy_df, estimate_df):
        # A regression pin, not a claim about recovering the truth: with lot
        # size rising monotonically, cumulative units and lot size move
        # together, so the learning and rate exponents trade off against each
        # other even though the combined fit is excellent. If these move, the
        # fit moved.
        proj, ctx = M.run_lot_cost_model(analogy_df, estimate_df)
        assert proj["LC T1 First Unit Cost ($K)"].iloc[0] == pytest.approx(
            1111.17, abs=0.01
        )
        assert proj["LC Learning Slope (%)"].iloc[0] == pytest.approx(
            83.89, abs=0.01
        )
        assert proj["LC+Rate T1 First Unit Cost ($K)"].iloc[0] == (
            pytest.approx(1340.23, abs=0.01)
        )
        assert proj["LC+Rate Learning Slope (%)"].iloc[0] == pytest.approx(
            91.24, abs=0.01
        )
        assert proj["LC+Rate Rate Slope (%)"].iloc[0] == pytest.approx(
            87.06, abs=0.01
        )
        assert ctx["n_keep"] == 6

    def test_the_example_exercises_all_three_models(
        self, analogy_df, estimate_df
    ):
        # The point of this data: quantities spread widely enough that a rate
        # term has something to regress against, so every model fits and the
        # selection is made on merit rather than by default.
        proj, ctx = M.run_lot_cost_model(analogy_df, estimate_df)
        summary = M.generate_analyst_summary(ctx, {"Program": "TEST"})
        fitted = summary.loc[summary["Item"] == "Fitted"].iloc[0]
        assert [fitted["LC"], fitted["Rate"], fitted["LC+Rate"]] == [
            "Yes", "Yes", "Yes",
        ]
        assert ctx["rate_ok"] is True

        selected = summary.loc[summary["Item"] == "SELECTED"].iloc[0]
        assert selected["LC+Rate"] == "YES"

        t_row = summary.loc[summary["Item"] == "t (rate coefficient)"].iloc[0]
        assert abs(float(t_row["LC+Rate"])) >= M.SETTINGS["TGate"]

    def test_the_fixture_matches_the_example_in_the_app(
        self, analogy_df, estimate_df
    ):
        # The Load Example buttons and these fixtures must not drift apart.
        assert [
            (str(int(r["Lot FY"])), str(int(r["Qty"])), f"{r['AUC ($K)']:.2f}")
            for _, r in analogy_df.iterrows()
        ] == [tuple(row) for row in M.EXAMPLE_ANALOGY]
        assert [
            (str(int(r["Lot FY"])), str(int(r["Qty"])), str(r["Complexity"]))
            for _, r in estimate_df.iterrows()
        ] == [tuple(row) for row in M.EXAMPLE_ESTIMATE]

    def test_projects_one_row_per_forecast_lot(self, analogy_df, estimate_df):
        proj, _ = M.run_lot_cost_model(analogy_df, estimate_df)
        assert len(proj) == len(estimate_df)

    def test_learning_curve_cost_falls_across_the_buy(
        self, analogy_df, estimate_df
    ):
        # True of LC specifically, because its only driver is the midpoint,
        # which rises with every lot. It is not true of the models carrying a
        # rate term; see the test below.
        proj, _ = M.run_lot_cost_model(analogy_df, estimate_df)
        costs = proj["LC Unit Cost ($K)"].to_numpy()
        assert np.all(np.diff(costs) < 0)

    def test_a_smaller_lot_costs_more_per_unit_under_a_rate_model(
        self, analogy_df, estimate_df
    ):
        # The rate term's whole content. The buy tapers from 40 units to 10,
        # and unit cost turns back up, which a pure learning curve cannot do.
        proj, _ = M.run_lot_cost_model(analogy_df, estimate_df)
        biggest = proj["Lot Quantity"].idxmax()
        smallest = proj["Lot Quantity"].idxmin()
        assert (
            proj["LC+Rate Unit Cost ($K)"].iloc[smallest]
            > proj["LC+Rate Unit Cost ($K)"].iloc[biggest]
        )
        assert not np.all(
            np.diff(proj["LC+Rate Unit Cost ($K)"].to_numpy()) < 0
        )

    def test_complexity_factor_scales_the_lot_cost(
        self, analogy_df, estimate_df
    ):
        doubled = estimate_df.copy()
        doubled["Complexity"] = estimate_df["Complexity"] * 2
        base, _ = M.run_lot_cost_model(analogy_df, estimate_df)
        twice, _ = M.run_lot_cost_model(analogy_df, doubled)
        assert twice["LC Lot Cost After Complexity ($)"].sum() == pytest.approx(
            2 * base["LC Lot Cost After Complexity ($)"].sum(), rel=1e-9
        )

    def test_blank_complexity_carries_the_previous_lot_forward(
        self, analogy_df, estimate_df
    ):
        sparse = estimate_df.copy()
        sparse["Complexity"] = [2.0] + [np.nan] * (len(sparse) - 1)
        proj, _ = M.run_lot_cost_model(analogy_df, sparse)
        assert set(proj["Complexity Factor"]) == {2.0}

    def test_quantity_only_lots_keep_their_units(
        self, analogy_df, estimate_df
    ):
        gapped = analogy_df.copy()
        gapped.loc[3, "AUC ($K)"] = np.nan  # lot 4 has no usable cost
        _, ctx = M.run_lot_cost_model(gapped, estimate_df)
        assert ctx["n_keep"] == 5      # fitted on five lots
        assert ctx["n_unit"] == 6      # but six lots' units are tracked
        # Quantities are 5, 9, 14, 22, 34, 50, so lot 5 starts at unit 51.
        # It must still start there, after lot 4's 22 units, not in place of
        # them at unit 29.
        starts = [d["S"] for d in ctx["fit_se"]]
        assert starts == [1, 6, 15, 51, 85]

    def test_refuses_fewer_than_three_costed_lots(self, estimate_df):
        thin = pd.DataFrame(
            {
                "Lot": [1, 2],
                "Lot FY": [2015, 2016],
                "Qty": [10.0, 20.0],
                "AUC ($K)": [800.0, 640.0],
            }
        )
        with pytest.raises(ValueError, match="at least 3"):
            M.run_lot_cost_model(thin, estimate_df)

    def test_refuses_an_empty_estimate_table(self, analogy_df):
        empty = pd.DataFrame(columns=["Lot", "Lot FY", "Qty", "Complexity"])
        with pytest.raises(ValueError):
            M.run_lot_cost_model(analogy_df, empty)

    def test_rate_model_is_gated_off_when_quantities_are_uniform(
        self, estimate_df
    ):
        # No spread in lot quantity means nothing to say about a rate effect.
        flat = pd.DataFrame(
            {
                "Lot": [1, 2, 3, 4],
                "Lot FY": [2015, 2016, 2017, 2018],
                "Qty": [20.0, 20.0, 20.0, 20.0],
                "AUC ($K)": [800.0, 760.0, 735.0, 720.0],
            }
        )
        _, ctx = M.run_lot_cost_model(flat, estimate_df)
        assert ctx["rate_ok"] is False
        assert "uniform" in ctx["rate_why"] or "spread" in ctx["rate_why"]


class TestSummary:
    def test_selection_is_reported(self, analogy_df, estimate_df):
        _, ctx = M.run_lot_cost_model(analogy_df, estimate_df)
        summary = M.generate_analyst_summary(ctx, {"Program": "TEST"})
        selected = summary.loc[summary["Item"] == "SELECTED"]
        assert len(selected) == 1
        assert "YES" in selected.iloc[0].tolist()

    def test_base_year_absence_is_called_out(self, analogy_df, estimate_df):
        _, ctx = M.run_lot_cost_model(analogy_df, estimate_df)
        summary = M.generate_analyst_summary(ctx, {"BaseYear": ""})
        basis = summary.loc[summary["Item"] == "Cost basis", "Value"].iloc[0]
        assert "NOT STATED" in basis


class TestFitChartData:
    def test_one_row_per_fitted_lot(self, analogy_df, estimate_df):
        _, ctx = M.run_lot_cost_model(analogy_df, estimate_df)
        chart = M.generate_fit_chart_data(ctx)
        assert len(chart) == 6

    def test_residual_percent_agrees_with_the_costs(
        self, analogy_df, estimate_df
    ):
        _, ctx = M.run_lot_cost_model(analogy_df, estimate_df)
        chart = M.generate_fit_chart_data(ctx)
        row = chart.iloc[0]
        expected = (
            row["Actual AUC ($K)"] / row["LC Estimated AUC ($K)"] - 1.0
        ) * 100
        assert row["LC Residual (%)"] == pytest.approx(expected, abs=0.01)
