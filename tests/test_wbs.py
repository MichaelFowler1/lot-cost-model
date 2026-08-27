"""Rolling several WBS elements into one programme estimate.

The arithmetic here is easy to get right and easy to get subtly wrong, so
these check the two things a reviewer would: that the total really is the sum
of its parts on every axis, and that each element kept its own curve rather
than being quietly blended with the others.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import wbs

FY_HIST = [2015, 2016, 2017, 2018, 2019, 2020]
FY_BUY = [2028, 2029, 2030, 2031, 2032, 2033]
AIRCRAFT = [12.0, 20.0, 30.0, 40.0, 25.0, 10.0]


def analogy(qty, auc):
    return pd.DataFrame(
        {
            "Lot": range(1, len(qty) + 1),
            "Lot FY": FY_HIST,
            "Qty": [float(q) for q in qty],
            "AUC ($K)": auc,
        }
    )


def airframe(**kw):
    return wbs.Element(
        name="1.1 Airframe",
        analogy=analogy(
            [5, 9, 14, 22, 34, 50],
            [857.91, 645.57, 531.74, 437.51, 380.10, 332.21],
        ),
        quantities=list(AIRCRAFT),
        complexity=1.15,
        **kw,
    )


def propulsion(**kw):
    # Two engines per aircraft plus a 10% spares provision.
    return wbs.Element(
        name="1.2 Propulsion",
        analogy=analogy(
            [12, 20, 30, 44, 68, 100],
            [402.10, 331.55, 288.90, 254.30, 228.75, 210.40],
        ),
        quantities=[round(q * 2 * 1.10) for q in AIRCRAFT],
        complexity=1.0,
        **kw,
    )


def avionics(**kw):
    return wbs.Element(
        name="1.3 Avionics kit",
        analogy=analogy(
            [6, 11, 16, 26, 38, 55],
            [610.00, 486.20, 421.30, 366.10, 330.55, 302.80],
        ),
        quantities=[q + 2 for q in AIRCRAFT],
        complexity=1.05,
        **kw,
    )


def program(*elements, name="TEST_PROGRAM"):
    return wbs.Program(
        name=name,
        fiscal_years=FY_BUY,
        elements=list(elements) or [airframe(), propulsion(), avionics()],
    )


@pytest.fixture(scope="module")
def rolled():
    return wbs.roll_up(program(), simulate=False)


class TestRollUpArithmetic:
    def test_total_is_the_sum_of_the_elements(self, rolled):
        assert rolled.total == pytest.approx(
            sum(e.total for e in rolled.elements), rel=1e-9
        )

    def test_total_is_also_the_sum_down_the_lots(self, rolled):
        assert rolled.by_lot["Program Total ($)"].sum() == pytest.approx(
            rolled.total, rel=1e-6
        )

    def test_every_lot_row_is_the_sum_across_elements(self, rolled):
        names = [e.name for e in rolled.elements]
        across = rolled.by_lot[names].sum(axis=1).to_numpy()
        np.testing.assert_allclose(
            across, rolled.by_lot["Program Total ($)"].to_numpy(), atol=0.02
        )

    def test_one_row_per_lot_in_the_schedule(self, rolled):
        assert len(rolled.by_lot) == len(FY_BUY)
        assert rolled.by_lot["Fiscal Year"].tolist() == FY_BUY

    def test_element_share_sums_to_one(self, rolled):
        table = wbs.element_summary(rolled)
        shares = table.loc[
            table["WBS Element"] != "PROGRAM TOTAL", "Share of Program"
        ]
        assert shares.sum() == pytest.approx(1.0, abs=1e-3)


class TestElementsStayIndependent:
    def test_each_element_gets_its_own_fit(self, rolled):
        t1s = [
            e.projections[f"{e.model} T1 First Unit Cost ($K)"].iloc[0]
            for e in rolled.elements
        ]
        assert len(set(round(t, 2) for t in t1s)) == len(t1s)

    def test_each_element_selects_its_own_model(self, rolled):
        # Nothing forces them to agree, and the summary must report each
        # element's own choice rather than one blended answer.
        assert {e.name for e in rolled.elements} == {
            "1.1 Airframe", "1.2 Propulsion", "1.3 Avionics kit"
        }
        for e in rolled.elements:
            assert e.model in ("LC", "Rate", "LC+Rate")

    def test_quantities_differ_between_elements(self, rolled):
        # The point of the design: a kit buy and a spares provision change
        # the count without changing the schedule.
        units = {
            e.name: int(e.projections["Lot Quantity"].sum())
            for e in rolled.elements
        }
        assert units["1.2 Propulsion"] > units["1.1 Airframe"]
        assert units["1.3 Avionics kit"] > units["1.1 Airframe"]

    def test_changing_one_element_leaves_the_others_alone(self):
        base = wbs.roll_up(program(), simulate=False)
        dearer = airframe()
        dearer.complexity = 2.30            # double the airframe only
        changed = wbs.roll_up(
            program(dearer, propulsion(), avionics()), simulate=False
        )
        by_name = {e.name: e.total for e in base.elements}
        for e in changed.elements:
            if e.name == "1.1 Airframe":
                assert e.total == pytest.approx(2 * by_name[e.name], rel=1e-6)
            else:
                assert e.total == pytest.approx(by_name[e.name], rel=1e-9)


class TestLotsAnElementSkips:
    def test_a_zero_quantity_lot_costs_nothing_and_keeps_its_place(self):
        late = avionics()
        late.quantities = [0.0, 0.0] + list(AIRCRAFT[2:])
        rolled = wbs.roll_up(
            program(airframe(), late), simulate=False
        )
        row = [e for e in rolled.elements if e.name == late.name][0]
        assert row.by_lot[0] == 0.0 and row.by_lot[1] == 0.0
        assert len(row.by_lot) == len(FY_BUY)
        assert row.by_lot[2] > 0

    def test_the_program_total_still_reconciles(self):
        late = avionics()
        late.quantities = [0.0, 0.0] + list(AIRCRAFT[2:])
        rolled = wbs.roll_up(program(airframe(), late), simulate=False)
        assert rolled.by_lot["Program Total ($)"].sum() == pytest.approx(
            rolled.total, rel=1e-6
        )


class TestValidation:
    def test_rejects_a_quantity_vector_of_the_wrong_length(self):
        bad = airframe()
        bad.quantities = [1.0, 2.0]
        with pytest.raises(wbs.ProgramError, match="lot quantities"):
            wbs.roll_up(program(bad), simulate=False)

    def test_rejects_duplicate_element_names(self):
        with pytest.raises(wbs.ProgramError, match="both named"):
            wbs.roll_up(program(airframe(), airframe()), simulate=False)

    def test_rejects_an_element_bought_in_no_lot(self):
        idle = airframe()
        idle.quantities = [0.0] * len(FY_BUY)
        with pytest.raises(wbs.ProgramError, match="no lot"):
            wbs.roll_up(program(idle), simulate=False)

    def test_rejects_a_negative_quantity(self):
        bad = airframe()
        bad.quantities = [-1.0] + list(AIRCRAFT[1:])
        with pytest.raises(wbs.ProgramError, match="negative"):
            wbs.roll_up(program(bad), simulate=False)

    def test_rejects_a_program_with_no_elements(self):
        with pytest.raises(wbs.ProgramError, match="at least one element"):
            wbs.roll_up(wbs.Program("P", FY_BUY, []), simulate=False)

    def test_an_element_that_cannot_be_fitted_stops_the_roll_up(self):
        # Better than a total that quietly omits an element.
        thin = airframe()
        thin.analogy = thin.analogy.head(2)
        with pytest.raises(wbs.ProgramError, match="1.1 Airframe"):
            wbs.roll_up(program(thin), simulate=False)


# The correlated roll-up needs cost_core.
risk = pytest.mark.skipif(
    not wbs.RISK_AVAILABLE,
    reason=f"cost_core not installed: {wbs.RISK_IMPORT_ERROR}",
)


@pytest.fixture(scope="module")
def simulated():
    return wbs.roll_up(program(), n_iter=8000, seed=11)


@risk
class TestProgramRisk:
    def test_percentiles_are_ordered(self, simulated):
        assert simulated.p50 < simulated.p80 < simulated.p90

    def test_point_estimate_sits_near_the_middle(self, simulated):
        assert 30 < simulated.point_percentile < 70

    def test_every_element_carries_its_own_draws(self, simulated):
        for e in simulated.elements:
            assert e.totals is not None
            assert len(e.totals) == 8000

    def test_the_program_is_wider_than_any_single_element(self, simulated):
        # A sum of correlated positives has more absolute spread than any of
        # its parts, even though its CV is smaller.
        spread = simulated.p80 - simulated.p50
        for e in simulated.elements:
            element_spread = float(
                np.percentile(e.totals, 80) - np.percentile(e.totals, 50)
            )
            assert spread > element_spread

    def test_same_seed_reproduces_the_program_p80(self):
        a = wbs.roll_up(program(), n_iter=4000, seed=5)
        b = wbs.roll_up(program(), n_iter=4000, seed=5)
        assert a.p80 == b.p80

    def test_more_correlation_widens_the_program_total(self):
        low = wbs.roll_up(program(), n_iter=8000, seed=5, correlation=0.0)
        high = wbs.roll_up(program(), n_iter=8000, seed=5, correlation=0.8)
        assert high.cv > low.cv

    def test_the_sampler_agrees_with_the_algebra_on_correlation(
        self, simulated
    ):
        # 1 + rho(k-1) is the ratio for *equally variable* elements, and
        # these are not: their CVs run from 0.014 to 0.023. So the closed
        # form lands below 1.5 rather than on it, and the real check is that
        # the measured inflation matches whatever the algebra says it is.
        analytic = simulated.variance_ratio_analytic
        assert 1.0 < analytic < 1.5
        assert simulated.independence_understates_sd_by == pytest.approx(
            np.sqrt(analytic), rel=0.05
        )

    def test_the_correlation_assumption_is_stated(self, simulated):
        text = " ".join(
            wbs.program_summary(simulated)["Value"].astype(str).tolist()
        )
        assert "0.25" in text
        assert "independen" in text.lower()

    def test_scurve_is_monotonic_and_agrees_with_the_percentiles(
        self, simulated
    ):
        sc = simulated.scurve
        assert len(sc) == 99
        assert np.all(np.diff(sc["Program Total ($)"].to_numpy()) >= 0)
        at = sc.set_index((sc["Percentile"] * 100).round().astype(int))[
            "Program Total ($)"
        ]
        assert at[80] == pytest.approx(simulated.p80, rel=1e-6)

    def test_adding_element_p80s_overstates_the_program_p80(self):
        # The whole reason this is not a column of SUMs. Checked at zero
        # correlation, where the diversification is unambiguous: at the 0.25
        # default the two sit within the half-percent the distribution
        # handoff costs, so the comparison would not mean anything.
        indep = wbs.roll_up(program(), n_iter=20000, seed=5, correlation=0.0)
        naive = sum(
            float(np.percentile(e.totals, 80)) for e in indep.elements
        )
        assert naive > indep.p80


@risk
class TestSummaries:
    def test_element_summary_ends_with_the_program_total(self, simulated):
        table = wbs.element_summary(simulated)
        assert table.iloc[-1]["WBS Element"] == "PROGRAM TOTAL"
        assert table.iloc[-1]["Cost Before Risk ($)"] == pytest.approx(
            round(simulated.total, 2)
        )

    def test_program_summary_reports_the_reserve(self, simulated):
        items = set(wbs.program_summary(simulated)["Item"])
        assert "Program P80 ($)" in items
        assert "Reserve to P80 ($)" in items


class TestWithoutCostCore:
    def test_the_point_estimate_still_rolls_up(self, monkeypatch):
        # The deterministic total must not depend on the risk library.
        monkeypatch.setattr(wbs, "RISK_AVAILABLE", False)
        rolled = wbs.roll_up(program(), simulate=True)
        assert rolled.total > 0
        assert rolled.p80 is None
        assert any("cost_core" in w for w in rolled.warnings)


@risk
class TestDistributionHandoff:
    """cost_core's WBS model takes distributions, not draws.

    Summarising each element as a lognormal to get it there costs accuracy,
    so the cost is measured and bounded rather than assumed away.
    """

    def test_the_fitted_spec_tracks_the_element_it_replaces(self, simulated):
        from scipy import stats

        for e in simulated.elements:
            spec = wbs._lognormal_spec(e.totals)
            fitted = stats.lognorm(s=spec["sigma"], scale=np.exp(spec["mean"]))
            for level in (0.05, 0.50, 0.80, 0.90, 0.95):
                empirical = float(np.percentile(e.totals, level * 100))
                assert fitted.ppf(level) == pytest.approx(
                    empirical, rel=wbs.SPEC_TOLERANCE
                ), f"{e.name} at P{level * 100:.0f}"

    def test_the_median_survives_the_handoff_almost_exactly(self, simulated):
        from scipy import stats

        for e in simulated.elements:
            spec = wbs._lognormal_spec(e.totals)
            fitted = stats.lognorm(s=spec["sigma"], scale=np.exp(spec["mean"]))
            assert fitted.ppf(0.5) == pytest.approx(
                float(np.median(e.totals)), rel=0.002
            )

    def test_the_approximation_is_disclosed(self, simulated):
        text = " ".join(simulated.notes)
        assert "lognormal" in text and "half a percent" in text


class TestManyElements:
    """No coded ceiling, but the practical limits are worth pinning."""

    def test_sheet_names_stay_unique_when_the_first_31_chars_match(self):
        # Excel truncates at 31, which is short enough for two real WBS
        # names to collide and for one element's sheet to be lost.
        taken = set()
        names = [
            "1.1 Air Vehicle Structural Assembly Group A",
            "1.1 Air Vehicle Structural Assembly Group B",
            "1.1 Air Vehicle Structural Assembly Group C",
        ]
        made = [wbs._sheet_name(n, taken) for n in names]
        assert len(set(made)) == len(names)
        assert all(len(m) <= 31 for m in made)

    def test_sheet_names_avoid_the_program_sheets(self):
        taken = {"Program_Summary", "Program_By_Lot"}
        assert wbs._sheet_name("Program_Summary", taken) != "Program_Summary"

    def test_illegal_characters_are_stripped(self):
        assert "/" not in wbs._sheet_name("1.1 Air/Ground [x]", set())
        assert "[" not in wbs._sheet_name("1.1 Air/Ground [x]", set())

    def test_a_dozen_elements_roll_up_and_reconcile(self):
        many = []
        for i in range(12):
            el = airframe()
            el.name = f"1.{i + 1} Element {i + 1}"
            # Vary the buy so they are not all the same number.
            el.quantities = [q + i for q in AIRCRAFT]
            many.append(el)
        rolled = wbs.roll_up(program(*many), simulate=False)
        assert len(rolled.elements) == 12
        assert rolled.total == pytest.approx(
            sum(e.total for e in rolled.elements), rel=1e-9
        )
        assert rolled.by_lot["Program Total ($)"].sum() == pytest.approx(
            rolled.total, rel=1e-6
        )


@risk
class TestTornado:
    def test_shares_add_to_one(self, simulated):
        # The covariance decomposition Cov(X_i, T)/Var(T) sums to exactly one,
        # which a ranking on input spread alone does not.
        assert simulated.tornado is not None
        assert simulated.tornado["variance_share"].sum() == pytest.approx(
            1.0, abs=1e-9
        )

    def test_one_row_per_element_largest_first(self, simulated):
        shares = simulated.tornado["variance_share"].to_numpy()
        assert len(simulated.tornado) == len(simulated.elements)
        assert np.all(np.diff(shares) <= 0)

    def test_ranking_need_not_match_cost_share(self, simulated):
        # The point of ranking on variance: contribution to spread is a
        # different question from size.
        by_variance = list(simulated.tornado["component"])
        by_cost = [
            e.name
            for e in sorted(
                simulated.elements, key=lambda e: e.total, reverse=True
            )
        ]
        assert set(by_variance) == set(by_cost)

    def test_absent_without_risk(self, rolled):
        assert rolled.tornado is None


@risk
class TestInfluence:
    def test_one_row_per_analogy_lot(self, rolled):
        for e in rolled.elements:
            table = wbs.influence(e)
            assert table is not None
            assert len(table) == e.n_lots_fitted

    def test_flags_are_present_and_boolean(self, rolled):
        table = wbs.influence(rolled.elements[0])
        for col in ("Leverage", "Cook's D", "High leverage", "Influential"):
            assert col in table.columns
        assert table["Influential"].dtype == bool

    def test_the_combined_table_names_its_elements(self, rolled):
        combined = wbs.influence_table(rolled)
        assert combined is not None
        assert set(combined["WBS Element"]) == {
            e.name for e in rolled.elements
        }


class TestBuySensitivity:
    def test_one_row_per_factor_with_the_baseline_at_zero_change(self):
        frame = wbs.buy_profile_sensitivity(
            program(), factors=(0.8, 1.0, 1.25)
        )
        assert len(frame) == 3
        base = frame.loc[frame["Buy Multiplier"] == 1.0].iloc[0]
        assert base["Change vs Baseline"] == pytest.approx(0.0, abs=1e-9)

    def test_buying_more_costs_more_in_total(self):
        frame = wbs.buy_profile_sensitivity(
            program(), factors=(0.6, 1.0, 1.5)
        )
        totals = frame["Program Total ($)"].to_numpy()
        assert np.all(np.diff(totals) > 0)

    def test_buying_more_costs_less_per_unit(self):
        # Learning plus the rate term: bigger lots are cheaper per unit.
        frame = wbs.buy_profile_sensitivity(
            program(), factors=(0.6, 1.0, 1.5)
        )
        per_unit = frame["Cost per Unit ($)"].to_numpy()
        assert np.all(np.diff(per_unit) < 0)

    def test_quantities_stay_whole_units(self):
        # 0.6 x 12 is 7.2, and you cannot buy 7.2 airframes.
        scaled = wbs.buy_profile_sensitivity(program(), factors=(0.6,))
        assert float(scaled[scaled.columns[1]].iloc[0]).is_integer()

    def test_the_program_is_not_modified(self):
        prog = program()
        before = [list(e.quantities) for e in prog.elements]
        wbs.buy_profile_sensitivity(prog, factors=(0.5, 2.0))
        assert [list(e.quantities) for e in prog.elements] == before

    def test_an_unknown_reference_element_is_refused(self):
        with pytest.raises(wbs.ProgramError, match="not an element"):
            wbs.buy_profile_sensitivity(
                program(), factors=(1.0,), reference_element="1.9 Nope"
            )

    def test_a_lot_an_element_sits_out_stays_at_zero(self):
        late = avionics()
        late.quantities = [0.0, 0.0] + list(AIRCRAFT[2:])
        prog = program(airframe(), late)
        frame = wbs.buy_profile_sensitivity(prog, factors=(1.4,))
        # Scaling must not conjure a buy in a lot the element skips.
        rolled = wbs.roll_up(prog, simulate=False)
        row = [e for e in rolled.elements if e.name == late.name][0]
        assert row.by_lot[0] == 0.0
        assert len(frame) == 1


def se_pm_program():
    """Hardware, engineering as a percentage of it, and one-off tooling."""
    return wbs.Program(
        name="FULL_WBS",
        fiscal_years=FY_BUY,
        elements=[
            airframe(), propulsion(), avionics(),
            wbs.factor_of("1.4 Systems Engineering", 0.08),
            wbs.factor_of(
                "1.5 Program Management", 0.05,
                basis=["1.1 Airframe", "1.2 Propulsion", "1.3 Avionics kit",
                       "1.4 Systems Engineering"],
            ),
            wbs.flat_amount("1.6 Tooling", [8e6, 4e6, 0, 0, 0, 0]),
        ],
    )


HARDWARE = ["1.1 Airframe", "1.2 Propulsion", "1.3 Avionics kit"]


class TestFactorElements:
    def test_a_factor_is_exactly_its_share_of_the_basis(self):
        res = wbs.roll_up(se_pm_program(), simulate=False)
        by = {r.name: r for r in res.elements}
        hardware = sum(by[n].total for n in HARDWARE)
        assert by["1.4 Systems Engineering"].total == pytest.approx(
            0.08 * hardware, rel=1e-9
        )

    def test_a_factor_can_sit_on_another_factor(self):
        res = wbs.roll_up(se_pm_program(), simulate=False)
        by = {r.name: r for r in res.elements}
        base = sum(by[n].total for n in HARDWARE) + by[
            "1.4 Systems Engineering"
        ].total
        assert by["1.5 Program Management"].total == pytest.approx(
            0.05 * base, rel=1e-9
        )

    def test_a_factor_is_applied_lot_by_lot_not_to_the_total(self):
        # Engineering effort follows the hardware it supports, so the
        # phasing has to match rather than being spread evenly.
        res = wbs.roll_up(se_pm_program(), simulate=False)
        by = {r.name: r for r in res.elements}
        hardware = np.sum([by[n].by_lot for n in HARDWARE], axis=0)
        np.testing.assert_allclose(
            by["1.4 Systems Engineering"].by_lot, 0.08 * hardware, rtol=1e-9
        )

    def test_an_empty_basis_means_every_fitted_element(self):
        res = wbs.roll_up(se_pm_program(), simulate=False)
        se = [r for r in res.elements
              if r.name == "1.4 Systems Engineering"][0]
        assert sorted(se.basis) == sorted(HARDWARE)

    def test_the_model_column_says_what_it_is_a_percentage_of(self):
        res = wbs.roll_up(se_pm_program(), simulate=False)
        se = [r for r in res.elements
              if r.name == "1.4 Systems Engineering"][0]
        assert "8.0%" in se.model
        assert "1.1 Airframe" in se.model

    def test_totals_still_reconcile_with_derived_elements_present(self):
        res = wbs.roll_up(se_pm_program(), simulate=False)
        assert res.total == pytest.approx(
            sum(e.total for e in res.elements), rel=1e-9
        )
        assert res.by_lot["Program Total ($)"].sum() == pytest.approx(
            res.total, rel=1e-6
        )


class TestAmountElements:
    def test_it_costs_exactly_what_was_entered(self):
        res = wbs.roll_up(se_pm_program(), simulate=False)
        by = {r.name: r for r in res.elements}
        assert by["1.6 Tooling"].total == pytest.approx(12e6)
        np.testing.assert_allclose(
            by["1.6 Tooling"].by_lot, [8e6, 4e6, 0, 0, 0, 0]
        )

    def test_it_is_not_dragged_into_a_factor_by_default(self):
        # The default basis is the fitted elements, so tooling is not
        # silently swept into an engineering percentage.
        res = wbs.roll_up(se_pm_program(), simulate=False)
        se = [r for r in res.elements
              if r.name == "1.4 Systems Engineering"][0]
        assert "1.6 Tooling" not in se.basis

    def test_the_wrong_number_of_amounts_is_refused(self):
        prog = se_pm_program()
        prog.elements[-1].amounts = [1.0, 2.0]
        with pytest.raises(wbs.ProgramError, match="lot amounts"):
            wbs.roll_up(prog, simulate=False)


class TestKindValidation:
    def test_a_factor_element_needs_a_factor(self):
        prog = se_pm_program()
        prog.elements[3].factor = None
        with pytest.raises(wbs.ProgramError, match="needs a factor"):
            wbs.roll_up(prog, simulate=False)

    def test_a_basis_naming_an_unknown_element_is_refused(self):
        prog = se_pm_program()
        prog.elements[3].basis = ["1.9 Imaginary"]
        with pytest.raises(wbs.ProgramError, match="not.*an element"):
            wbs.roll_up(prog, simulate=False)

    def test_an_element_cannot_be_a_percentage_of_itself(self):
        prog = se_pm_program()
        prog.elements[3].basis = ["1.4 Systems Engineering"]
        with pytest.raises(wbs.ProgramError, match="percentage of itself"):
            wbs.roll_up(prog, simulate=False)

    def test_a_circle_between_factors_is_refused(self):
        prog = se_pm_program()
        prog.elements[3].basis = ["1.5 Program Management"]
        prog.elements[4].basis = ["1.4 Systems Engineering"]
        with pytest.raises(wbs.ProgramError, match="circle"):
            wbs.roll_up(prog, simulate=False)

    def test_a_program_of_only_derived_elements_is_refused(self):
        prog = wbs.Program(
            name="P", fiscal_years=FY_BUY,
            elements=[wbs.flat_amount("Only tooling", [1e6] * 6)],
        )
        with pytest.raises(wbs.ProgramError, match="no fitted element"):
            wbs.roll_up(prog, simulate=False)

    def test_an_unknown_kind_is_refused(self):
        prog = se_pm_program()
        prog.elements[-1].kind = "guesswork"
        with pytest.raises(wbs.ProgramError, match="expected one of"):
            wbs.roll_up(prog, simulate=False)


@pytest.fixture(scope="module")
def full():
    return wbs.roll_up(se_pm_program(), n_iter=8000, seed=11)


@risk
class TestDerivedElementsUnderRisk:
    """Derived kinds inherit uncertainty; they never invent it."""

    def test_a_factor_moves_perfectly_with_its_basis(self, full):
        # It is a fixed percentage on every iteration, so the correlation is
        # 1 by construction rather than by assumption.
        by = {r.name: r for r in full.elements}
        basis = np.sum([by[n].totals for n in HARDWARE], axis=0)
        se = by["1.4 Systems Engineering"].totals
        assert np.corrcoef(basis, se)[0, 1] == pytest.approx(1.0, abs=1e-9)

    def test_a_factor_keeps_its_share_on_every_draw(self, full):
        by = {r.name: r for r in full.elements}
        basis = np.sum([by[n].totals for n in HARDWARE], axis=0)
        ratio = by["1.4 Systems Engineering"].totals / basis
        assert np.allclose(ratio, ratio[0], rtol=1e-9)

    def test_an_amount_does_not_move_at_all(self, full):
        by = {r.name: r for r in full.elements}
        tooling = by["1.6 Tooling"].totals
        assert np.std(tooling) == pytest.approx(0.0, abs=1e-6)
        assert np.allclose(tooling, 12e6)

    def test_an_amount_contributes_no_variance(self, full):
        row = full.tornado.loc[
            full.tornado["component"] == "1.6 Tooling"
        ].iloc[0]
        assert row["variance_share"] == pytest.approx(0.0, abs=1e-12)

    def test_the_tornado_covers_every_kind_and_still_sums_to_one(self, full):
        assert set(full.tornado["kind"]) == {"fitted", "factor", "amount"}
        assert len(full.tornado) == len(full.elements)
        assert full.tornado["variance_share"].sum() == pytest.approx(
            1.0, abs=1e-9
        )

    def test_an_amount_still_lands_in_every_percentile(self, full):
        # It does not move, but it is still money and must be in the total.
        without = wbs.roll_up(
            wbs.Program(
                name="no tooling",
                fiscal_years=FY_BUY,
                elements=[e for e in se_pm_program().elements
                          if e.kind != "amount"],
            ),
            n_iter=8000, seed=11,
        )
        assert full.p80 - without.p80 == pytest.approx(12e6, rel=0.02)

    def test_percentiles_are_still_ordered(self, full):
        assert full.p50 < full.p80 < full.p90

    def test_the_derived_kinds_are_disclosed(self, full):
        text = " ".join(full.notes)
        assert "percentage of" in text and "spread of its own" in text


class TestSensitivityAcrossKinds:
    """Scaling a buy has to respect what each kind actually is."""

    def test_it_runs_at_all_with_derived_elements_present(self):
        # It did not: scaling walked every element's quantities, and a factor
        # element has none, so the whole sensitivity died on a TypeError.
        frame = wbs.buy_profile_sensitivity(
            se_pm_program(), factors=(0.6, 1.0, 1.5)
        )
        assert len(frame) == 3

    def test_a_factor_keeps_its_share_at_every_buy_size(self):
        for f in (0.6, 1.0, 1.5):
            scaled = wbs.Program(
                "P", list(FY_BUY),
                [wbs._scale_element(e, f) for e in se_pm_program().elements],
            )
            res = wbs.roll_up(scaled, simulate=False)
            by = {r.name: r for r in res.elements}
            hardware = sum(by[n].total for n in HARDWARE)
            assert by["1.4 Systems Engineering"].total == pytest.approx(
                0.08 * hardware, rel=1e-9
            )

    def test_a_nonrecurring_amount_does_not_scale_with_the_buy(self):
        # Buying 40% fewer aircraft does not buy 40% less tooling. Scaling it
        # would turn a one-off into a variable cost and flatter every
        # small-buy case.
        for f in (0.6, 1.0, 1.5):
            scaled = wbs._scale_element(
                wbs.flat_amount("1.6 Tooling", [8e6, 4e6, 0, 0, 0, 0]), f
            )
            assert scaled.amounts == [8e6, 4e6, 0, 0, 0, 0]

    def test_only_fitted_quantities_move(self):
        big = [wbs._scale_element(e, 2.0) for e in se_pm_program().elements]
        by = {e.name: e for e in big}
        assert by["1.1 Airframe"].quantities == [
            q * 2 for q in AIRCRAFT
        ]
        assert by["1.4 Systems Engineering"].quantities is None
        assert by["1.6 Tooling"].amounts == [8e6, 4e6, 0, 0, 0, 0]

    def test_unit_cost_still_falls_as_the_buy_grows(self):
        frame = wbs.buy_profile_sensitivity(
            se_pm_program(), factors=(0.6, 1.0, 1.5)
        )
        per_unit = frame["Cost per Unit ($)"].to_numpy()
        assert np.all(np.diff(per_unit) < 0)

    def test_fixed_tooling_makes_small_buys_worse_per_unit(self):
        # A one-off spread over fewer units costs more each, on top of the
        # learning and rate effects. Its share should rise as the buy shrinks.
        small = wbs.roll_up(
            wbs.Program("P", list(FY_BUY),
                        [wbs._scale_element(e, 0.6)
                         for e in se_pm_program().elements]),
            simulate=False,
        )
        large = wbs.roll_up(
            wbs.Program("P", list(FY_BUY),
                        [wbs._scale_element(e, 1.5)
                         for e in se_pm_program().elements]),
            simulate=False,
        )

        def share(res):
            by = {r.name: r for r in res.elements}
            return by["1.6 Tooling"].total / res.total

        assert share(small) > share(large)
