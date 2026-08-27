"""Saved runs and the provenance stamp.

A saved run is only worth having if reloading it reproduces the same numbers,
so the round-trip test runs the model on both sides and compares the
projections rather than just comparing the JSON to itself.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import lot_cost_model as M


class TestProvenance:
    def test_names_the_version(self):
        assert provenance_value("Tool version").startswith(M.TOOL_VERSION)

    def test_stamps_a_timestamp(self):
        stamp = provenance_value("Run timestamp")
        # Parses back as a date, so it is a real timestamp and not a label.
        assert pd.to_datetime(stamp[:19]) is not None

    def test_says_the_rate_projection_is_corrected_by_default(self):
        assert "corrected" in provenance_value("Rate projection")

    def test_calls_out_a_legacy_run_in_capitals(self):
        legacy = M.provenance({"LegacyRateOmission": True})
        assert "LEGACY" in legacy["Rate projection"]
        assert "overstated" in legacy["Rate projection"]

    def test_reaches_the_analyst_summary(self, analogy_df, estimate_df):
        _, ctx = M.run_lot_cost_model(analogy_df, estimate_df)
        summary = M.generate_analyst_summary(ctx, {"Program": "TEST"})
        items = set(summary["Item"])
        assert {"Tool version", "Run timestamp", "Rate projection"} <= items

    def test_the_summary_records_a_legacy_run_as_legacy(
        self, analogy_df, estimate_df
    ):
        # The whole point: a workbook must be able to say what produced it.
        _, ctx = M.run_lot_cost_model(
            analogy_df, estimate_df, {"LegacyRateOmission": True}
        )
        summary = M.generate_analyst_summary(ctx, {"Program": "TEST"})
        row = summary.loc[summary["Item"] == "Rate projection", "Value"].iloc[0]
        assert "LEGACY" in row

    def test_version_is_no_longer_hardcoded_in_settings(self):
        # It used to read "2.0-dev" for every build ever released, which is
        # why an old workbook cannot be dated from the inside.
        assert M.SETTINGS["ToolVersion"] is None


def provenance_value(item: str) -> str:
    return M.provenance()[item]


class TestRunFileValidation:
    def test_rejects_a_file_that_is_not_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("this is not json", encoding="utf-8")
        with pytest.raises(M.RunFileError, match="not a saved run"):
            M.read_run_file(str(path))

    def test_rejects_json_that_is_not_a_run(self, tmp_path):
        path = tmp_path / "other.json"
        path.write_text('{"hello": "world"}', encoding="utf-8")
        with pytest.raises(M.RunFileError, match="not a lot cost model run"):
            M.read_run_file(str(path))

    def test_rejects_a_newer_format(self, tmp_path):
        path = tmp_path / "future.json"
        path.write_text(
            json.dumps(
                {
                    "format": M.RUN_FORMAT,
                    "format_version": M.RUN_FORMAT_VERSION + 5,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(M.RunFileError, match="Update the tool"):
            M.read_run_file(str(path))

    def test_reports_a_missing_file_clearly(self, tmp_path):
        with pytest.raises(M.RunFileError, match="Could not open"):
            M.read_run_file(str(tmp_path / "nope.json"))

    def test_accepts_the_current_format(self, tmp_path):
        path = tmp_path / "ok.json"
        path.write_text(
            json.dumps(
                {
                    "format": M.RUN_FORMAT,
                    "format_version": M.RUN_FORMAT_VERSION,
                    "analogy_lots": [],
                }
            ),
            encoding="utf-8",
        )
        assert M.read_run_file(str(path))["format"] == M.RUN_FORMAT


# The GUI round-trip needs a display, so it is skipped where there is none.
tk = pytest.importorskip("tkinter")


@pytest.fixture(scope="module")
def app():
    """One window for the whole module.

    Tkinter allows a single root per process, and building and tearing one
    down for every test turned out to fail intermittently on Windows, which
    showed up as tests quietly skipping rather than running. Each test wipes
    the window instead.
    """
    try:
        instance = M.LotCostApp()
    except tk.TclError as exc:  # pragma: no cover - headless CI
        pytest.skip(f"no display: {exc}")
    instance.withdraw()
    yield instance
    instance.destroy()


def wipe(app):
    """Blank the window, so reloading has to restore everything itself.

    Tkinter supports one root per process, so these tests reuse a single
    window rather than building a second one to load into. Clearing it first
    is also the stronger check: a fresh window could pass by defaulting to
    the same values the run happened to hold.
    """
    app.grid_analogy.clear()
    app.grid_estimate.clear()
    app.grid_analogy.add_row()
    app.grid_estimate.add_row()
    for var in (app.var_runid, app.var_program, app.var_label,
                app.var_baseyear):
        var.set("")
    for var in (app.var_costscale, app.var_totalscale, app.var_defaultcf,
                app.var_tgate, app.var_fitprior, app.var_fcstprior):
        var.set("0")
    app.var_legacy_rate.set(False)


class TestRoundTrip:
    def test_a_reloaded_run_reproduces_the_same_projections(
        self, app, tmp_path
    ):
        app.grid_analogy.load(M.EXAMPLE_ANALOGY)
        app.grid_estimate.load(M.EXAMPLE_ESTIMATE)
        app.var_program.set("ROUNDTRIP")
        app.var_baseyear.set("2025")
        app.var_fcstprior.set("40")
        app.var_tgate.set("1.5")

        before, _ = M.run_lot_cost_model(
            app._collect_analogy(),
            app._collect_estimate(),
            app._collect_overrides(),
        )

        path = tmp_path / "run.json"
        path.write_text(json.dumps(app.run_state()), encoding="utf-8")

        wipe(app)
        app.apply_run_state(M.read_run_file(str(path)))
        after, _ = M.run_lot_cost_model(
            app._collect_analogy(),
            app._collect_estimate(),
            app._collect_overrides(),
        )

        pd.testing.assert_frame_equal(before, after)

    def test_carries_the_run_info_and_settings(self, app):
        app.grid_analogy.load(M.EXAMPLE_ANALOGY)
        app.grid_estimate.load(M.EXAMPLE_ESTIMATE)
        app.var_runid.set("R042")
        app.var_program.set("CARRIED")
        app.var_label.set("a label")
        app.var_baseyear.set("2031")
        app.var_fitprior.set("7")

        state = app.run_state()
        wipe(app)
        app.apply_run_state(state)

        assert app.var_runid.get() == "R042"
        assert app.var_program.get() == "CARRIED"
        assert app.var_label.get() == "a label"
        assert app.var_baseyear.get() == "2031"
        assert app.var_fitprior.get() == "7"

    def test_quantity_only_lots_survive_the_round_trip(self, app):
        lots = [("2015", "10", "800.61"), ("2016", "20", ""),
                ("2017", "25", "563.66"), ("2018", "25", "520.05")]
        app.grid_analogy.load(lots)
        app.grid_estimate.load(M.EXAMPLE_ESTIMATE)
        state = app.run_state()
        assert state["analogy_lots"][1] == ["2016", "20", ""]

        wipe(app)
        app.apply_run_state(state)
        assert app.grid_analogy.get_rows() == [list(r) for r in lots]

    def test_a_version_one_file_loads_as_a_single_element(self, app):
        # Written before elements existed. It must open rather than be
        # refused, and become one element.
        legacy = {
            "format": M.RUN_FORMAT,
            "format_version": 1,
            "analogy_lots": [list(r) for r in M.EXAMPLE_ANALOGY],
            "estimate_lots": [list(r) for r in M.EXAMPLE_ESTIMATE],
            "run_info": {"Program": "LEGACY"},
        }
        wipe(app)
        app.apply_run_state(legacy)
        assert len(app.elements) == 1
        assert app.grid_analogy.get_rows() == [
            list(r) for r in M.EXAMPLE_ANALOGY
        ]

    def test_the_saved_file_is_self_describing(self, app):
        app.grid_analogy.load(M.EXAMPLE_ANALOGY)
        app.grid_estimate.load(M.EXAMPLE_ESTIMATE)
        state = app.run_state()
        assert state["format"] == M.RUN_FORMAT
        assert state["format_version"] == M.RUN_FORMAT_VERSION
        assert state["tool_version"] == M.TOOL_VERSION
        assert state["saved_at"]

    def test_a_legacy_run_reloads_as_legacy_and_warns(self, app, monkeypatch):
        # Restoring the setting silently would reproduce overstated costs
        # without telling anybody.
        app.grid_analogy.load(M.EXAMPLE_ANALOGY)
        app.grid_estimate.load(M.EXAMPLE_ESTIMATE)
        app.var_legacy_rate.set(True)
        state = app.run_state()
        assert state["settings"]["LegacyRateOmission"] is True

        warned = []
        monkeypatch.setattr(
            M.messagebox, "showwarning", lambda t, m: warned.append(t)
        )
        wipe(app)
        app.apply_run_state(state)

        assert app.var_legacy_rate.get() is True
        assert warned, "reloading a legacy run must say so"


class TestElementManagement:
    """Tabs 1 to 5 always show one element; the window holds several."""

    def test_starts_with_a_single_element(self, app):
        wipe(app)
        assert len(app.elements) >= 1

    def test_switching_elements_keeps_each_ones_lots(self, app):
        wipe(app)
        app.elements = [app._blank_element("A")]
        app._refresh_element_list(0)

        app.grid_analogy.load([("2015", "5", "857.91")])
        app.grid_estimate.load([("2028", "12", "1.0")])
        app._capture_element()

        app.elements.append(app._blank_element("B"))
        app._refresh_element_list(1)
        app.grid_analogy.load([("2016", "9", "645.57")])
        app.grid_estimate.load([("2028", "20", "1.0")])
        app._capture_element()

        # Back to A: its own lots, not B's.
        app._refresh_element_list(0)
        assert app.grid_analogy.get_rows() == [["2015", "5", "857.91"]]
        app._refresh_element_list(1)
        assert app.grid_analogy.get_rows() == [["2016", "9", "645.57"]]

    def test_a_new_element_inherits_the_schedule_but_no_quantities(
        self, app, monkeypatch
    ):
        wipe(app)
        app.elements = [app._blank_element("A")]
        app._refresh_element_list(0)
        app.grid_estimate.load(
            [("2028", "12", "1.15"), ("2029", "20", "1.15")]
        )
        app._capture_element()

        # add_element now asks for the kind first, and both dialogs are
        # modal, so both have to be answered for the test to return.
        monkeypatch.setattr(app, "_ask_kind", lambda: "fitted")
        monkeypatch.setattr(app, "_ask_name", lambda *a, **k: "Propulsion")
        app.add_element()

        rows = app.grid_estimate.get_rows()
        assert [r[0] for r in rows] == ["2028", "2029"]
        assert all(r[1] == "" for r in rows), (
            "a new element should start with the years but no counts"
        )

    def test_duplicate_names_are_made_unique(self, app, monkeypatch):
        wipe(app)
        app.elements = [app._blank_element("Airframe")]
        app._refresh_element_list(0)
        monkeypatch.setattr(app, "_ask_kind", lambda: "fitted")
        monkeypatch.setattr(app, "_ask_name", lambda *a, **k: "Airframe")
        app.add_element()
        assert len({e["name"] for e in app.elements}) == len(app.elements)

    def test_the_last_element_cannot_be_removed(self, app, monkeypatch):
        wipe(app)
        app.elements = [app._blank_element("Only")]
        app._refresh_element_list(0)
        told = []
        monkeypatch.setattr(
            M.messagebox, "showinfo", lambda t, m: told.append(t)
        )
        app.remove_element()
        assert len(app.elements) == 1
        assert told

    def test_every_element_survives_the_round_trip(self, app, monkeypatch):
        wipe(app)
        app.elements = [
            {
                "name": "1.1 Airframe",
                "analogy": [["2015", "5", "857.91"], ["2016", "9", "645.57"]],
                "estimate": [["2028", "12", "1.15"]],
            },
            {
                "name": "1.2 Propulsion",
                "analogy": [["2015", "12", "402.10"]],
                "estimate": [["2028", "26", "1.0"]],
            },
        ]
        app._refresh_element_list(0)

        state = app.run_state()
        assert len(state["elements"]) == 2

        wipe(app)
        app.elements = [app._blank_element("wiped")]
        app._refresh_element_list(0)
        app.apply_run_state(state)

        assert [e["name"] for e in app.elements] == [
            "1.1 Airframe", "1.2 Propulsion"
        ]
        assert app.elements[1]["analogy"] == [["2015", "12", "402.10"]]


class TestRollUpFillsTheResultsTab:
    """A roll-up prices every element, so tab 4 should not claim otherwise."""

    def _two_elements(self, app):
        fy = ["2028", "2029", "2030", "2031", "2032", "2033"]
        ac = [12, 20, 30, 40, 25, 10]
        hist = list(range(2015, 2021))

        def rows(q, a):
            return [[str(y), str(x), f"{c:.2f}"]
                    for y, x, c in zip(hist, q, a)]

        app.elements = [
            {"name": "1.1 Airframe",
             "analogy": rows([5, 9, 14, 22, 34, 50],
                             [857.91, 645.57, 531.74, 437.51, 380.10,
                              332.21]),
             "estimate": [[f, str(q), "1.15"] for f, q in zip(fy, ac)]},
            {"name": "1.2 Propulsion",
             "analogy": rows([12, 20, 30, 44, 68, 100],
                             [402.10, 331.55, 288.90, 254.30, 228.75,
                              210.40]),
             "estimate": [[f, str(round(q * 2 * 1.1)), "1.0"]
                          for f, q in zip(fy, ac)]},
        ]
        app._refresh_element_list(0)

    def test_a_roll_up_fills_the_results_tab_for_the_selected_element(
        self, app, tmp_path
    ):
        if M.wbs is None:
            pytest.skip("wbs.py not importable")
        wipe(app)
        self._two_elements(app)
        app.var_program.set("DEMO")
        app.var_outfile.set(str(tmp_path / "out.xlsx"))
        app.var_prog_risk.set(False)

        assert len(app.tree.get_children()) == 0
        app.run_program()
        assert len(app.tree.get_children()) > 0
        assert "1.1 Airframe" in app.lbl_result.cget("text")

    def test_it_does_not_steal_the_tab(self, app, tmp_path):
        if M.wbs is None:
            pytest.skip("wbs.py not importable")
        wipe(app)
        self._two_elements(app)
        app.var_outfile.set(str(tmp_path / "out.xlsx"))
        app.var_prog_risk.set(False)
        app.run_program()
        # The roll-up is what was asked for, so that is what stays on screen.
        assert app.nb.tab(app.nb.select(), "text").strip().startswith("6.")

    def test_it_follows_the_selected_element(self, app, tmp_path):
        if M.wbs is None:
            pytest.skip("wbs.py not importable")
        wipe(app)
        self._two_elements(app)
        app.var_outfile.set(str(tmp_path / "out.xlsx"))
        app.var_prog_risk.set(False)
        app._refresh_element_list(1)
        app.run_program()
        assert "1.2 Propulsion" in app.lbl_result.cget("text")


class TestElementKindsInTheWindow:
    """The three kinds have to survive the window, not just the engine."""

    def test_a_new_element_carries_the_kind_that_was_chosen(
        self, app, monkeypatch
    ):
        wipe(app)
        app.elements = [app._blank_element("1.1 Airframe")]
        app._refresh_element_list(0)
        monkeypatch.setattr(app, "_ask_kind", lambda: "factor")
        monkeypatch.setattr(app, "_ask_name", lambda *a, **k: "1.4 SE")
        app.add_element()
        assert app.elements[-1]["kind"] == "factor"

    def test_the_kind_bar_follows_the_selected_element(self, app):
        wipe(app)
        app.elements = [
            app._blank_element("1.1 Airframe"),
            app._blank_element("1.4 SE", "factor"),
            app._blank_element("1.6 Tooling", "amount"),
        ]
        app._refresh_element_list(0)
        assert "analogy lots" in app.var_kind_note.get()
        app._refresh_element_list(1)
        assert "Factor" in app.var_kind_note.get()
        app._refresh_element_list(2)
        assert "no curve" in app.var_kind_note.get()

    def test_kinds_survive_the_save_and_reload(self, app):
        wipe(app)
        app.elements = [
            {"name": "1.1 Airframe", "kind": "fitted",
             "analogy": [list(r) for r in M.EXAMPLE_ANALOGY],
             "estimate": [list(r) for r in M.EXAMPLE_ESTIMATE],
             "factor": 0.08, "basis": []},
            {"name": "1.4 SE", "kind": "factor", "analogy": [],
             "estimate": [], "factor": 0.12,
             "basis": ["1.1 Airframe"]},
        ]
        app._refresh_element_list(0)
        state = app.run_state()

        wipe(app)
        app.elements = [app._blank_element("wiped")]
        app._refresh_element_list(0)
        app.apply_run_state(state)

        assert [e["kind"] for e in app.elements] == ["fitted", "factor"]
        se = app.elements[1]
        assert se["factor"] == pytest.approx(0.12)
        assert se["basis"] == ["1.1 Airframe"]

    def test_a_program_of_the_three_kinds_prices_from_the_window(self, app):
        if M.wbs is None:
            pytest.skip("wbs.py not importable")
        wipe(app)
        fy = [r[0] for r in M.EXAMPLE_ESTIMATE]
        app.elements = [
            {"name": "1.1 Airframe", "kind": "fitted",
             "analogy": [list(r) for r in M.EXAMPLE_ANALOGY],
             "estimate": [list(r) for r in M.EXAMPLE_ESTIMATE],
             "factor": 0.08, "basis": []},
            {"name": "1.4 SE", "kind": "factor", "analogy": [],
             "estimate": [], "factor": 0.10, "basis": []},
            {"name": "1.6 Tooling", "kind": "amount", "analogy": [],
             "estimate": [[f, "1000000", ""] for f in fy],
             "factor": 0.08, "basis": []},
        ]
        app._refresh_element_list(0)

        program = app.build_program()
        assert [e.kind for e in program.elements] == [
            "fitted", "factor", "amount"
        ]
        rolled = M.wbs.roll_up(program, simulate=False)
        by = {r.name: r for r in rolled.elements}
        assert by["1.4 SE"].total == pytest.approx(
            0.10 * by["1.1 Airframe"].total, rel=1e-9
        )
        assert by["1.6 Tooling"].total == pytest.approx(6_000_000.0)

    def test_a_factor_needs_no_lots_of_its_own(self, app):
        # It has no schedule to contribute, so it must not be asked for one.
        if M.wbs is None:
            pytest.skip("wbs.py not importable")
        wipe(app)
        app.elements = [
            {"name": "1.1 Airframe", "kind": "fitted",
             "analogy": [list(r) for r in M.EXAMPLE_ANALOGY],
             "estimate": [list(r) for r in M.EXAMPLE_ESTIMATE],
             "factor": 0.08, "basis": []},
            {"name": "1.4 SE", "kind": "factor", "analogy": [],
             "estimate": [], "factor": 0.08, "basis": []},
        ]
        app._refresh_element_list(0)
        program = app.build_program()          # must not raise
        assert len(program.fiscal_years) == len(M.EXAMPLE_ESTIMATE)


class TestResultsFollowTheSelectedElement:
    """Tabs 4 and 5 show one element, so they must track the element bar."""

    def _three(self, app):
        fy = ["2028", "2029", "2030", "2031", "2032", "2033"]
        ac = [12, 20, 30, 40, 25, 10]
        hist = list(range(2015, 2021))

        def rows(q, a):
            return [[str(y), str(x), f"{c:.2f}"]
                    for y, x, c in zip(hist, q, a)]

        app.elements = [
            {"name": "1.1 Airframe", "kind": "fitted",
             "analogy": rows([5, 9, 14, 22, 34, 50],
                             [857.91, 645.57, 531.74, 437.51, 380.10,
                              332.21]),
             "estimate": [[f, str(q), "1.15"] for f, q in zip(fy, ac)],
             "factor": 0.08, "basis": []},
            {"name": "1.2 Propulsion", "kind": "fitted",
             "analogy": rows([12, 20, 30, 44, 68, 100],
                             [402.10, 331.55, 288.90, 254.30, 228.75,
                              210.40]),
             "estimate": [[f, str(round(q * 2 * 1.1)), "1.0"]
                          for f, q in zip(fy, ac)],
             "factor": 0.08, "basis": []},
            {"name": "1.4 SE", "kind": "factor", "analogy": [],
             "estimate": [], "factor": 0.08, "basis": []},
        ]
        app._refresh_element_list(0)

    def test_switching_element_updates_the_results_tab(self, app, tmp_path):
        if M.wbs is None:
            pytest.skip("wbs.py not importable")
        wipe(app)
        self._three(app)
        app.var_outfile.set(str(tmp_path / "o.xlsx"))
        app.var_prog_risk.set(False)
        app.run_program()
        assert "1.1 Airframe" in app.lbl_result.cget("text")

        app._refresh_element_list(1)
        # It used to keep naming Airframe while Propulsion was selected,
        # which is worse than stale: the heading and the table disagreed.
        assert "1.2 Propulsion" in app.lbl_result.cget("text")
        assert "1.1 Airframe" not in app.lbl_result.cget("text")
        assert len(app.tree.get_children()) > 0

    def test_a_factor_element_shows_no_fit_statistics(self, app, tmp_path):
        if M.wbs is None:
            pytest.skip("wbs.py not importable")
        wipe(app)
        self._three(app)
        app.var_outfile.set(str(tmp_path / "o.xlsx"))
        app.var_prog_risk.set(False)
        app.run_program()

        app._refresh_element_list(2)
        assert len(app.tree.get_children()) == 0
        text = app.lbl_result.cget("text")
        assert "1.4 SE" in text and "no curve" in text

    def test_an_element_with_no_run_says_so(self, app, tmp_path):
        if M.wbs is None:
            pytest.skip("wbs.py not importable")
        wipe(app)
        self._three(app)
        app.var_outfile.set(str(tmp_path / "o.xlsx"))
        app.var_prog_risk.set(False)
        app.run_program()

        app.elements.append(app._blank_element("1.9 Added later"))
        app._refresh_element_list(3)
        assert len(app.tree.get_children()) == 0
        assert "Nothing run for 1.9 Added later" in app.lbl_result.cget("text")

    def test_switching_clears_another_elements_intervals(self, app, tmp_path):
        if M.wbs is None or M.risk is None or not M.risk.AVAILABLE:
            pytest.skip("cost_core not installed")
        wipe(app)
        self._three(app)
        app.var_outfile.set(str(tmp_path / "o.xlsx"))
        app.var_baseyear.set("2025")
        app.run_risk()
        assert len(app.tree_risk.get_children()) > 0
        assert app.risk_result_element == "1.1 Airframe"

        app._refresh_element_list(1)
        # Those intervals belong to Airframe; leaving them up under a
        # Propulsion heading would invite reading one for the other.
        assert len(app.tree_risk.get_children()) == 0
        assert app.risk_result is None
