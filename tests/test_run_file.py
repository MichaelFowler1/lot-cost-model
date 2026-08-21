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
