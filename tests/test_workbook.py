"""The Excel output.

Most of these exist because of failures that produced a perfectly valid
workbook with something missing from the charts. openpyxl raises nothing when
you set an attribute a class does not have, so the only way to know a chart
feature survived is to read it back out of the XML.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

import openpyxl
import pytest

from cost_core.reporting import lot_workbook as LW

import lot_cost_model as M
import risk as R

ROOT = pathlib.Path(M.__file__).resolve().parent


@pytest.fixture(scope="module")
def plain_book(tmp_path_factory, analogy_df, estimate_df):
    """A workbook with no risk analysis."""
    path = tmp_path_factory.mktemp("wb") / "plain.xlsx"
    proj, ctx = M.run_lot_cost_model(analogy_df, estimate_df)
    M.save_complete_excel_workbook(
        str(path),
        proj,
        M.generate_analyst_summary(ctx, {"Program": "TEST"}),
        M.generate_fit_chart_data(ctx),
    )
    return path


@pytest.fixture(scope="module")
def risk_book(tmp_path_factory, analogy_df, estimate_df, cfg):
    """A workbook including the risk sheets."""
    path = tmp_path_factory.mktemp("wb") / "risk.xlsx"
    proj, ctx = M.run_lot_cost_model(analogy_df, estimate_df)
    summary = M.generate_analyst_summary(ctx, {"Program": "TEST"})
    res = R.run_risk(
        ctx, proj, summary, R.RiskOptions(n_iter=4000, seed=11)
    )
    M.save_complete_excel_workbook(
        str(path),
        proj,
        summary,
        M.generate_fit_chart_data(ctx),
        R.summary_frame(res),
        res.intervals,
        res.scurve,
    )
    return path


def chart_xml(path) -> list[str]:
    with zipfile.ZipFile(path) as z:
        names = sorted(
            n for n in z.namelist()
            if "charts/chart" in n and n.endswith(".xml")
        )
        return [z.read(n).decode("utf-8") for n in names]


# openpyxl serialises through lxml when it is installed and the standard
# library otherwise, and the two differ on trivia: <val v="1"/> against
# <val v="1" />. Assert on parsed elements so a missing lxml cannot fail a
# test about chart content.
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def elements(xml: str, name: str, within: str | None = None):
    """Every element with this local tag name, optionally scoped to a parent."""
    root = ET.fromstring(xml)
    scopes = (
        [el for el in root.iter() if _local(el.tag) == within]
        if within
        else [root]
    )
    found = []
    for scope in scopes:
        found += [el for el in scope.iter() if _local(el.tag) == name]
    return found


def flag(xml: str, name: str, within: str | None = None) -> list[str | None]:
    """The `val` attribute of each matching element."""
    return [el.get("val") for el in elements(xml, name, within)]


class TestSheets:
    def test_three_sheets_without_risk(self, plain_book):
        assert openpyxl.load_workbook(plain_book).sheetnames == [
            "Analyst_Summary",
            "Estimate_Projections",
            "Fit_Chart_Data",
        ]

    def test_six_sheets_with_risk(self, risk_book):
        assert openpyxl.load_workbook(risk_book).sheetnames == [
            "Analyst_Summary",
            "Estimate_Projections",
            "Fit_Chart_Data",
            "Risk_Summary",
            "Risk_Intervals",
            "Risk_SCurve",
        ]


class TestChartsExist:
    def test_three_fit_charts(self, plain_book):
        wb = openpyxl.load_workbook(plain_book)
        assert len(wb["Fit_Chart_Data"]._charts) == 3

    def test_risk_sheets_each_carry_a_chart(self, risk_book):
        wb = openpyxl.load_workbook(risk_book)
        assert len(wb["Risk_Intervals"]._charts) == 1
        assert len(wb["Risk_SCurve"]._charts) == 1


class TestAxesAreVisible:
    """openpyxl writes delete="1" on a new axis, which hides it in Excel."""

    def test_no_axis_is_marked_deleted(self, plain_book):
        for xml in chart_xml(plain_book):
            deletes = flag(xml, "delete")
            assert deletes, "no delete flag written on either axis"
            assert "1" not in deletes
            assert len([d for d in deletes if d == "0"]) >= 2

    def test_tick_labels_are_positioned(self, plain_book):
        for xml in chart_xml(plain_book):
            assert "nextTo" in flag(xml, "tickLblPos")

    def test_titles_do_not_overlay_the_plot(self, plain_book):
        for xml in chart_xml(plain_book):
            assert "0" in flag(xml, "overlay")


class TestDataLabels:
    """A Series has no `dataLabels` alias, only `dLbls`. Assigning to the
    wrong one fails silently and draws nothing, which is what shipped."""

    def test_actual_auc_labels_reach_the_xml(self, plain_book):
        for xml in chart_xml(plain_book):
            assert elements(xml, "dLbls"), "no data labels written"
            assert flag(xml, "showVal", within="dLbls") == ["1"]

    def test_labels_do_not_also_print_the_series_name(self, plain_book):
        # Every show flag is written explicitly; Excel treats an absent flag
        # as inherited rather than false.
        xml = chart_xml(plain_book)[0]
        assert flag(xml, "showSerName", within="dLbls") == ["0"]
        assert flag(xml, "showLegendKey", within="dLbls") == ["0"]
        assert flag(xml, "showCatName", within="dLbls") == ["0"]

    def test_labels_are_shrunk_to_fit(self, plain_book):
        # 8pt, so the rate chart stays legible where lots of equal quantity
        # sit almost on top of each other.
        xml = chart_xml(plain_book)[0]
        sizes = [
            el.get("sz")
            for el in elements(xml, "defRPr", within="dLbls")
            if el.get("sz")
        ]
        assert "800" in sizes


class TestChartLayout:
    def test_anchors_are_spaced_wider_than_the_chart(self):
        # The three fit charts sit side by side. Their anchors are derived
        # from the chart width so widening one cannot overlap the next.
        width = 18
        cols = [
            openpyxl.utils.column_index_from_string(
                re.match(r"([A-Z]+)", LW._chart_anchor(i, width)).group(1)
            )
            for i in range(3)
        ]
        step_cm = (cols[1] - cols[0]) * LW._COL_CM
        assert cols == sorted(cols)
        assert step_cm > width, (
            f"charts step {step_cm:.1f}cm apart but are {width}cm wide"
        )

    def test_charts_sit_below_the_data(self, plain_book):
        wb = openpyxl.load_workbook(plain_book)
        ws = wb["Fit_Chart_Data"]
        for chart in ws._charts:
            assert chart.anchor._from.row + 1 > ws.max_row


class TestSCurveSheet:
    def test_one_row_per_percentile(self, risk_book):
        wb = openpyxl.load_workbook(risk_book)
        assert wb["Risk_SCurve"].max_row == 100  # header plus 99

    def test_named_markers_for_p50_and_p80(self, risk_book):
        ws = openpyxl.load_workbook(risk_book)["Risk_SCurve"]
        assert ws.cell(row=1, column=4).value.startswith("P50")
        assert ws.cell(row=1, column=6).value.startswith("P80")
        assert ws.cell(row=2, column=4).value == pytest.approx(0.50)
        assert ws.cell(row=2, column=6).value == pytest.approx(0.80)
        assert ws.cell(row=2, column=7).value > ws.cell(row=2, column=5).value

    def test_marker_names_carry_the_cost(self, risk_book):
        # The label has to name the percentile and show its cost, and the
        # series name is the only place both fit.
        ws = openpyxl.load_workbook(risk_book)["Risk_SCurve"]
        for name_col, cost_col in ((4, 5), (6, 7)):
            label = ws.cell(row=1, column=name_col).value
            cost = ws.cell(row=2, column=cost_col).value
            assert "$" in label
            assert label.split()[-1] == LW._money_short(cost)

    def test_money_short_picks_a_sensible_unit(self):
        assert LW._money_short(250_000_000) == "$250.0M"
        assert LW._money_short(2_500_000_000) == "$2.5B"
        assert LW._money_short(45_200) == "$45.2K"
        assert LW._money_short(870) == "$870"

    def test_markers_are_named_in_the_legend_not_beside_the_curve(
        self, risk_book
    ):
        # Excel can only put a data label immediately next to its point, and
        # the curve runs through the point, so a label there gets the line
        # drawn across it. The legend is the one place nothing can overlap.
        xml = [x for x in chart_xml(risk_book) if "S-Curve" in x]
        assert xml, "no S-curve chart found"
        assert not elements(xml[0], "dLbls"), (
            "S-curve markers should be named in the legend, not by data labels"
        )
        assert elements(xml[0], "legend"), "S-curve needs its legend"
        assert "b" in flag(xml[0], "legendPos")

    def test_curve_and_two_markers(self, risk_book):
        xml = [x for x in chart_xml(risk_book) if "S-Curve" in x][0]
        assert len(elements(xml, "ser")) == 3


# Run inside the archive by a subprocess, not here. It prices the bundled
# example lots and reports where each module was imported from, which is the
# only way to tell an archive that works from an archive that quietly fell
# back to the repository sitting next to it.
ARCHIVE_PROBE = r'''
import json
import sys

archive = sys.argv[1]
sys.path.insert(0, archive)

# An editable install of the library puts a finder on sys.meta_path, and
# meta_path is consulted before sys.path. Drop it, or a passing test would
# only prove the checkout it was built from still imports.
sys.meta_path = [
    f for f in sys.meta_path
    if not getattr(f, "__module__", "").startswith("__editable__")
]

import pandas as pd

import cost_core
import lot_cost_model as M
import risk

analogy = pd.DataFrame(
    [
        (i + 1, int(fy), float(qty), float(auc))
        for i, (fy, qty, auc) in enumerate(M.EXAMPLE_ANALOGY)
    ],
    columns=["Lot", "Lot FY", "Qty", "AUC ($K)"],
)
estimate = pd.DataFrame(
    [
        (i + 1, int(fy), float(qty), float(cf))
        for i, (fy, qty, cf) in enumerate(M.EXAMPLE_ESTIMATE)
    ],
    columns=["Lot", "Lot FY", "Qty", "Complexity"],
)
proj, ctx = M.run_lot_cost_model(analogy, estimate)

print(json.dumps({
    "app_file": M.__file__,
    "risk_file": risk.__file__,
    "cost_core_file": cost_core.__file__,
    "tool_version": M.TOOL_VERSION,
    "lots": len(proj),
    "total": float(proj["LC+Rate Lot Cost After Complexity ($)"].sum()),
}))
'''


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    """Build the .pyz once. Every test below reads the same archive."""
    sys.path.insert(0, str(ROOT / "tools"))
    import build_pyz

    return build_pyz.build(ROOT, tmp_path_factory.mktemp("pyz"))


class TestSingleFileBuild:
    """The tool has to survive being bundled into one archive."""

    def test_it_builds_and_contains_every_module(self, archive):
        assert archive.exists()
        names = set(zipfile.ZipFile(archive).namelist())

        # The window and its entry point.
        assert {"__main__.py", "lot_cost_model.py", "risk.py"} <= names
        # wbs.py moved into the library. If it comes back, something is
        # shipping a second copy of the roll-up.
        assert "wbs.py" not in names

        # The vendored library, and the modules the window actually calls
        # by name. Without these the archive imports and then dies on the
        # first Run Model.
        assert {
            "cost_core/__init__.py",
            "cost_core/lotmodel/__init__.py",
            "cost_core/lotmodel/engine.py",
            "cost_core/program/rollup.py",
            "cost_core/reporting/lot_workbook.py",
            "cost_core/reporting/program_workbook.py",
        } <= names

        # Every subpackage came with it, whatever the library's shape is
        # today. Copying only the top level leaves an archive that imports
        # cost_core and then fails on the first submodule.
        package = pathlib.Path(LW.__file__).resolve().parent.parent
        expected = {
            f"cost_core/{sub.parent.relative_to(package).as_posix()}/__init__.py"
            for sub in package.rglob("__init__.py")
            if sub.parent != package
        }
        assert expected <= names, sorted(expected - names)

    def test_the_archive_is_compressed(self, archive):
        # zipapp stores rather than deflates unless asked, which left the
        # archive about three times the size it needs to be.
        info = zipfile.ZipFile(archive).getinfo("lot_cost_model.py")
        assert info.compress_type == zipfile.ZIP_DEFLATED
        assert info.compress_size < info.file_size

    def test_it_carries_no_tests_or_scratch(self, archive):
        names = zipfile.ZipFile(archive).namelist()
        for name in names:
            parts = pathlib.PurePosixPath(name).parts
            assert "__pycache__" not in parts, name
            assert "tests" not in parts, name
            assert not name.endswith((".pyc", ".pyo")), name
            # Model output and built archives. Both are gitignored, so they
            # are only ever here by accident, and a workbook full of real
            # program data is not something to hand a colleague by mistake.
            assert not name.endswith((".xlsx", ".xls", ".csv", ".pyz")), name

    def test_the_archive_imports_and_prices(self, archive, tmp_path):
        # This used to import lot_cost_model in-process, where the repository
        # copy was already in sys.modules, so it passed without ever opening
        # the archive. Run it out of process, from a working directory that
        # is not the repository, with PYTHONPATH cleared.
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["MPLBACKEND"] = "Agg"
        done = subprocess.run(
            [sys.executable, "-c", ARCHIVE_PROBE, str(archive)],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        out = json.loads(done.stdout.strip().splitlines()[-1])

        # Every module came out of the archive, not out of the repository.
        for key in ("app_file", "risk_file", "cost_core_file"):
            assert out[key].startswith(str(archive)), (key, out[key])

        assert out["tool_version"] == M.TOOL_VERSION
        assert out["lots"] == len(M.EXAMPLE_ESTIMATE)
        assert out["total"] > 0

    def test_the_entry_point_is_importable(self):
        # main() has to be callable, not buried in a __main__ guard, or the
        # archive has nothing to start.
        assert callable(M.main)
