"""Prediction intervals and Monte Carlo, borrowed from cost_core.

The deterministic side of this tool (learning curve, rate, LC+Rate, model
selection, complexity factors) is its own. What it never had was any statement
of uncertainty. Rather than write that a second time, this module hands the
finished fit to `cost_core` from the cost-risk-toolkit and reports what comes
back.

The handoff is deliberately thin. `cost_core.lots.projection_intervals` and
`simulate_buy` take the very objects `run_lot_cost_model` already returns, so
nothing is refitted and nothing is re-derived: the intervals and the simulation
describe *this tool's own* selected model, on this tool's own lot positions,
with the complexity factors already applied. The point estimate underneath the
distribution is therefore identical to the one on the projections sheet, by
construction rather than by luck.

An earlier version of this bridge fitted its own curve through cost_core and
redirected a private hook to line the lot positions up. That is all gone; the
public functions do the job directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from cost_core.lots import (
        projection_intervals,
        selected_model_name,
        simulate_buy,
    )

    AVAILABLE = True
    IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on the environment
    AVAILABLE = False
    IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

INSTALL_HINT = (
    "The risk analysis needs cost_core, which is not installed.\n\n"
    "Install it with:\n"
    "    pip install git+https://github.com/MichaelFowler1/cost-risk-toolkit.git\n\n"
    "Everything else in this tool works without it."
)


@dataclass
class RiskOptions:
    """Everything the analyst chooses about the risk run."""

    level: float = 0.80
    n_iter: int = 20000
    seed: int = 11
    lot_correlation: float = 0.30
    simulate: bool = True


@dataclass
class RiskResult:
    """Intervals on each forecast lot, and the simulated total of the buy."""

    model: str
    n_obs: int
    df: int
    t1: float
    slope: float
    sigma: float
    intervals: pd.DataFrame
    total_point: float
    total_lower: float
    total_upper: float
    level: float
    p50: float | None = None
    p80: float | None = None
    p90: float | None = None
    sim_mean: float | None = None
    sim_std: float | None = None
    sim_cv: float | None = None
    point_percentile: float | None = None
    reserve_to_p80: float | None = None
    n_iter: int | None = None
    scurve: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _scurve_frame(totals, step: int = 1) -> pd.DataFrame:
    """Cumulative distribution of the buy total, as a percentile table.

    The simulation produces tens of thousands of draws. Writing every one to a
    worksheet is useless; the readable form is the S-curve, which is just
    those draws read back at each percentile.
    """
    pct = np.arange(step, 100, step)
    return pd.DataFrame(
        {
            "Percentile": pct / 100.0,
            "Buy Total ($)": np.round(np.percentile(totals, pct), 2),
        }
    )


def _stat(projections: pd.DataFrame, model: str, suffix: str, default=np.nan):
    """Read one fit statistic for the selected model off the projections."""
    prefix = {"LC": "LC", "Rate": "Rate", "LC+Rate": "LC+Rate"}.get(model, "LC")
    col = f"{prefix} {suffix}"
    if col in projections.columns and len(projections):
        return projections[col].iloc[0]
    return default


def run_risk(
    ctx: dict,
    projections: pd.DataFrame,
    summary: pd.DataFrame,
    opts: RiskOptions,
) -> RiskResult:
    """Put intervals and a simulated total around an already-fitted run.

    Args:
        ctx: The second return value of ``run_lot_cost_model``.
        projections: Its first return value, one row per forecast lot.
        summary: The analyst summary, read only to learn which model was
            selected.
        opts: Analyst choices for this run.

    Raises:
        RuntimeError: If cost_core is not installed, or it could not read a
            selected model out of the summary.
    """
    if not AVAILABLE:
        raise RuntimeError(INSTALL_HINT)

    try:
        model = selected_model_name(summary)
    except Exception as exc:
        raise RuntimeError(
            f"cost_core could not tell which model was selected: {exc}"
        ) from exc

    notes: list[str] = []
    warns: list[str] = []

    cfg = ctx.get("cfg", {})
    prior = int(cfg.get("FcstPriorUnits", 0) or 0)
    if prior == 0:
        notes.append(
            "Forecast lots are priced from unit 1, so this is the analogy "
            "case: a different programme carrying this curve's slope. Whether "
            "the slope transfers is a judgement, and the error it introduces "
            "is not in any interval below."
        )
    else:
        notes.append(
            f"Forecast lots continue from unit {prior + 1}, so the intervals "
            "describe further production on this same curve."
        )

    n_keep = int(ctx.get("n_keep", 0))
    n_qty_only = int(ctx.get("n_unit", n_keep)) - n_keep
    if n_qty_only > 0:
        notes.append(
            f"{n_qty_only} quantity-only lot(s) took no part in the fit, but "
            "their units stay in the cumulative count, so later lots sit "
            "where they actually fall on the curve."
        )

    notes.append(
        "The complexity factor is already in these numbers and is treated as "
        "a certain multiplier. Its own uncertainty is not modelled."
    )

    intervals = projection_intervals(
        ctx, projections, model, level=opts.level
    ).copy()

    # cost_core returns the priced lots without their unit ranges. They are
    # row-aligned with the projections, so carry them across: the band chart
    # plots against cumulative units, and a reader wants to see them anyway.
    for col in ("First Unit in Lot", "Last Unit in Lot", "Complexity Factor"):
        if col in projections.columns and col not in intervals.columns:
            intervals[col] = projections[col].to_numpy()

    dof = _stat(projections, model, "df")
    try:
        dof = int(dof)
    except (TypeError, ValueError):
        dof = 0
    if dof and dof <= 2:
        warns.append(
            f"Only {dof} degree(s) of freedom. Read the percentiles rather "
            "than the mean or CV, and treat the interval as a floor on the "
            "real uncertainty."
        )

    total_point = float(intervals["Lot Cost ($)"].sum())
    total_lower = float(intervals["Lot Cost Lower"].sum())
    total_upper = float(intervals["Lot Cost Upper"].sum())

    result = RiskResult(
        model=str(model),
        n_obs=n_keep,
        df=dof,
        t1=float(_stat(projections, model, "T1 First Unit Cost ($K)")),
        slope=float(
            _stat(
                projections,
                model,
                "Learning Slope (%)" if model != "Rate" else "Slope (%)",
            )
        ),
        sigma=float(_stat(projections, model, "SEy")),
        intervals=intervals,
        total_point=total_point,
        total_lower=total_lower,
        total_upper=total_upper,
        level=opts.level,
        notes=notes,
        warnings=warns,
    )

    if opts.simulate:
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            buy = simulate_buy(
                ctx,
                projections,
                model,
                n_iter=int(opts.n_iter),
                seed=int(opts.seed),
                lot_correlation=float(opts.lot_correlation),
            )
        for w in caught:
            text = str(w.message)
            if text not in result.warnings:
                result.warnings.append(text)
        if getattr(buy, "clipped", 0):
            result.warnings.append(
                f"{buy.clipped} of {opts.n_iter} draws were clipped to keep "
                "the output finite. That many lots cannot support a confident "
                "forecast."
            )

        totals = np.asarray(buy.totals, dtype=float)
        result.p50 = float(buy.p50)
        result.p80 = float(buy.p80)
        result.p90 = float(buy.p90)
        result.sim_mean = float(buy.mean)
        result.sim_std = float(buy.std)
        result.sim_cv = float(buy.cv)
        result.point_percentile = float(buy.point_estimate_percentile)
        result.reserve_to_p80 = float(buy.p80 - buy.point_estimate)
        result.n_iter = int(buy.n_iter)
        result.scurve = _scurve_frame(totals)

    return result


def summary_frame(res: RiskResult) -> pd.DataFrame:
    """The risk result as a two-column table, for the results pane and Excel."""

    def money(v):
        return "n/a" if v is None or pd.isna(v) else f"{v:,.2f}"

    def num(v, fmt="{:,.4f}"):
        return "n/a" if v is None or pd.isna(v) else fmt.format(v)

    rows = [
        ("Enriched by", "cost_core (cost-risk-toolkit)"),
        ("Model", res.model),
        ("Analogy lots in fit", str(res.n_obs)),
        ("Degrees of freedom", str(res.df)),
        ("T1 first unit cost ($K)", money(res.t1)),
        ("Learning curve slope (%)", num(res.slope, "{:,.2f}")),
        ("SEy (log)", num(res.sigma)),
        ("", ""),
        ("Buy total, point estimate ($)", money(res.total_point)),
        (
            f"Buy total, {res.level:.0%} interval ($)",
            f"{res.total_lower:,.2f} to {res.total_upper:,.2f}",
        ),
    ]
    if res.p50 is not None:
        rows += [
            ("", ""),
            ("Monte Carlo iterations", f"{res.n_iter:,}"),
            ("Simulated P50 ($)", money(res.p50)),
            ("Simulated P80 ($)", money(res.p80)),
            ("Simulated P90 ($)", money(res.p90)),
            ("Simulated mean ($)", money(res.sim_mean)),
            ("Simulated CV", num(res.sim_cv)),
            ("Reserve to P80 ($)", money(res.reserve_to_p80)),
            (
                "Point estimate falls at",
                f"P{res.point_percentile:.0f} of the simulated buy",
            ),
        ]
    for i, note in enumerate(res.notes):
        rows.append(("Note" if i == 0 else "", note))
    for i, w in enumerate(res.warnings):
        rows.append(("Warning" if i == 0 else "", w))
    return pd.DataFrame(rows, columns=["Item", "Value"])
