"""Prediction intervals and Monte Carlo, borrowed from cost_core.

The deterministic side of this tool (learning curve, rate, LC+Rate, model
selection, complexity factors) is its own. What it never had was any statement
of uncertainty. Rather than write that a second time, this module hands the
same lots to `cost_core` from the cost-risk-toolkit and reports what comes
back.

Two things are worth knowing about the handoff.

`cost_core` fits the *exact* lot average, in closed form under Wright and by
summation under Crawford. This tool fits the lot midpoint approximation. They
are different estimators and will not agree to the last decimal, which is why
the fitted parameters from both are reported side by side: if they diverge by
much, something about the data deserves a look before either number gets used.

`cost_core.lots` reads a CSV and drops any row missing a cost, which also drops
those units from the cumulative count and shifts every later lot's position on
the curve. This tool holds quantity-only lots, so instead of handing over a
lot series we fit against explicit unit ranges, which keeps those units in the
count where they belong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from cost_core.learning_curve import fit_curve
    from cost_core.lots import LotFitReport, LotSeries

    AVAILABLE = True
    IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on the environment
    AVAILABLE = False
    IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

THEORIES = ("crawford", "wright")
METHODS = ("ols", "mupe", "zmpe")

INSTALL_HINT = (
    "The risk analysis needs cost_core, which is not installed.\n\n"
    "Install it with:\n"
    "    pip install git+https://github.com/MichaelFowler1/cost-risk-toolkit.git\n\n"
    "Everything else in this tool works without it."
)


def unit_spans(quantities, prior_units: int = 0) -> np.ndarray:
    """First and last unit of each lot, as an (n, 2) integer array.

    Mirrors ``track_units`` in the main module so both halves of the tool
    place lots on the curve identically.
    """
    q = np.asarray(quantities, dtype=int)
    cums = np.cumsum(q) + int(prior_units)
    return np.column_stack([cums - q + 1, cums]).astype(int)


def _maybe_call(obj, name):
    """Read an attribute that cost_core may expose as a value or a method."""
    attr = getattr(obj, name, None)
    if attr is None:
        return None
    try:
        return attr() if callable(attr) else attr
    except Exception:
        return None


@dataclass
class RiskOptions:
    """Everything the analyst chooses about the risk run."""

    theory: str = "crawford"
    method: str = "ols"
    level: float = 0.80
    n_iter: int = 20000
    seed: int = 11
    residual_correlation: float = 0.30
    include_residual: bool = True
    cost_basis: str = "recurring"
    dollar_year: int | None = None
    program: str = "unnamed program"
    simulate: bool = True


@dataclass
class RiskResult:
    """Fitted parameters, per-lot intervals, and the simulated buy total."""

    t1: float
    b: float
    slope: float
    slope_interval: tuple[float, float] | None
    sigma: float
    r_squared: float
    n_obs: int
    df: int
    equation: str
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
    n_iter: int | None = None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def run_risk(
    analogy_qty,
    analogy_auc,
    fcst_qty,
    fcst_cf,
    cfg: dict,
    opts: RiskOptions,
) -> RiskResult:
    """Fit with cost_core and return intervals plus a simulated buy total.

    Args:
        analogy_qty: Units in each historical lot, in build order.
        analogy_auc: Average unit cost of each historical lot, NaN where the
            lot is quantity-only. Those lots keep their units in the
            cumulative count but take no part in the fit.
        fcst_qty: Units in each forecast lot.
        fcst_cf: Complexity factor per forecast lot.
        cfg: The tool's settings, for the unit and total scales and the prior
            unit counts.
        opts: Analyst choices for this run.

    Raises:
        RuntimeError: If cost_core is not installed.
        ValueError: If fewer than three costed lots remain, or the dollar year
            is missing.
    """
    if not AVAILABLE:
        raise RuntimeError(INSTALL_HINT)
    if opts.dollar_year is None:
        raise ValueError(
            "A base year is required for the risk analysis.\n\n"
            "Constant dollars are constant relative to a year, and an interval "
            "nobody can place in a year cannot be compared to anything later. "
            "Set 'Base year ($)' on the Run Info tab."
        )

    qty = np.asarray(analogy_qty, dtype=float)
    auc = np.asarray(analogy_auc, dtype=float)
    ranges = unit_spans(qty, int(cfg.get("FitPriorUnits", 0)))

    costed = np.isfinite(auc) & (auc > 0)
    n_costed = int(costed.sum())
    if n_costed < 3:
        raise ValueError(
            f"The risk fit needs at least 3 lots with a cost. Found {n_costed}."
        )

    notes: list[str] = []
    warns: list[str] = []

    n_qty_only = int(len(qty) - n_costed)
    if n_qty_only:
        notes.append(
            f"{n_qty_only} quantity-only lot(s) took no part in the fit, but "
            "their units stay in the cumulative count, so later lots sit "
            "where they actually fall on the curve."
        )

    lot_costs = auc[costed] * qty[costed]
    fit = fit_curve(
        theory=opts.theory,
        method=opts.method,
        lots=ranges[costed],
        lot_costs=lot_costs,
    )

    if fit.df <= 2:
        warns.append(
            f"Only {fit.df} degree(s) of freedom. Read the percentiles rather "
            "than the mean or CV, and treat the interval as a floor on the "
            "real uncertainty."
        )

    fq = np.asarray(fcst_qty, dtype=int)
    cf = np.asarray(fcst_cf, dtype=float)
    cf = np.where(np.isfinite(cf) & (cf > 0), cf, 1.0)
    prior = int(cfg.get("FcstPriorUnits", 0))
    fspans = unit_spans(fq, prior)

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

    iv = fit.forecast_lots(fspans, level=opts.level, kind="prediction")

    unit_scale = float(cfg.get("CostUnitScale", 1.0))
    total_scale = float(cfg.get("TotalScale", 1.0))

    out = pd.DataFrame(
        {
            "Lot": np.arange(1, len(fq) + 1),
            "Lot Quantity": fq,
            "First Unit in Lot": iv["first_unit"].to_numpy(),
            "Last Unit in Lot": iv["last_unit"].to_numpy(),
            "Complexity Factor": cf,
        }
    )
    for label, src in (("Unit Cost", "lot_average"),
                       ("Unit Cost Low", "lot_average_lower"),
                       ("Unit Cost High", "lot_average_upper")):
        out[f"{label} ($K)"] = np.round(iv[src].to_numpy() * unit_scale, 2)
    for label, src in (("Lot Cost", "lot_cost"),
                       ("Lot Cost Low", "lot_cost_lower"),
                       ("Lot Cost High", "lot_cost_upper")):
        out[f"{label} ($)"] = np.round(
            iv[src].to_numpy() * unit_scale * total_scale * cf, 2
        )

    notes.append(
        "The complexity factor scales the interval as a certain multiplier. "
        "Its own uncertainty is not modelled here."
    )

    total_point = float(out["Lot Cost ($)"].sum())
    total_lower = float(out["Lot Cost Low ($)"].sum())
    total_upper = float(out["Lot Cost High ($)"].sum())

    slope_iv = _maybe_call(fit, "slope_interval")
    try:
        slope_iv = tuple(float(v) for v in slope_iv) if slope_iv else None
    except (TypeError, ValueError):
        slope_iv = None
    equation = _maybe_call(fit, "equation") or ""

    result = RiskResult(
        t1=float(fit.t1) * unit_scale,
        b=float(fit.model.b),
        slope=float(fit.slope),
        slope_interval=slope_iv,
        sigma=float(fit.result.sigma),
        r_squared=float(fit.r_squared),
        n_obs=int(fit.n_obs),
        df=int(fit.df),
        equation=str(equation),
        intervals=out,
        total_point=total_point,
        total_lower=total_lower,
        total_upper=total_upper,
        level=opts.level,
        notes=notes,
        warnings=warns,
    )

    if opts.simulate:
        _simulate(
            fit,
            qty[costed],
            lot_costs,
            fspans,
            fq,
            cf,
            unit_scale,
            total_scale,
            opts,
            result,
        )

    return result


def _simulate(
    fit,
    fit_qty,
    fit_lot_costs,
    fspans,
    fq,
    cf,
    unit_scale,
    total_scale,
    opts,
    result,
):
    """Run cost_core's forecast simulation over our own lot positions.

    ``LotFitReport.simulate_forecast`` derives lot positions from the series it
    was built on, which always continues from the last unit built. This tool
    also prices buys from unit 1, so the span helper is redirected to the spans
    already used for the intervals. Everything else, including the t-scaled
    parameter draws and the correlated residual shocks, is cost_core's.
    """
    import warnings as _warnings

    series = LotSeries(
        quantities=np.asarray(fit_qty, dtype=float),
        costs=np.asarray(fit_lot_costs, dtype=float),
        cost_basis=opts.cost_basis,
        first_unit=1,
        dollar_year=opts.dollar_year,
        program=opts.program,
    )
    report = LotFitReport(series=series, fit=fit)
    object.__setattr__(report, "_forecast_spans", lambda q: fspans)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        sim = report.simulate_forecast(
            [int(q) for q in fq],
            n_iter=int(opts.n_iter),
            seed=int(opts.seed),
            include_residual=opts.include_residual,
            residual_correlation=float(opts.residual_correlation),
        )
    for w in caught:
        text = str(w.message)
        if text not in result.warnings:
            result.warnings.append(text)

    # Apply the complexity factors lot by lot, then re-percentile the total.
    per_lot = np.asarray(sim.per_lot, dtype=float)
    scaled = per_lot * unit_scale * total_scale * cf[None, :]
    totals = scaled.sum(axis=1)

    result.p50 = float(np.percentile(totals, 50))
    result.p80 = float(np.percentile(totals, 80))
    result.p90 = float(np.percentile(totals, 90))
    result.sim_mean = float(np.mean(totals))
    result.sim_std = float(np.std(totals, ddof=1))
    result.sim_cv = (
        float(result.sim_std / result.sim_mean)
        if result.sim_mean
        else float("nan")
    )
    result.point_percentile = float(
        (totals < result.total_point).mean() * 100.0
    )
    result.n_iter = int(opts.n_iter)


def summary_frame(res: RiskResult) -> pd.DataFrame:
    """The risk result as a two-column table, for the results pane and Excel."""
    def money(v):
        return "n/a" if v is None else f"{v:,.2f}"

    rows = [
        ("Fitted with", "cost_core (cost-risk-toolkit)"),
        ("Observations in fit", str(res.n_obs)),
        ("Degrees of freedom", str(res.df)),
        ("T1 first unit cost ($K)", f"{res.t1:,.2f}"),
        ("Learning exponent (b)", f"{res.b:.6f}"),
        ("Learning curve slope", f"{res.slope:.2%}"),
        (
            "Slope interval",
            "n/a"
            if not res.slope_interval
            else f"{res.slope_interval[0]:.2%} to {res.slope_interval[1]:.2%}",
        ),
        ("Residual sigma (log)", f"{res.sigma:.4f}"),
        ("R2", f"{res.r_squared:.4f}"),
        ("Equation", res.equation),
        ("", ""),
        (f"Buy total, point estimate ($)", money(res.total_point)),
        (
            f"Buy total, {res.level:.0%} interval ($)",
            f"{res.total_lower:,.2f} to {res.total_upper:,.2f}",
        ),
    ]
    if res.p50 is not None:
        rows += [
            ("", ""),
            (f"Monte Carlo iterations", f"{res.n_iter:,}"),
            ("Simulated P50 ($)", money(res.p50)),
            ("Simulated P80 ($)", money(res.p80)),
            ("Simulated P90 ($)", money(res.p90)),
            ("Simulated mean ($)", money(res.sim_mean)),
            ("Simulated CV", f"{res.sim_cv:.4f}"),
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
