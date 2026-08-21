import numpy as np
import openpyxl
from openpyxl.chart import Reference, ScatterChart, Series
from openpyxl.chart.label import DataLabelList
from openpyxl.chart.layout import Layout, ManualLayout
from openpyxl.utils import get_column_letter
import pandas as pd

# ============================================================================
# 1. SETTINGS & CONFIGURATION
# ============================================================================
SETTINGS = {
    "AnalogyTableName": "AnalogyLots",
    "EstimateTableName": "EstimateLots",
    "CostUnitScale": 1.0,  # 1 = $K, 1000 = full dollars
    "TotalScale": 1000.0,  # Applied on top of CostUnitScale for totals
    "ToolMatchProjection": True,  # True = Rate & LC+Rate project on lot midpoint
    "DefaultCF": 1.0,
    "FitPriorUnits": 0,
    "FcstPriorUnits": 0,
    "SeedB": -0.152003093,
    "MaxIter": 100,
    "Tol": 1e-9,
    "RateSdFloor": 0.05,
    "SingularTol": 1e-12,
    "TGate": 2.0,
    "AiccTie": 2.0,
    "ToolVersion": "2.0-dev",
    "DefaultRunID": "R001",
    "DefaultProgram": "TEST",
    "DefaultRunLabel": "unlabeled run",
    "BaseYear": "",
}


# ============================================================================
# 2. SHARED MATHEMATICAL & STATISTICAL HELPERS
# ============================================================================
def find_col(df_columns: list, candidates: list) -> str | None:
    """Case-insensitive search for the first matching column name."""
    col_map = {c.strip().lower(): c for c in df_columns}
    for cand in candidates:
        if cand.strip().lower() in col_map:
            return col_map[cand.strip().lower()]
    return None


def to_num(val):
    """Clean string currency/commas and convert to numeric float."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return float(val)
    if isinstance(val, str):
        cleaned = val.replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned) if cleaned != "" else np.nan
        except ValueError:
            return np.nan
    return np.nan


def lmp_func(
    s: float, e: float, q: float, b: float | None
) -> float | None:
    """Calculate Lot Midpoint (LMP) given Start, End, Qty, and b-slope."""
    if b is None or pd.isna(b):
        return np.nan
    if q <= 1:
        return float(s)
    if abs(b) < 1e-12:
        return (s + e) / 2.0
    if abs(b + 1.0) < 1e-6:
        lo = max(s - 0.5, 1e-6)
        return q / (np.log(e + 0.5) - np.log(lo))

    p = b + 1.0
    lo = max(s - 0.5, 1e-6)
    v = ((e + 0.5) ** p - lo**p) / (p * q)
    if v <= 0:
        return np.nan
    return v ** (1.0 / b)


def ols_fit(
    x_cols: list[np.ndarray], y: np.ndarray, singular_tol: float = 1e-12
):
    """Ordinary Least Squares matching Excel/M Fit Space Statistics."""
    n = len(y)
    k = len(x_cols) + 1  # Intercept + predictors
    x = np.column_stack([np.ones(n)] + x_cols)

    xtx = x.T @ x
    xty = x.T @ y

    det = np.linalg.det(xtx)
    diag_prod = np.prod(np.abs(np.diag(xtx)))
    if diag_prod <= 0 or (abs(det) / diag_prod) < singular_tol:
        return None

    try:
        inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return None

    beta = inv @ xty
    fitted = x @ beta
    res = y - fitted
    sse = float(np.sum(res**2))
    ybar = float(np.mean(y))
    sst = float(np.sum((y - ybar) ** 2))
    ssr = sst - sse
    df = n - k

    s2 = (sse / df) if df > 0 else np.nan
    r2 = (1.0 - (sse / sst)) if sst > 0 else np.nan
    ar2 = (
        (1.0 - (1.0 - r2) * (n - 1) / df)
        if (df > 0 and pd.notna(r2))
        else np.nan
    )
    fstat = (
        ((ssr / (k - 1)) / (sse / df))
        if (df > 0 and k > 1 and sse > 0)
        else np.nan
    )

    se = []
    for a in range(k):
        v = s2 * inv[a, a] if pd.notna(s2) else np.nan
        se.append(np.sqrt(v) if pd.notna(v) and v > 0 else np.nan)

    return {
        "Beta": beta.tolist(),
        "SE": se,
        "Fitted": fitted,
        "SSE": sse,
        "InvDiag": np.diag(inv).tolist(),
        "R2": r2,
        "AdjR2": ar2,
        "SEy": np.sqrt(s2) if pd.notna(s2) and s2 > 0 else np.nan,
        "F": fstat,
        "DF": df,
        "SSreg": ssr,
        "SSresid": sse,
        "N": n,
        "K": k,
    }


def solve_model(
    fit_q: np.ndarray,
    fit_c: np.ndarray,
    fit_se: list[dict],
    use_rate: bool,
    cfg: dict,
):
    """Iterative solver for Learning Curve parameter b (Goal Seek equivalent)."""
    ln_y = np.log(fit_c)
    ln_r = np.log(fit_q)

    def mid_at(b_val):
        return np.array(
            [
                np.log(lmp_func(se["S"], se["E"], q, b_val))
                for se, q in zip(fit_se, fit_q)
            ]
        )

    b = cfg["SeedB"]
    delta = 1.0
    iteration = 0

    while iteration < cfg["MaxIter"] and delta > cfg["Tol"]:
        bp = b
        x_pred = [mid_at(bp), ln_r] if use_rate else [mid_at(bp)]
        fit = ols_fit(x_pred, ln_y, cfg["SingularTol"])
        if fit is None:
            return None
        b = fit["Beta"][1]
        delta = abs(b - bp)
        iteration += 1

    x_pred = [mid_at(b), ln_r] if use_rate else [mid_at(b)]
    final_fit = ols_fit(x_pred, ln_y, cfg["SingularTol"])
    if final_fit is None:
        return None

    resid = abs(final_fit["Beta"][1] - b)
    final_fit["Iter"] = iteration
    final_fit["Delta"] = resid
    final_fit["Converged"] = resid <= cfg["Tol"]
    return final_fit


def track_units(quantities: np.ndarray, prior: int):
    cums = np.cumsum(quantities) + prior
    starts = cums - quantities + 1
    return [{"S": s, "E": e} for s, e in zip(starts, cums)]


# ============================================================================
# 3. QUERY 1: CORE PROJECTIONS FACT TABLE
# ============================================================================
def run_lot_cost_model(
    analogy_df: pd.DataFrame,
    estimate_df: pd.DataFrame,
    config_overrides: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    cfg = SETTINGS.copy()
    if config_overrides:
        cfg.update(config_overrides)

    if analogy_df.empty:
        raise ValueError(
            f"Analogy table '{cfg['AnalogyTableName']}' contains no rows."
        )
    if estimate_df.empty:
        raise ValueError(
            f"Estimate table '{cfg['EstimateTableName']}' contains no rows."
        )

    # Resolve Columns - Analogy Table
    a_cols = list(analogy_df.columns)
    a_yr = find_col(
        a_cols,
        ["Year", "FY", "Fiscal Year", "Lot FY", "Lot Year", "FY of Lot"],
    )
    a_q = find_col(
        a_cols,
        [
            "Analogy Qtys",
            "Analogy Qty",
            "AnalogyQtys",
            "AnalogyQty",
            "Analogy Quantity",
            "Qty",
            "Qtys",
            "Quantity",
            "Lot Qty",
            "Lot Quantity",
        ],
    )
    a_c = find_col(
        a_cols,
        [
            "AnalogyUnitCost_K",
            "AnalogyUnitCost_Dollars",
            "AnalogyUnitCost",
            "Analogy Unit Cost",
            "AUC",
            "CP24",
            "AUC_K",
            "AUC ($K)",
            "Unit Cost",
            "UnitCost",
        ],
    )
    a_seq = find_col(
        a_cols,
        [
            "AnalogySeq",
            "Analogy Seq",
            "Analogy Lot",
            "Seq",
            "Sequence",
            "Lot",
            "Lot No",
            "Lot No.",
            "LotNo",
        ],
    )

    if not a_q or not a_c:
        raise ValueError(
            f"Analogy table needs a quantity column and a unit-cost column. Seen: {a_cols}"
        )

    df_a = analogy_df.copy()
    df_a["RowNo"] = np.arange(len(df_a))
    df_a["AQty"] = df_a[a_q].apply(to_num)
    df_a["ACost"] = df_a[a_c].apply(to_num)
    df_a["_Seq"] = (
        df_a[a_seq].apply(to_num) if a_seq else np.nan
    )
    df_a["_Yr"] = df_a[a_yr].apply(to_num) if a_yr else np.nan

    unit_keep = df_a[
        df_a["AQty"].notna() & (df_a["AQty"] > 0)
    ].copy()
    n_unit = len(unit_keep)
    seq_ok = (
        n_unit > 0
        and unit_keep["_Seq"].notna().sum() == n_unit
    )
    yr_ok = (
        n_unit > 0
        and unit_keep["_Yr"].notna().sum() == n_unit
    )

    if seq_ok:
        unit_set = unit_keep.sort_values(
            by=["_Seq", "RowNo"], ascending=[True, True]
        ).reset_index(drop=True)
    elif yr_ok:
        unit_set = unit_keep.sort_values(
            by=["_Yr", "RowNo"], ascending=[True, True]
        ).reset_index(drop=True)
    else:
        unit_set = unit_keep.sort_values(
            by=["RowNo"], ascending=[True]
        ).reset_index(drop=True)

    fit_set = unit_set[
        unit_set["ACost"].notna() & (unit_set["ACost"] > 0)
    ].reset_index(drop=True)
    n_keep = len(fit_set)

    # Resolve Columns - Estimate Table
    e_cols = list(estimate_df.columns)
    c_lot = find_col(
        e_cols,
        [
            "Lot",
            "Lot #",
            "LRIP",
            "Lot ID",
            "Lot No",
            "Lot No.",
            "LotNo",
            "LRIP Lot",
        ],
    )
    c_yr = find_col(
        e_cols,
        [
            "Year",
            "FY",
            "Fiscal Year",
            "Lot FY",
            "Lot Year",
            "Delivery FY",
            "Buy FY",
        ],
    )
    c_qty = find_col(
        e_cols,
        [
            "Qty",
            "Quantity",
            "Units",
            "Lot Qty",
            "Lot Quantity",
            "Estimate Qty",
            "Buy Qty",
            "LRIP Qty",
        ],
    )
    c_cf = find_col(
        e_cols,
        [
            "ComplexityFactor",
            "Complexity Factor",
            "Complexity/CP",
            "Complexity",
            "CF",
        ],
    )

    if not c_qty:
        raise ValueError(
            f"Estimate table needs a quantity column. Seen: {e_cols}"
        )

    df_e = estimate_df.copy()
    df_e["RowNo"] = np.arange(len(df_e))
    df_e["Lot"] = (
        df_e[c_lot].fillna("").astype(str)
        if c_lot
        else [f"Lot {i+1}" for i in range(len(df_e))]
    )
    df_e["Year"] = (
        df_e[c_yr].apply(to_num) if c_yr else np.nan
    )
    df_e["Qty"] = df_e[c_qty].apply(to_num)
    df_e["ComplexityFactor"] = (
        df_e[c_cf].apply(to_num) if c_cf else np.nan
    )

    fcst_ord = (
        df_e[df_e["Qty"].notna() & (df_e["Qty"] > 0)]
        .sort_values(
            by=["Year", "RowNo"], ascending=[True, True]
        )
        .reset_index(drop=True)
    )

    cf_vals = []
    last_cf = cfg["DefaultCF"]
    for val in fcst_ord["ComplexityFactor"]:
        if pd.isna(val) or val <= 0:
            cf_vals.append(last_cf)
        else:
            last_cf = float(val)
            cf_vals.append(last_cf)
    fcst_ord["ComplexityFactor"] = cf_vals

    if n_keep < 3:
        raise ValueError(
            f"Learning curve needs at least 3 analogy lots with both quantity and cost. Found: {n_keep}"
        )
    if len(fcst_ord) < 1:
        raise ValueError("No forecast rows found.")

    # Unit Tracking
    unit_q = unit_set["AQty"].to_numpy()
    unit_se = track_units(unit_q, cfg["FitPriorUnits"])
    complete_idx = unit_set[
        unit_set["ACost"].notna() & (unit_set["ACost"] > 0)
    ].index.tolist()

    fit_q = fit_set["AQty"].to_numpy()
    fit_c = fit_set["ACost"].to_numpy()
    fit_se = [unit_se[i] for i in complete_idx]

    fcst_q = fcst_ord["Qty"].to_numpy()
    fcst_se = track_units(fcst_q, cfg["FcstPriorUnits"])

    # Model Fitting
    ln_r = np.log(fit_q)
    ln_y = np.log(fit_c)
    rate_sd = (
        np.std(ln_r, ddof=1) if len(ln_r) > 1 else np.nan
    )

    rate_why = ""
    if n_keep < 4:
        rate_why = "fewer than 4 analogy lots"
    elif pd.isna(rate_sd):
        rate_why = "no spread in ln(lot qty)"
    elif rate_sd < cfg["RateSdFloor"]:
        rate_why = (
            "lot quantities too uniform for a rate term"
        )

    rate_ok = rate_why == ""

    mdl_lc = solve_model(
        fit_q, fit_c, fit_se, use_rate=False, cfg=cfg
    )
    mdl_rt = (
        ols_fit([ln_r], ln_y, cfg["SingularTol"])
        if rate_ok
        else None
    )
    mdl_lcr = (
        solve_model(
            fit_q, fit_c, fit_se, use_rate=True, cfg=cfg
        )
        if rate_ok
        else None
    )

    if mdl_lc is None:
        raise RuntimeError("Learning curve fit failed.")

    gb = (
        lambda m, i: (
            m["Beta"][i]
            if (m and len(m["Beta"]) > i)
            else np.nan
        )
    )
    gs = (
        lambda m, i: (
            m["SE"][i] if (m and len(m["SE"]) > i) else np.nan
        )
    )
    gf = lambda m, f: m.get(f, np.nan) if m else np.nan
    slope = (
        lambda b_val: (
            round((2**b_val) * 100, 2)
            if pd.notna(b_val)
            else np.nan
        )
    )
    t1_of = (
        lambda m: (
            round(
                np.exp(m["Beta"][0]) * cfg["CostUnitScale"], 2
            )
            if (m and "Beta" in m)
            else np.nan
        )
    )

    def stat_msg(m, name):
        if m is None:
            return f"{name} suppressed: {rate_why}"
        if not m.get("Converged", True):
            return f"{name} NOT CONVERGED"
        return f"{name} ok"

    fit_status = f"{stat_msg(mdl_lc, 'LC')}; {stat_msg(mdl_rt, 'Rate')}; {stat_msg(mdl_lcr, 'LC+Rate')}"
    data_source = f"{cfg['AnalogyTableName']} + {cfg['EstimateTableName']}"

    t1_lc, b_lc = np.exp(mdl_lc["Beta"][0]), gb(mdl_lc, 1)
    t1_rt, b_rt = (
        (np.exp(mdl_rt["Beta"][0]), gb(mdl_rt, 1))
        if mdl_rt
        else (np.nan, np.nan)
    )
    t1_br, b_br, c_br = (
        (
            np.exp(mdl_lcr["Beta"][0]),
            gb(mdl_lcr, 1),
            gb(mdl_lcr, 2),
        )
        if mdl_lcr
        else (np.nan, np.nan, np.nan)
    )

    # Projections on Forecast Lots
    res_df = fcst_ord.copy()
    res_df["FirstUnitInLot"] = [se["S"] for se in fcst_se]
    res_df["LastUnitInLot"] = [se["E"] for se in fcst_se]

    res_df["LC_LMP"] = [
        lmp_func(s, e, q, b_lc)
        for s, e, q in zip(
            res_df["FirstUnitInLot"],
            res_df["LastUnitInLot"],
            res_df["Qty"],
        )
    ]
    res_df["LC_UnitCost"] = (
        t1_lc
        * (res_df["LC_LMP"] ** b_lc)
        * cfg["CostUnitScale"]
    )
    res_df["LC_BaseTotal"] = (
        res_df["LC_UnitCost"]
        * res_df["Qty"]
        * cfg["TotalScale"]
    )
    res_df["LC_AdjTotal"] = (
        res_df["LC_BaseTotal"] * res_df["ComplexityFactor"]
    )

    res_df["RT_LMP"] = [
        lmp_func(s, e, q, b_rt)
        for s, e, q in zip(
            res_df["FirstUnitInLot"],
            res_df["LastUnitInLot"],
            res_df["Qty"],
        )
    ]
    if pd.notna(t1_rt):
        if cfg["ToolMatchProjection"]:
            res_df["RT_UnitCost"] = (
                t1_rt
                * (res_df["RT_LMP"] ** b_rt)
                * cfg["CostUnitScale"]
            )
        else:
            res_df["RT_UnitCost"] = (
                t1_rt
                * (res_df["Qty"] ** b_rt)
                * cfg["CostUnitScale"]
            )
        res_df["RT_BaseTotal"] = (
            res_df["RT_UnitCost"]
            * res_df["Qty"]
            * cfg["TotalScale"]
        )
        res_df["RT_AdjTotal"] = (
            res_df["RT_BaseTotal"] * res_df["ComplexityFactor"]
        )
    else:
        res_df["RT_UnitCost"] = res_df["RT_BaseTotal"] = (
            res_df["RT_AdjTotal"]
        ) = np.nan

    res_df["LCR_LMP"] = [
        lmp_func(s, e, q, b_br)
        for s, e, q in zip(
            res_df["FirstUnitInLot"],
            res_df["LastUnitInLot"],
            res_df["Qty"],
        )
    ]
    if pd.notna(t1_br):
        rate_factor = (
            1.0
            if cfg["ToolMatchProjection"]
            else (res_df["Qty"] ** c_br)
        )
        res_df["LCR_UnitCost"] = (
            t1_br
            * (res_df["LCR_LMP"] ** b_br)
            * rate_factor
            * cfg["CostUnitScale"]
        )
        res_df["LCR_BaseTotal"] = (
            res_df["LCR_UnitCost"]
            * res_df["Qty"]
            * cfg["TotalScale"]
        )
        res_df["LCR_AdjTotal"] = (
            res_df["LCR_BaseTotal"]
            * res_df["ComplexityFactor"]
        )
    else:
        res_df["LCR_UnitCost"] = res_df["LCR_BaseTotal"] = (
            res_df["LCR_AdjTotal"]
        ) = np.nan

    # Fit Statistics Columns
    rnd = (
        lambda v, d: (
            round(v, d) if pd.notna(v) else np.nan
        )
    )
    pct = (
        lambda v: (
            round(v * 100, 2) if pd.notna(v) else np.nan
        )
    )

    res_df["LC_T1"] = t1_of(mdl_lc)
    res_df["LC_Icept"] = rnd(gb(mdl_lc, 0), 4)
    res_df["LC_IceptSE"] = rnd(gs(mdl_lc, 0), 4)
    res_df["LC_Learn"] = rnd(gb(mdl_lc, 1), 4)
    res_df["LC_LearnSE"] = rnd(gs(mdl_lc, 1), 4)
    res_df["LC_LearnSlope"] = slope(gb(mdl_lc, 1))
    res_df["LC_R2"] = pct(gf(mdl_lc, "R2"))
    res_df["LC_SEy"] = rnd(gf(mdl_lc, "SEy"), 4)
    res_df["LC_F"] = rnd(gf(mdl_lc, "F"), 2)
    res_df["LC_df"] = gf(mdl_lc, "DF")
    res_df["LC_SSreg"] = rnd(gf(mdl_lc, "SSreg"), 4)
    res_df["LC_SSresid"] = rnd(gf(mdl_lc, "SSresid"), 4)
    res_df["LC_AdjR2"] = pct(gf(mdl_lc, "AdjR2"))

    res_df["RT_T1"] = t1_of(mdl_rt)
    res_df["RT_Icept"] = rnd(gb(mdl_rt, 0), 4)
    res_df["RT_IceptSE"] = rnd(gs(mdl_rt, 0), 4)
    res_df["RT_Rate"] = rnd(gb(mdl_rt, 1), 4)
    res_df["RT_RateSE"] = rnd(gs(mdl_rt, 1), 4)
    res_df["RT_RateSlope"] = slope(gb(mdl_rt, 1))
    res_df["RT_R2"] = pct(gf(mdl_rt, "R2"))
    res_df["RT_SEy"] = rnd(gf(mdl_rt, "SEy"), 4)
    res_df["RT_F"] = rnd(gf(mdl_rt, "F"), 2)
    res_df["RT_df"] = gf(mdl_rt, "DF")
    res_df["RT_SSreg"] = rnd(gf(mdl_rt, "SSreg"), 4)
    res_df["RT_SSresid"] = rnd(gf(mdl_rt, "SSresid"), 4)
    res_df["RT_AdjR2"] = pct(gf(mdl_rt, "AdjR2"))

    res_df["LCR_T1"] = t1_of(mdl_lcr)
    res_df["LCR_Icept"] = rnd(gb(mdl_lcr, 0), 4)
    res_df["LCR_IceptSE"] = rnd(gs(mdl_lcr, 0), 4)
    res_df["LCR_Learn"] = rnd(gb(mdl_lcr, 1), 4)
    res_df["LCR_LearnSE"] = rnd(gs(mdl_lcr, 1), 4)
    res_df["LCR_LearnSlope"] = slope(gb(mdl_lcr, 1))
    res_df["LCR_Rate"] = rnd(gb(mdl_lcr, 2), 4)
    res_df["LCR_RateSE"] = rnd(gs(mdl_lcr, 2), 4)
    res_df["LCR_RateSlope"] = slope(gb(mdl_lcr, 2))
    res_df["LCR_R2"] = pct(gf(mdl_lcr, "R2"))
    res_df["LCR_SEy"] = rnd(gf(mdl_lcr, "SEy"), 4)
    res_df["LCR_F"] = rnd(gf(mdl_lcr, "F"), 2)
    res_df["LCR_df"] = gf(mdl_lcr, "DF")
    res_df["LCR_SSreg"] = rnd(gf(mdl_lcr, "SSreg"), 4)
    res_df["LCR_SSresid"] = rnd(gf(mdl_lcr, "SSresid"), 4)
    res_df["LCR_AdjR2"] = pct(gf(mdl_lcr, "AdjR2"))

    res_df["FitStatus"] = fit_status
    res_df["DataSource"] = data_source

    if "Lot" not in res_df.columns:
        res_df["Lot"] = [
            f"Lot {i+1}" for i in range(len(res_df))
        ]
    if "Year" not in res_df.columns:
        res_df["Year"] = np.nan
    if "ComplexityFactor" not in res_df.columns:
        res_df["ComplexityFactor"] = cfg["DefaultCF"]

    round_cols_4 = [
        "LC_LMP",
        "RT_LMP",
        "LCR_LMP",
        "ComplexityFactor",
    ]
    round_cols_2 = [
        "LC_UnitCost",
        "LC_BaseTotal",
        "LC_AdjTotal",
        "RT_UnitCost",
        "RT_BaseTotal",
        "RT_AdjTotal",
        "LCR_UnitCost",
        "LCR_BaseTotal",
        "LCR_AdjTotal",
    ]
    for c in round_cols_4:
        res_df[c] = res_df[c].apply(lambda v: rnd(v, 4))
    for c in round_cols_2:
        res_df[c] = res_df[c].apply(lambda v: rnd(v, 2))

    res_df["Lot"] = res_df["Lot"].astype(str)
    res_df["Year"] = pd.to_numeric(
        res_df["Year"], errors="coerce"
    ).astype("Int64")
    res_df["Qty"] = pd.to_numeric(
        res_df["Qty"], errors="coerce"
    ).astype("Int64")
    res_df["FirstUnitInLot"] = pd.to_numeric(
        res_df["FirstUnitInLot"], errors="coerce"
    ).astype("Int64")
    res_df["LastUnitInLot"] = pd.to_numeric(
        res_df["LastUnitInLot"], errors="coerce"
    ).astype("Int64")

    rename_dict = {
        "Lot": "Lot",
        "Year": "Fiscal Year",
        "Qty": "Lot Quantity",
        "FirstUnitInLot": "First Unit in Lot",
        "LastUnitInLot": "Last Unit in Lot",
        "ComplexityFactor": "Complexity Factor",
        "LC_LMP": "LC Lot Midpoint (unit no.)",
        "LC_UnitCost": "LC Unit Cost ($K)",
        "LC_BaseTotal": "LC Lot Cost Before Complexity ($)",
        "LC_AdjTotal": "LC Lot Cost After Complexity ($)",
        "RT_LMP": "Rate Lot Midpoint (unit no.)",
        "RT_UnitCost": "Rate Unit Cost ($K)",
        "RT_BaseTotal": "Rate Lot Cost Before Complexity ($)",
        "RT_AdjTotal": "Rate Lot Cost After Complexity ($)",
        "LCR_LMP": "LC+Rate Lot Midpoint (unit no.)",
        "LCR_UnitCost": "LC+Rate Unit Cost ($K)",
        "LCR_BaseTotal": (
            "LC+Rate Lot Cost Before Complexity ($)"
        ),
        "LCR_AdjTotal": (
            "LC+Rate Lot Cost After Complexity ($)"
        ),
        "LC_T1": "LC T1 First Unit Cost ($K)",
        "LC_Icept": "LC Intercept Coeff",
        "LC_IceptSE": "LC Intercept SE",
        "LC_Learn": "LC Learning Coeff",
        "LC_LearnSE": "LC Learning SE",
        "LC_LearnSlope": "LC Learning Slope (%)",
        "LC_R2": "LC R2 (%)",
        "LC_SEy": "LC SEy",
        "LC_F": "LC F",
        "LC_df": "LC df",
        "LC_SSreg": "LC SSreg",
        "LC_SSresid": "LC SSresid",
        "LC_AdjR2": "LC Adj R2 (%)",
        "RT_T1": "Rate T1 First Unit Cost ($K)",
        "RT_Icept": "Rate Intercept Coeff",
        "RT_IceptSE": "Rate Intercept SE",
        "RT_Rate": "Rate Coeff",
        "RT_RateSE": "Rate SE",
        "RT_RateSlope": "Rate Slope (%)",
        "RT_R2": "Rate R2 (%)",
        "RT_SEy": "Rate SEy",
        "RT_F": "Rate F",
        "RT_df": "Rate df",
        "RT_SSreg": "Rate SSreg",
        "RT_SSresid": "Rate SSresid",
        "RT_AdjR2": "Rate Adj R2 (%)",
        "LCR_T1": "LC+Rate T1 First Unit Cost ($K)",
        "LCR_Icept": "LC+Rate Intercept Coeff",
        "LCR_IceptSE": "LC+Rate Intercept SE",
        "LCR_Learn": "LC+Rate Learning Coeff",
        "LCR_LearnSE": "LC+Rate Learning SE",
        "LCR_LearnSlope": "LC+Rate Learning Slope (%)",
        "LCR_Rate": "LC+Rate Rate Coeff",
        "LCR_RateSE": "LC+Rate Rate SE",
        "LCR_RateSlope": "LC+Rate Rate Slope (%)",
        "LCR_R2": "LC+Rate R2 (%)",
        "LCR_SEy": "LC+Rate SEy",
        "LCR_F": "LC+Rate F",
        "LCR_df": "LC+Rate df",
        "LCR_SSreg": "LC+Rate SSreg",
        "LCR_SSresid": "LC+Rate SSresid",
        "LCR_AdjR2": "LC+Rate Adj R2 (%)",
        "FitStatus": "Fit Status",
        "DataSource": "Input Table",
    }

    ordered_cols = list(rename_dict.keys())
    projections_df = res_df[ordered_cols].rename(
        columns=rename_dict
    )

    models_context = {
        "mdl_lc": mdl_lc,
        "mdl_rt": mdl_rt,
        "mdl_lcr": mdl_lcr,
        "fit_q": fit_q,
        "fit_c": fit_c,
        "fit_se": fit_se,
        "n_keep": n_keep,
        "n_unit": n_unit,
        "rate_sd": rate_sd,
        "rate_ok": rate_ok,
        "rate_why": rate_why,
        "t1_lc": t1_lc,
        "b_lc": b_lc,
        "t1_rt": t1_rt,
        "b_rt": b_rt,
        "t1_br": t1_br,
        "b_br": b_br,
        "c_br": c_br,
        "cfg": cfg,
    }

    return projections_df, models_context


# ============================================================================
# 4. QUERY 2: ANALYST SUMMARY ENGINE (Human-Readable Report)
# ============================================================================
def generate_analyst_summary(
    models_context: dict, run_info: dict | None = None
) -> pd.DataFrame:
    ctx = models_context
    cfg = ctx["cfg"]
    ri = run_info or {}

    run_id = ri.get("RunID", cfg["DefaultRunID"])
    program = ri.get("Program", cfg["DefaultProgram"])
    run_label = ri.get("RunLabel", cfg["DefaultRunLabel"])
    base_year = ri.get("BaseYear", cfg["BaseYear"])

    cost_basis_txt = (
        "NOT STATED - declare BaseYear in the RunInfo table"
        if not base_year
        else f"BY{base_year} $K as entered (no escalation applied by the tool)"
    )

    n_keep = ctx["n_keep"]
    n_unit = ctx["n_unit"]
    fit_c = ctx["fit_c"]
    rate_sd = ctx["rate_sd"]
    rate_ok = ctx["rate_ok"]
    rate_why = ctx["rate_why"]

    ln_y = np.log(fit_c)
    ybar = np.mean(ln_y)
    sst = np.sum((ln_y - ybar) ** 2)

    def stat_for(m, k, rate_idx=None):
        if m is None:
            return None
        sse0 = m["SSE"]
        dfe = n_keep - k
        see = np.sqrt(sse0 / dfe) if dfe > 0 else None
        r2 = (1.0 - sse0 / sst) if sst > 0 else None
        adj = (
            (1.0 - (1.0 - r2) * (n_keep - 1) / dfe)
            if (r2 is not None and dfe > 0)
            else None
        )
        cv = (
            np.sqrt(np.exp(see * see) - 1.0)
            if see is not None
            else None
        )

        fit_u = np.exp(m["Fitted"])
        mape = np.mean(np.abs(fit_c - fit_u) / fit_c)
        bias = np.mean(fit_c / fit_u - 1.0)

        kp = k + 1
        sseg = max(sse0, 1e-30)
        aicc = (
            (
                n_keep * np.log(sseg / n_keep)
                + 2 * kp
                + 2 * kp * (kp + 1) / (n_keep - kp - 1)
            )
            if (n_keep - kp - 1 > 0)
            else None
        )

        sec = (
            see * np.sqrt(m["InvDiag"][rate_idx])
            if (rate_idx is not None and see is not None)
            else None
        )
        tc = (
            (m["Beta"][rate_idx] / sec)
            if (sec is not None and sec >= 1e-15)
            else None
        )

        return {
            "SEE": see,
            "R2": r2,
            "Adj": adj,
            "CV": cv,
            "MAPE": mape,
            "Bias": bias,
            "AICc": aicc,
            "T": tc,
        }

    s_lc = stat_for(ctx["mdl_lc"], 2, None)
    s_rt = stat_for(ctx["mdl_rt"], 2, 1)
    s_lcr = stat_for(ctx["mdl_lcr"], 3, 2)

    gs = lambda s, f: s.get(f) if s else None

    t_gate = cfg["TGate"]
    aicc_tie = cfg["AiccTie"]

    lcr_gate = (
        s_lcr is not None
        and gs(s_lcr, "T") is not None
        and abs(gs(s_lcr, "T")) >= t_gate
    )
    rt_gate = (
        s_rt is not None
        and gs(s_rt, "T") is not None
        and abs(gs(s_rt, "T")) >= t_gate
    )

    if lcr_gate:
        sel = "LC+Rate"
    elif (
        ctx["mdl_lc"] is None and ctx["mdl_rt"] is not None
    ):
        sel = "Rate"
    elif (
        ctx["mdl_lc"] is not None
        and rt_gate
        and gs(s_rt, "AICc") is not None
        and gs(s_lc, "AICc") is not None
        and (
            gs(s_rt, "AICc") + aicc_tie < gs(s_lc, "AICc")
        )
    ):
        sel = "Rate"
    elif ctx["mdl_lc"] is not None:
        sel = "LC"
    else:
        sel = None

    aicc_vals = [
        v
        for v in [
            gs(s_lc, "AICc"),
            gs(s_rt, "AICc"),
            gs(s_lcr, "AICc"),
        ]
        if v is not None
    ]
    best_aicc = min(aicc_vals) if aicc_vals else None

    def d_aicc(s):
        if (
            s is None
            or gs(s, "AICc") is None
            or best_aicc is None
        ):
            return None
        return gs(s, "AICc") - best_aicc

    lcr_aicc_disagrees = (
        sel == "LC+Rate"
        and gs(s_lcr, "AICc") is not None
        and gs(s_lc, "AICc") is not None
        and (
            gs(s_lc, "AICc") + aicc_tie
            < gs(s_lcr, "AICc")
        )
    )

    rate_why_not = (
        ""
        if rate_ok
        else (
            "needs >= 4 costed lots"
            if n_keep < 4
            else f"SD(ln qty) {rate_sd:.4f} < {cfg['RateSdFloor']} floor"
        )
    )

    if sel == "LC+Rate":
        sel_note = (
            f"Rate coefficient significant (|t| >= {t_gate})."
            + (
                " Note: AICc favors LC at this sample size - state both in the BOE."
                if lcr_aicc_disagrees
                else ""
            )
        )
    elif sel == "Rate":
        sel_note = f"Slope significant and beats LC by more than {aicc_tie} AICc."
    elif sel == "LC":
        base_msg = "Default model. "
        if not rate_ok:
            base_msg += (
                f"Rate models gated off ({rate_why_not})."
            )
        elif (
            ctx["mdl_lcr"] is not None and not lcr_gate
        ):
            base_msg += f"Rate coefficient not significant (|t| < {t_gate})."
        if gs(s_lc, "AICc") is None:
            base_msg += " n too small for AICc; comparison on coefficient significance only."
        sel_note = base_msg
    else:
        sel_note = "No model could be fitted."

    def fit_txt(m):
        if m is not None:
            return "Yes"
        if not rate_ok:
            return f"No - rate gate ({rate_why_not})"
        return "No - did not converge or singular fit"

    fmt_n = (
        lambda v, f: (
            "n/a"
            if (v is None or pd.isna(v))
            else (
                f"{v:,.2f}"
                if f == "#,##0.00"
                else (
                    f"{v:.4f}"
                    if f == "0.0000"
                    else (
                        f"{v:.6f}"
                        if f == "0.000000"
                        else f"{v:.2f}"
                    )
                )
            )
        )
    )
    fmt_p = (
        lambda v: (
            "n/a"
            if (v is None or pd.isna(v))
            else f"{v * 100:.2f}%"
        )
    )
    fmt_ps = (
        lambda v: (
            "n/a"
            if (v is None or pd.isna(v))
            else f"{v * 100:+.2f}%"
        )
    )

    def mk_col(m, s, b_idx, c_idx):
        dash = "-"
        gc = (
            lambda idx: (
                m["Beta"][idx]
                if (
                    m is not None
                    and idx is not None
                    and len(m["Beta"]) > idx
                )
                else None
            )
        )
        return {
            "Fitted": fit_txt(m),
            "T1": (
                dash
                if m is None
                else fmt_n(
                    np.exp(m["Beta"][0])
                    * cfg["CostUnitScale"],
                    "#,##0.00",
                )
            ),
            "B": (
                dash
                if gc(b_idx) is None
                else fmt_n(gc(b_idx), "0.000000")
            ),
            "BS": (
                dash
                if gc(b_idx) is None
                else fmt_p(2 ** gc(b_idx))
            ),
            "C": (
                dash
                if gc(c_idx) is None
                else fmt_n(gc(c_idx), "0.000000")
            ),
            "CS": (
                dash
                if gc(c_idx) is None
                else fmt_p(2 ** gc(c_idx))
            ),
            "R2": (
                dash
                if m is None
                else fmt_n(gs(s, "R2"), "0.0000")
            ),
            "Adj": (
                dash
                if m is None
                else fmt_n(gs(s, "Adj"), "0.0000")
            ),
            "SEE": (
                dash
                if m is None
                else fmt_n(gs(s, "SEE"), "0.0000")
            ),
            "CV": dash if m is None else fmt_p(gs(s, "CV")),
            "MAPE": (
                dash
                if m is None
                else fmt_p(gs(s, "MAPE"))
            ),
            "Bias": (
                dash
                if m is None
                else fmt_ps(gs(s, "Bias"))
            ),
            "AICc": (
                dash
                if m is None
                else fmt_n(gs(s, "AICc"), "0.00")
            ),
            "DAI": (
                dash
                if m is None
                else fmt_n(d_aicc(s), "0.00")
            ),
            "T": (
                dash
                if m is None
                else fmt_n(gs(s, "T"), "0.00")
            ),
        }

    col_lc = mk_col(ctx["mdl_lc"], s_lc, 1, None)
    col_rt = mk_col(ctx["mdl_rt"], s_rt, None, 1)
    col_br = mk_col(ctx["mdl_lcr"], s_lcr, 1, 2)

    def r5(item, val, a, b, c):
        return {
            "Item": item,
            "Value": val,
            "LC": a,
            "Rate": b,
            "LC+Rate": c,
        }

    rows = [
        r5("Run ID", run_id, "", "", ""),
        r5("Program", program, "", "", ""),
        r5("Run label", run_label, "", "", ""),
        r5("Tool version", cfg["ToolVersion"], "", "", ""),
        r5("Cost basis", cost_basis_txt, "", "", ""),
        r5("Source table", cfg["AnalogyTableName"], "", "", ""),
        r5("Analogy lots in fit", str(n_keep), "", "", ""),
        r5(
            "Quantity-only lots (units held, not fit)",
            str(n_unit - n_keep),
            "",
            "",
            "",
        ),
        r5("SD(ln qty)", fmt_n(rate_sd, "0.0000"), "", "", ""),
        r5(
            "Rate models",
            (
                "enabled"
                if rate_ok
                else f"gated off ({rate_why_not})"
            ),
            "",
            "",
            "",
        ),
        r5("", "", "", "", ""),
        r5(
            "Fitted",
            "",
            col_lc["Fitted"],
            col_rt["Fitted"],
            col_br["Fitted"],
        ),
        r5(
            "SELECTED",
            "",
            "YES" if sel == "LC" else "",
            "YES" if sel == "Rate" else "",
            "YES" if sel == "LC+Rate" else "",
        ),
        r5(
            "T1 ($K)",
            "",
            col_lc["T1"],
            col_rt["T1"],
            col_br["T1"],
        ),
        r5(
            "Learning exponent (b)",
            "",
            col_lc["B"],
            col_rt["B"],
            col_br["B"],
        ),
        r5(
            "Learning curve slope",
            "",
            col_lc["BS"],
            col_rt["BS"],
            col_br["BS"],
        ),
        r5(
            "Rate exponent (c)",
            "",
            col_lc["C"],
            col_rt["C"],
            col_br["C"],
        ),
        r5(
            "Rate slope",
            "",
            col_lc["CS"],
            col_rt["CS"],
            col_br["CS"],
        ),
        r5(
            "R2 (log)",
            "",
            col_lc["R2"],
            col_rt["R2"],
            col_br["R2"],
        ),
        r5(
            "Adj R2",
            "",
            col_lc["Adj"],
            col_rt["Adj"],
            col_br["Adj"],
        ),
        r5(
            "SEE (log)",
            "",
            col_lc["SEE"],
            col_rt["SEE"],
            col_br["SEE"],
        ),
        r5(
            "CV",
            "",
            col_lc["CV"],
            col_rt["CV"],
            col_br["CV"],
        ),
        r5(
            "MAPE",
            "",
            col_lc["MAPE"],
            col_rt["MAPE"],
            col_br["MAPE"],
        ),
        r5(
            "Mean bias",
            "",
            col_lc["Bias"],
            col_rt["Bias"],
            col_br["Bias"],
        ),
        r5(
            "AICc",
            "",
            col_lc["AICc"],
            col_rt["AICc"],
            col_br["AICc"],
        ),
        r5(
            "dAICc",
            "",
            col_lc["DAI"],
            col_rt["DAI"],
            col_br["DAI"],
        ),
        r5(
            "t (rate coefficient)",
            "",
            col_lc["T"],
            col_rt["T"],
            col_br["T"],
        ),
        r5(
            "Selection basis",
            "",
            sel_note if sel in ("LC", None) else "",
            sel_note if sel == "Rate" else "",
            sel_note if sel == "LC+Rate" else "",
        ),
    ]

    return pd.DataFrame(rows)


# ============================================================================
# 5. QUERY 3: FIT CHART DATA ENGINE
# ============================================================================
def generate_fit_chart_data(
    models_context: dict,
) -> pd.DataFrame:
    """Compute the fit chart evaluation data matching M code Fit Chart Data."""
    ctx = models_context
    cfg = ctx["cfg"]

    fit_q = ctx["fit_q"]
    fit_c = ctx["fit_c"]
    fit_se = ctx["fit_se"]

    t1_lc, b_lc = ctx["t1_lc"], ctx["b_lc"]
    t1_rt, b_rt = ctx["t1_rt"], ctx["b_rt"]
    t1_br, b_br, c_br = (
        ctx["t1_br"],
        ctx["b_br"],
        ctx["c_br"],
    )

    rnd = (
        lambda v, d: (
            round(v, d)
            if (v is not None and pd.notna(v))
            else np.nan
        )
    )

    chart_rows = []
    for k in range(len(fit_q)):
        lot_no = k + 1
        a_qty = int(fit_q[k])
        first_u = int(fit_se[k]["S"])
        last_u = int(fit_se[k]["E"])
        actual = rnd(fit_c[k] * cfg["CostUnitScale"], 2)

        # LC calculations
        lc_mid = rnd(
            lmp_func(first_u, last_u, a_qty, b_lc), 4
        )
        lc_est = (
            rnd(
                t1_lc
                * (
                    lmp_func(first_u, last_u, a_qty, b_lc)
                    ** b_lc
                )
                * cfg["CostUnitScale"],
                2,
            )
            if pd.notna(t1_lc)
            else np.nan
        )
        lc_res = (
            rnd(actual - lc_est, 2)
            if pd.notna(lc_est)
            else np.nan
        )
        lc_resp = (
            rnd(((actual / lc_est) - 1.0) * 100, 2)
            if (pd.notna(lc_est) and lc_est != 0)
            else np.nan
        )

        # Rate calculations (against Analogy Lot Quantity)
        rt_est = (
            rnd(
                t1_rt
                * (a_qty**b_rt)
                * cfg["CostUnitScale"],
                2,
            )
            if pd.notna(t1_rt)
            else np.nan
        )
        rt_res = (
            rnd(actual - rt_est, 2)
            if pd.notna(rt_est)
            else np.nan
        )
        rt_resp = (
            rnd(((actual / rt_est) - 1.0) * 100, 2)
            if (pd.notna(rt_est) and rt_est != 0)
            else np.nan
        )

        # LC+Rate calculations (True fitted value with rate term)
        lcr_mid = rnd(
            lmp_func(first_u, last_u, a_qty, b_br), 4
        )
        lcr_est = (
            rnd(
                t1_br
                * (
                    lmp_func(first_u, last_u, a_qty, b_br)
                    ** b_br
                )
                * (a_qty**c_br)
                * cfg["CostUnitScale"],
                2,
            )
            if pd.notna(t1_br)
            else np.nan
        )
        lcr_res = (
            rnd(actual - lcr_est, 2)
            if pd.notna(lcr_est)
            else np.nan
        )
        lcr_resp = (
            rnd(((actual / lcr_est) - 1.0) * 100, 2)
            if (pd.notna(lcr_est) and lcr_est != 0)
            else np.nan
        )

        chart_rows.append(
            {
                "Analogy Lot No.": lot_no,
                "Analogy Lot Quantity": a_qty,
                "First Unit in Lot": first_u,
                "Last Unit in Lot": last_u,
                "Actual AUC ($K)": actual,
                "LC Lot Midpoint": lc_mid,
                "LC Estimated AUC ($K)": lc_est,
                "LC Residual ($K)": lc_res,
                "LC Residual (%)": lc_resp,
                "Rate Estimated AUC ($K)": rt_est,
                "Rate Residual ($K)": rt_res,
                "Rate Residual (%)": rt_resp,
                "LC+Rate Lot Midpoint": lcr_mid,
                "LC+Rate Estimated AUC ($K)": lcr_est,
                "LC+Rate Residual ($K)": lcr_res,
                "LC+Rate Residual (%)": lcr_resp,
            }
        )

    chart_df = pd.DataFrame(chart_rows)
    return chart_df


# ============================================================================
# 6. EXCEL WORKBOOK & SCATTER CHART GENERATOR (FIXED OPENPYXL TICK MARKS)
# ============================================================================
#: A default Excel column is roughly 1.72 cm wide. Chart anchors are spaced off
#: this so widening a chart cannot silently overlap the one beside it.
_COL_CM = 1.72


def _chart_anchor(index: int, width_cm: float, start_col: int = 2,
                  row: int = 10) -> str:
    """Anchor cell for the nth chart in a left-to-right row of charts."""
    step = int(width_cm / _COL_CM) + 4  # +4 columns of breathing room
    return f"{get_column_letter(start_col + index * step)}{row}"


def _format_chart(
    chart,
    title: str,
    x_title: str,
    y_title: str,
    width: float = 18,
    height: float = 11,
):
    """Apply the same axis, title and legend treatment to every chart.

    Two Excel quirks are handled here. openpyxl writes ``delete="1"`` onto a
    freshly created axis, which tells Excel to hide that axis completely, tick
    numbers and all; and a chart title defaults to overlaying the plot rather
    than sitting above it. The manual plot-area layout then reserves room on
    the left and bottom so the axis titles do not land on top of the tick
    numbers.
    """
    chart.title = title
    chart.style = 13
    chart.title.overlay = False

    chart.x_axis.title = x_title
    chart.y_axis.title = y_title

    for axis in (chart.x_axis, chart.y_axis):
        axis.delete = False
        axis.tickLblPos = "nextTo"
        axis.majorTickMark = "out"
        axis.minorTickMark = "none"
        axis.numFmt = "#,##0"

    chart.width = width
    chart.height = height

    # Keep the legend off the data.
    if chart.legend is not None:
        chart.legend.position = "b"
        chart.legend.overlay = False

    chart.layout = Layout(
        manualLayout=ManualLayout(
            xMode="edge",
            yMode="edge",
            x=0.11,
            y=0.13,
            w=0.86,
            h=0.68,
        )
    )


def save_complete_excel_workbook(
    filename: str,
    projections_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    chart_df: pd.DataFrame,
    risk_summary_df: pd.DataFrame | None = None,
    risk_intervals_df: pd.DataFrame | None = None,
):
    """Write the tables and embed native Excel scatter plots.

    The two risk frames are optional: they are present only when cost_core is
    installed and the risk analysis ran.
    """
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        summary_df.to_excel(
            writer, sheet_name="Analyst_Summary", index=False
        )
        projections_df.to_excel(
            writer,
            sheet_name="Estimate_Projections",
            index=False,
        )
        chart_df.to_excel(
            writer, sheet_name="Fit_Chart_Data", index=False
        )
        if risk_summary_df is not None:
            risk_summary_df.to_excel(
                writer, sheet_name="Risk_Summary", index=False
            )
        if risk_intervals_df is not None:
            risk_intervals_df.to_excel(
                writer, sheet_name="Risk_Intervals", index=False
            )

    wb = openpyxl.load_workbook(filename)
    ws = wb["Fit_Chart_Data"]
    max_r = len(chart_df) + 1  # 1-indexed including header

    fit_chart_width = 18

    def build_scatter_chart(
        title: str,
        x_col: int,
        actual_col: int,
        est_col: int,
        slot: int,
        x_axis_title_text: str,
        y_axis_title_text: str = "Unit Cost / AUC ($K)",
    ):
        chart = ScatterChart()
        _format_chart(
            chart,
            title,
            x_axis_title_text,
            y_axis_title_text,
            fit_chart_width,
            11,
        )

        x_values = Reference(
            ws, min_col=x_col, min_row=2, max_row=max_r
        )
        y_actual = Reference(
            ws, min_col=actual_col, min_row=1, max_row=max_r
        )
        y_est = Reference(
            ws, min_col=est_col, min_row=1, max_row=max_r
        )

        # Actuals: Markers with data labels showing numbers
        s_act = Series(
            values=y_actual,
            xvalues=x_values,
            title_from_data=True,
        )
        s_act.marker.symbol = "circle"
        s_act.marker.size = 7
        s_act.graphicalProperties.line.noFill = True

        # Show actual cost values on scatter points
        s_act.dataLabels = DataLabelList()
        s_act.dataLabels.showVal = True

        # Estimates: Smooth Fitted Curve (No markers)
        s_est = Series(
            values=y_est,
            xvalues=x_values,
            title_from_data=True,
        )
        s_est.marker.symbol = "none"
        s_est.smooth = True

        chart.series.append(s_act)
        chart.series.append(s_est)
        ws.add_chart(chart, _chart_anchor(slot, fit_chart_width))

    # Chart 1: Learning Curve (LC) Fit -> X = LC Midpoint (Col 6), Y = Actual (Col 5) vs LC_Est (Col 7)
    build_scatter_chart(
        title="Learning Curve Fit: Actual vs Estimated AUC",
        x_col=6,
        actual_col=5,
        est_col=7,
        slot=0,
        x_axis_title_text="LC Lot Midpoint (Unit Number)",
        y_axis_title_text="Unit Cost / AUC ($K)",
    )

    # Chart 2: Rate Model Fit -> X = Lot Qty (Col 2), Y = Actual (Col 5) vs Rate_Est (Col 10)
    build_scatter_chart(
        title="Rate Model Fit: Actual vs Estimated AUC",
        x_col=2,
        actual_col=5,
        est_col=10,
        slot=1,
        x_axis_title_text="Analogy Lot Quantity (Units / Lot)",
        y_axis_title_text="Unit Cost / AUC ($K)",
    )

    # Chart 3: LC + Rate Fit -> X = LC+Rate Midpoint (Col 13), Y = Actual (Col 5) vs LCR_Est (Col 14)
    build_scatter_chart(
        title="LC+Rate Model Fit: Actual vs Estimated AUC",
        x_col=13,
        actual_col=5,
        est_col=14,
        slot=2,
        x_axis_title_text="LC+Rate Lot Midpoint (Unit Number)",
        y_axis_title_text="Unit Cost / AUC ($K)",
    )

    # Chart 4: the forecast with its prediction band, on the risk sheet.
    if risk_intervals_df is not None and len(risk_intervals_df):
        wsr = wb["Risk_Intervals"]
        last_r = len(risk_intervals_df) + 1
        band = ScatterChart()
        _format_chart(
            band,
            "Forecast Unit Cost with Prediction Interval",
            "Last Unit in Lot",
            "Unit Cost ($K)",
            20,
            11,
        )

        xs = Reference(wsr, min_col=4, min_row=2, max_row=last_r)
        for col, dashed in ((6, False), (7, True), (8, True)):
            s = Series(
                values=Reference(
                    wsr, min_col=col, min_row=1, max_row=last_r
                ),
                xvalues=xs,
                title_from_data=True,
            )
            s.marker.symbol = "circle" if not dashed else "none"
            s.smooth = False
            if dashed:
                s.graphicalProperties.line.dashStyle = "dash"
            band.series.append(s)
        wsr.add_chart(band, "M2")

    wb.save(filename)



# ============================================================================
# 7. GUI DATA ENTRY (tkinter - ships with Python, no install needed)
# ============================================================================
import os
import sys
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Optional. Without it every deterministic feature still works; only the
# Risk tab goes dark.
try:
    import risk
except Exception:  # pragma: no cover - depends on the install
    risk = None

# Demo data for the "Load Example" buttons. These numbers are invented:
# they were generated from a 90% learning curve with a $1,000K first unit
# and a little random scatter, purely so the tool has something to chew on.
EXAMPLE_ANALOGY = [
    ("2015", "10", "800.61"),
    ("2016", "20", "639.49"),
    ("2017", "25", "563.66"),
    ("2018", "25", "520.05"),
    ("2019", "15", "502.98"),
    ("2020", "15", "487.08"),
]

EXAMPLE_ESTIMATE = [
    ("2028", "8", "1.15"),
    ("2029", "16", "1.15"),
    ("2030", "16", "1.15"),
    ("2031", "16", "1.15"),
    ("2032", "12", "1.15"),
    ("2033", "6", "1.15"),
]


def parse_float(text: str) -> float:
    """Parse a number, tolerating $ signs and thousands separators."""
    return float(text.replace("$", "").replace(",", "").strip())


def split_row(line: str) -> list[str]:
    """Split a pasted row on tabs, commas, or runs of spaces."""
    line = line.rstrip("\r")
    if "\t" in line:
        parts = line.split("\t")
    elif "," in line:
        parts = line.split(",")
    else:
        parts = line.split()
    return [p.strip() for p in parts]


def default_output_dir() -> str:
    """Somewhere guaranteed writable: the script's folder, else Documents."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        probe = os.path.join(here, ".__writetest")
        with open(probe, "w") as fh:
            fh.write("x")
        os.remove(probe)
        return here
    except Exception:
        return os.path.join(os.path.expanduser("~"), "Documents")


class LotGrid(ttk.Frame):
    """A small scrollable spreadsheet of Entry widgets."""

    def __init__(self, parent, headers: list[str], widths: list[int]):
        super().__init__(parent)
        self.headers = headers
        self.widths = widths
        self.rows: list[list[tk.Entry]] = []

        head = ttk.Frame(self)
        head.pack(fill="x", padx=(4, 0))
        ttk.Label(head, text="#", width=4, anchor="center").grid(
            row=0, column=0, padx=1
        )
        for i, (h, w) in enumerate(zip(headers, widths)):
            ttk.Label(
                head, text=h, width=w, anchor="center", style="Head.TLabel"
            ).grid(row=0, column=i + 1, padx=1)

        canvas_wrap = ttk.Frame(self)
        canvas_wrap.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_wrap, highlightthickness=0, height=240)
        scroll = ttk.Scrollbar(
            canvas_wrap, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.body = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            ),
        )
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _on_wheel(self, event):
        try:
            self.canvas.yview_scroll(int(-event.delta / 120), "units")
        except Exception:
            pass

    def add_row(self, values: tuple = None):
        r = len(self.rows)
        ttk.Label(self.body, text=str(r + 1), width=4, anchor="center").grid(
            row=r, column=0, padx=1, pady=1
        )
        entries = []
        for i, w in enumerate(self.widths):
            e = tk.Entry(self.body, width=w, justify="right")
            e.grid(row=r, column=i + 1, padx=1, pady=1)
            if values and i < len(values):
                e.insert(0, values[i])
            e.bind("<Control-v>", self._on_paste)
            e.bind("<Return>", lambda ev: self.add_row())
            entries.append(e)
        self.rows.append(entries)
        self.body.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        return entries

    def _on_paste(self, event):
        """Multi-line clipboard content fills the grid from this row down."""
        try:
            data = self.clipboard_get()
        except tk.TclError:
            return None
        if "\n" not in data.strip():
            return None  # single value: let Tk paste normally

        widget = event.widget
        start = 0
        for idx, row in enumerate(self.rows):
            if widget in row:
                start = idx
                break

        lines = [ln for ln in data.split("\n") if ln.strip()]
        for offset, line in enumerate(lines):
            target = start + offset
            while target >= len(self.rows):
                self.add_row()
            parts = split_row(line)
            for col, entry in enumerate(self.rows[target]):
                entry.delete(0, tk.END)
                if col < len(parts):
                    entry.insert(0, parts[col])
        return "break"

    def delete_last(self):
        if not self.rows:
            return
        for w in self.body.grid_slaves(row=len(self.rows) - 1):
            w.destroy()
        self.rows.pop()
        self.body.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def clear(self):
        while self.rows:
            self.delete_last()

    def load(self, data: list):
        self.clear()
        for values in data:
            self.add_row(values)

    def get_rows(self) -> list[list[str]]:
        """Non-empty rows as raw strings."""
        out = []
        for row in self.rows:
            vals = [e.get().strip() for e in row]
            if any(v for v in vals):
                out.append(vals)
        return out


class LotCostApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lot Cost Model - Learning Curve / Rate Analysis")
        self.geometry("980x760")
        self.minsize(860, 640)

        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("Head.TLabel", font=("Segoe UI", 9, "bold"))
        style.configure("Title.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Sub.TLabel", foreground="#555")

        ttk.Label(
            self, text="Lot Cost Model", style="Title.TLabel"
        ).pack(anchor="w", padx=12, pady=(10, 0))
        ttk.Label(
            self,
            text=(
                "Enter historical analogy lots and forecast lots, then run the "
                "model. Paste directly from Excel with Ctrl+V."
            ),
            style="Sub.TLabel",
        ).pack(anchor="w", padx=12, pady=(0, 8))

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=12)
        self.tab_analogy = ttk.Frame(nb)
        self.tab_estimate = ttk.Frame(nb)
        self.tab_run = ttk.Frame(nb)
        self.tab_results = ttk.Frame(nb)
        self.tab_risk = ttk.Frame(nb)
        nb.add(self.tab_analogy, text="  1. Analogy Lots  ")
        nb.add(self.tab_estimate, text="  2. Estimate Lots  ")
        nb.add(self.tab_run, text="  3. Run Info & Settings  ")
        nb.add(self.tab_results, text="  4. Results  ")
        nb.add(self.tab_risk, text="  5. Risk & Intervals  ")
        self.nb = nb

        self.risk_result = None

        self._build_analogy()
        self._build_estimate()
        self._build_runinfo()
        self._build_results()
        self._build_risk()
        self._build_actionbar()

    # -- tabs ---------------------------------------------------------------
    def _build_analogy(self):
        f = self.tab_analogy
        ttk.Label(
            f,
            text=(
                "Historical lots used to fit the curve. Need at least 3 lots "
                "with a unit cost.\n"
                "Leave AUC blank for a quantity-only lot (its units count "
                "toward learning, but it is not fit)."
            ),
            style="Sub.TLabel",
            justify="left",
        ).pack(anchor="w", padx=8, pady=(8, 6))

        self.grid_analogy = LotGrid(
            f,
            ["Fiscal Year", "Lot Quantity", "Unit Cost AUC ($K)"],
            [14, 14, 20],
        )
        self.grid_analogy.pack(fill="both", expand=True, padx=8)

        bar = ttk.Frame(f)
        bar.pack(fill="x", padx=8, pady=8)
        ttk.Button(
            bar, text="Add Row", command=self.grid_analogy.add_row
        ).pack(side="left")
        ttk.Button(
            bar, text="Delete Last Row", command=self.grid_analogy.delete_last
        ).pack(side="left", padx=4)
        ttk.Button(
            bar, text="Clear", command=self.grid_analogy.clear
        ).pack(side="left")
        ttk.Button(
            bar,
            text="Load Example",
            command=lambda: self.grid_analogy.load(EXAMPLE_ANALOGY),
        ).pack(side="right")

        for _ in range(6):
            self.grid_analogy.add_row()

    def _build_estimate(self):
        f = self.tab_estimate
        ttk.Label(
            f,
            text=(
                "Forecast lots to be costed. Complexity Factor is optional; a "
                "blank one carries the\nprevious lot's value forward (1.0 to "
                "start)."
            ),
            style="Sub.TLabel",
            justify="left",
        ).pack(anchor="w", padx=8, pady=(8, 6))

        self.grid_estimate = LotGrid(
            f,
            ["Fiscal Year", "Lot Quantity", "Complexity Factor"],
            [14, 14, 20],
        )
        self.grid_estimate.pack(fill="both", expand=True, padx=8)

        bar = ttk.Frame(f)
        bar.pack(fill="x", padx=8, pady=8)
        ttk.Button(
            bar, text="Add Row", command=self.grid_estimate.add_row
        ).pack(side="left")
        ttk.Button(
            bar,
            text="Delete Last Row",
            command=self.grid_estimate.delete_last,
        ).pack(side="left", padx=4)
        ttk.Button(
            bar, text="Clear", command=self.grid_estimate.clear
        ).pack(side="left")
        ttk.Button(
            bar,
            text="Load Example",
            command=lambda: self.grid_estimate.load(EXAMPLE_ESTIMATE),
        ).pack(side="right")

        for _ in range(8):
            self.grid_estimate.add_row()

    def _build_runinfo(self):
        f = self.tab_run
        box = ttk.LabelFrame(f, text="Run Info")
        box.pack(fill="x", padx=8, pady=10)

        self.var_runid = tk.StringVar(value=SETTINGS["DefaultRunID"])
        self.var_program = tk.StringVar(value=SETTINGS["DefaultProgram"])
        self.var_label = tk.StringVar(value=SETTINGS["DefaultRunLabel"])
        self.var_baseyear = tk.StringVar(value="")

        fields = [
            ("Run ID", self.var_runid, ""),
            ("Program", self.var_program, ""),
            ("Run label", self.var_label, ""),
            ("Base year ($)", self.var_baseyear, "blank = not stated"),
        ]
        for r, (lbl, var, hint) in enumerate(fields):
            ttk.Label(box, text=lbl + ":").grid(
                row=r, column=0, sticky="e", padx=8, pady=5
            )
            ttk.Entry(box, textvariable=var, width=42).grid(
                row=r, column=1, sticky="w", pady=5
            )
            if hint:
                ttk.Label(box, text=hint, style="Sub.TLabel").grid(
                    row=r, column=2, sticky="w", padx=8
                )

        box2 = ttk.LabelFrame(f, text="Model Settings")
        box2.pack(fill="x", padx=8, pady=6)

        self.var_costscale = tk.StringVar(
            value=str(SETTINGS["CostUnitScale"])
        )
        self.var_totalscale = tk.StringVar(value=str(SETTINGS["TotalScale"]))
        self.var_defaultcf = tk.StringVar(value=str(SETTINGS["DefaultCF"]))
        self.var_tgate = tk.StringVar(value=str(SETTINGS["TGate"]))
        self.var_fitprior = tk.StringVar(
            value=str(SETTINGS["FitPriorUnits"])
        )
        self.var_fcstprior = tk.StringVar(
            value=str(SETTINGS["FcstPriorUnits"])
        )
        self.var_toolmatch = tk.BooleanVar(
            value=SETTINGS["ToolMatchProjection"]
        )

        s_fields = [
            ("Cost unit scale", self.var_costscale, "1 = $K, 1000 = dollars"),
            ("Total scale", self.var_totalscale, "applied on top, for totals"),
            ("Default complexity", self.var_defaultcf, "used if none given"),
            ("t-gate", self.var_tgate, "significance cutoff for rate term"),
            ("Prior units (fit)", self.var_fitprior, "units built before lot 1"),
            ("Prior units (forecast)", self.var_fcstprior, ""),
        ]
        for r, (lbl, var, hint) in enumerate(s_fields):
            ttk.Label(box2, text=lbl + ":").grid(
                row=r, column=0, sticky="e", padx=8, pady=4
            )
            ttk.Entry(box2, textvariable=var, width=16).grid(
                row=r, column=1, sticky="w", pady=4
            )
            if hint:
                ttk.Label(box2, text=hint, style="Sub.TLabel").grid(
                    row=r, column=2, sticky="w", padx=8
                )
        ttk.Checkbutton(
            box2,
            text="Project Rate & LC+Rate on lot midpoint (tool match)",
            variable=self.var_toolmatch,
        ).grid(row=len(s_fields), column=0, columnspan=3, sticky="w", padx=8, pady=6)

    def _build_results(self):
        f = self.tab_results
        self.lbl_result = ttk.Label(
            f,
            text="No run yet. Fill in the lots, then click Run Model.",
            style="Sub.TLabel",
        )
        self.lbl_result.pack(anchor="w", padx=8, pady=8)

        cols = ("Item", "Value", "LC", "Rate", "LC+Rate")
        self.tree = ttk.Treeview(f, columns=cols, show="headings", height=22)
        widths = (250, 330, 120, 120, 120)
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        vs = ttk.Scrollbar(f, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vs.set)
        vs.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.tree.tag_configure("sel", background="#dff0d8")

    def _build_risk(self):
        f = self.tab_risk

        available = risk is not None and risk.AVAILABLE
        if not available:
            why = (
                "risk.py is not next to this script."
                if risk is None
                else risk.IMPORT_ERROR
            )
            ttk.Label(
                f,
                text=(
                    "Risk analysis unavailable.\n\n"
                    + (risk.INSTALL_HINT if risk is not None else why)
                    + f"\n\nDetail: {why}"
                ),
                style="Sub.TLabel",
                justify="left",
            ).pack(anchor="w", padx=12, pady=14)
            self.var_do_risk = tk.BooleanVar(value=False)
            self.tree_risk = None
            self.tree_iv = None
            return

        ttk.Label(
            f,
            text=(
                "Prediction intervals and a Monte Carlo of the whole buy, "
                "fitted by cost_core from the\ncost-risk-toolkit. It fits the "
                "exact lot average rather than the lot midpoint, so its "
                "parameters\nwill sit close to the LC model on tab 4 without "
                "matching it exactly. A wide gap is worth investigating."
            ),
            style="Sub.TLabel",
            justify="left",
        ).pack(anchor="w", padx=8, pady=(8, 4))

        ctl = ttk.LabelFrame(f, text="Settings")
        ctl.pack(fill="x", padx=8, pady=4)

        self.var_theory = tk.StringVar(value="crawford")
        self.var_method = tk.StringVar(value="ols")
        self.var_level = tk.StringVar(value="80")
        self.var_iters = tk.StringVar(value="20000")
        self.var_seed = tk.StringVar(value="11")
        self.var_rho = tk.StringVar(value="0.30")
        self.var_basis = tk.StringVar(value="recurring")
        self.var_resid = tk.BooleanVar(value=True)
        self.var_do_risk = tk.BooleanVar(value=True)

        def combo(row, col, label, var, values, width=12):
            ttk.Label(ctl, text=label + ":").grid(
                row=row, column=col * 2, sticky="e", padx=(8, 4), pady=4
            )
            w = ttk.Combobox(
                ctl, textvariable=var, values=values, width=width,
                state="readonly",
            )
            w.grid(row=row, column=col * 2 + 1, sticky="w", pady=4)
            return w

        def entry(row, col, label, var, width=12):
            ttk.Label(ctl, text=label + ":").grid(
                row=row, column=col * 2, sticky="e", padx=(8, 4), pady=4
            )
            ttk.Entry(ctl, textvariable=var, width=width).grid(
                row=row, column=col * 2 + 1, sticky="w", pady=4
            )

        combo(0, 0, "Theory", self.var_theory, list(risk.THEORIES))
        combo(0, 1, "Method", self.var_method, list(risk.METHODS))
        combo(0, 2, "Interval %", self.var_level, ["80", "90", "95"], 8)
        entry(1, 0, "Iterations", self.var_iters)
        entry(1, 1, "Seed", self.var_seed)
        entry(1, 2, "Lot correlation", self.var_rho, 8)
        combo(2, 0, "Cost basis", self.var_basis, ["recurring", "total"])
        ttk.Checkbutton(
            ctl,
            text="Include lot-to-lot scatter (prediction, not confidence)",
            variable=self.var_resid,
        ).grid(row=2, column=2, columnspan=4, sticky="w", padx=8)
        ttk.Checkbutton(
            ctl,
            text="Run this automatically with Run Model and add it to the workbook",
            variable=self.var_do_risk,
        ).grid(row=3, column=0, columnspan=6, sticky="w", padx=8, pady=(2, 6))

        bar = ttk.Frame(f)
        bar.pack(fill="x", padx=8, pady=(2, 6))
        ttk.Button(
            bar, text="Run Risk Analysis", command=self.run_risk
        ).pack(side="left")
        self.var_risk_status = tk.StringVar(
            value="Not run yet. Needs a base year on tab 3."
        )
        ttk.Label(
            bar, textvariable=self.var_risk_status, style="Sub.TLabel"
        ).pack(side="left", padx=10)

        panes = ttk.Panedwindow(f, orient="vertical")
        panes.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        top = ttk.Frame(panes)
        self.tree_risk = ttk.Treeview(
            top, columns=("Item", "Value"), show="headings", height=9
        )
        self.tree_risk.heading("Item", text="Item")
        self.tree_risk.heading("Value", text="Value")
        self.tree_risk.column("Item", width=230, anchor="w")
        self.tree_risk.column("Value", width=680, anchor="w")
        sv = ttk.Scrollbar(top, orient="vertical",
                           command=self.tree_risk.yview)
        self.tree_risk.configure(yscrollcommand=sv.set)
        sv.pack(side="right", fill="y")
        self.tree_risk.pack(fill="both", expand=True)
        panes.add(top, weight=3)

        bot = ttk.Frame(panes)
        iv_cols = (
            "Lot", "Qty", "First", "Last", "CF",
            "Unit Cost ($K)", "Low", "High", "Lot Cost ($)",
        )
        self.tree_iv = ttk.Treeview(
            bot, columns=iv_cols, show="headings", height=8
        )
        for c, w in zip(iv_cols, (45, 45, 55, 55, 50, 110, 95, 95, 130)):
            self.tree_iv.heading(c, text=c)
            self.tree_iv.column(c, width=w, anchor="e")
        sv2 = ttk.Scrollbar(bot, orient="vertical", command=self.tree_iv.yview)
        self.tree_iv.configure(yscrollcommand=sv2.set)
        sv2.pack(side="right", fill="y")
        self.tree_iv.pack(fill="both", expand=True)
        panes.add(bot, weight=2)

    def _risk_options(self) -> "risk.RiskOptions":
        def num(var, label, caster):
            try:
                return caster(var.get().strip())
            except ValueError:
                raise ValueError(f"Risk setting '{label}' must be a number.")

        year_txt = self.var_baseyear.get().strip()
        year = None
        if year_txt:
            try:
                year = int(float(year_txt))
            except ValueError:
                raise ValueError(
                    f"Base year '{year_txt}' is not a year. Set it on tab 3."
                )

        return risk.RiskOptions(
            theory=self.var_theory.get(),
            method=self.var_method.get(),
            level=num(self.var_level, "Interval %", float) / 100.0,
            n_iter=num(self.var_iters, "Iterations", int),
            seed=num(self.var_seed, "Seed", int),
            residual_correlation=num(self.var_rho, "Lot correlation", float),
            include_residual=bool(self.var_resid.get()),
            cost_basis=self.var_basis.get(),
            dollar_year=year,
            program=self.var_program.get().strip() or "unnamed program",
            simulate=True,
        )

    def _compute_risk(self):
        """Collect the grids and run the risk analysis. Returns a RiskResult."""
        analogy_df = self._collect_analogy()
        estimate_df = self._collect_estimate()
        cfg = SETTINGS.copy()
        cfg.update(self._collect_overrides())

        cf_raw = estimate_df["Complexity"].to_numpy(dtype=float)
        last = cfg["DefaultCF"]
        cf = []
        for v in cf_raw:
            if pd.isna(v) or v <= 0:
                cf.append(last)
            else:
                last = float(v)
                cf.append(last)

        return risk.run_risk(
            analogy_df["Qty"].to_numpy(dtype=float),
            analogy_df["AUC ($K)"].to_numpy(dtype=float),
            estimate_df["Qty"].to_numpy(dtype=int),
            cf,
            cfg,
            self._risk_options(),
        )

    def run_risk(self):
        if risk is None or not risk.AVAILABLE:
            messagebox.showinfo("Risk analysis unavailable", risk.INSTALL_HINT)
            return
        self.var_risk_status.set("Running...")
        self.update_idletasks()
        try:
            self.risk_result = self._compute_risk()
            self._show_risk(self.risk_result)
            self.var_risk_status.set(
                f"Ran {self.risk_result.n_iter:,} iterations."
                if self.risk_result.n_iter
                else "Intervals computed."
            )
            self.nb.select(self.tab_risk)
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("Risk analysis", str(exc))
            self.var_risk_status.set("Did not run.")
        except Exception as exc:
            messagebox.showerror(
                "Risk analysis failed",
                f"{type(exc).__name__}: {exc}\n\n"
                f"{traceback.format_exc(limit=3)}",
            )
            self.var_risk_status.set("Failed.")

    def _show_risk(self, res):
        self.tree_risk.delete(*self.tree_risk.get_children())
        for _, row in risk.summary_frame(res).iterrows():
            self.tree_risk.insert(
                "", "end", values=(str(row["Item"]), str(row["Value"]))
            )
        self.tree_iv.delete(*self.tree_iv.get_children())
        for _, r in res.intervals.iterrows():
            self.tree_iv.insert(
                "",
                "end",
                values=(
                    int(r["Lot"]),
                    int(r["Lot Quantity"]),
                    int(r["First Unit in Lot"]),
                    int(r["Last Unit in Lot"]),
                    f"{r['Complexity Factor']:.4g}",
                    f"{r['Unit Cost ($K)']:,.2f}",
                    f"{r['Unit Cost Low ($K)']:,.2f}",
                    f"{r['Unit Cost High ($K)']:,.2f}",
                    f"{r['Lot Cost ($)']:,.2f}",
                ),
            )

    def _build_actionbar(self):
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=12, pady=10)

        ttk.Label(bar, text="Save workbook to:").pack(side="left")
        self.var_outfile = tk.StringVar(
            value=os.path.join(
                default_output_dir(), "Lot_Cost_Model_Complete_Suite.xlsx"
            )
        )
        ttk.Entry(bar, textvariable=self.var_outfile).pack(
            side="left", fill="x", expand=True, padx=6
        )
        ttk.Button(bar, text="Browse...", command=self._browse).pack(
            side="left"
        )
        self.btn_run = ttk.Button(
            bar, text="Run Model", command=self.run_model
        )
        self.btn_run.pack(side="left", padx=(10, 0))

        self.var_status = tk.StringVar(value="Ready.")
        ttk.Label(
            self, textvariable=self.var_status, style="Sub.TLabel"
        ).pack(anchor="w", padx=12, pady=(0, 8))

    def _browse(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel workbook", "*.xlsx")],
            initialfile=os.path.basename(self.var_outfile.get()),
            initialdir=os.path.dirname(self.var_outfile.get())
            or default_output_dir(),
            title="Save model workbook as",
        )
        if path:
            self.var_outfile.set(path)

    # -- input parsing ------------------------------------------------------
    def _collect_analogy(self) -> pd.DataFrame:
        rows = self.grid_analogy.get_rows()
        if not rows:
            raise ValueError("No analogy lots entered (tab 1).")

        fy, qty, auc = [], [], []
        for i, r in enumerate(rows, start=1):
            try:
                q = parse_float(r[1])
            except ValueError:
                raise ValueError(
                    f"Analogy row {i}: Lot Quantity '{r[1]}' is not a number."
                )
            if q <= 0:
                raise ValueError(
                    f"Analogy row {i}: Lot Quantity must be greater than 0."
                )
            try:
                y = parse_float(r[0]) if r[0] else np.nan
            except ValueError:
                raise ValueError(
                    f"Analogy row {i}: Fiscal Year '{r[0]}' is not a number."
                )
            if r[2]:
                try:
                    c = parse_float(r[2])
                except ValueError:
                    raise ValueError(
                        f"Analogy row {i}: Unit Cost '{r[2]}' is not a number."
                    )
                if c <= 0:
                    raise ValueError(
                        f"Analogy row {i}: Unit Cost must be greater than 0 "
                        "(leave it blank for a quantity-only lot)."
                    )
            else:
                c = np.nan
            fy.append(y)
            qty.append(q)
            auc.append(c)

        n_costed = sum(1 for c in auc if pd.notna(c))
        if n_costed < 3:
            raise ValueError(
                "The learning curve needs at least 3 analogy lots with both a "
                f"quantity and a unit cost. Found {n_costed}."
            )

        return pd.DataFrame(
            {
                "Lot": list(range(1, len(rows) + 1)),
                "Lot FY": fy,
                "Qty": qty,
                "AUC ($K)": auc,
            }
        )

    def _collect_estimate(self) -> pd.DataFrame:
        rows = self.grid_estimate.get_rows()
        if not rows:
            raise ValueError("No estimate lots entered (tab 2).")

        fy, qty, cf = [], [], []
        for i, r in enumerate(rows, start=1):
            try:
                q = parse_float(r[1])
            except ValueError:
                raise ValueError(
                    f"Estimate row {i}: Lot Quantity '{r[1]}' is not a number."
                )
            if q <= 0:
                raise ValueError(
                    f"Estimate row {i}: Lot Quantity must be greater than 0."
                )
            try:
                y = parse_float(r[0]) if r[0] else np.nan
            except ValueError:
                raise ValueError(
                    f"Estimate row {i}: Fiscal Year '{r[0]}' is not a number."
                )
            if r[2]:
                try:
                    c = parse_float(r[2])
                except ValueError:
                    raise ValueError(
                        f"Estimate row {i}: Complexity Factor '{r[2]}' is not "
                        "a number."
                    )
            else:
                c = np.nan
            fy.append(y)
            qty.append(q)
            cf.append(c)

        return pd.DataFrame(
            {
                "Lot": list(range(1, len(rows) + 1)),
                "Lot FY": fy,
                "Qty": qty,
                "Complexity": cf,
            }
        )

    def _collect_overrides(self) -> dict:
        def num(var, label, caster=float):
            try:
                return caster(var.get().strip())
            except ValueError:
                raise ValueError(f"Setting '{label}' must be a number.")

        return {
            "CostUnitScale": num(self.var_costscale, "Cost unit scale"),
            "TotalScale": num(self.var_totalscale, "Total scale"),
            "DefaultCF": num(self.var_defaultcf, "Default complexity"),
            "TGate": num(self.var_tgate, "t-gate"),
            "FitPriorUnits": num(self.var_fitprior, "Prior units (fit)", int),
            "FcstPriorUnits": num(
                self.var_fcstprior, "Prior units (forecast)", int
            ),
            "ToolMatchProjection": bool(self.var_toolmatch.get()),
        }

    def _save_workbook(
        self, path, proj, summ, chart, risk_summ=None, risk_iv=None
    ) -> str | None:
        """Save, re-prompting if the path is locked or not writable."""
        while True:
            try:
                save_complete_excel_workbook(
                    path, proj, summ, chart, risk_summ, risk_iv
                )
                return path
            except PermissionError:
                retry = messagebox.askretrycancel(
                    "Cannot write the file",
                    f"Could not write to:\n{path}\n\n"
                    "The file may be open in Excel, or the folder may be "
                    "read-only.\n\nClose the file and click Retry, or Cancel "
                    "to choose a different location.",
                )
                if retry:
                    continue
                new = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    filetypes=[("Excel workbook", "*.xlsx")],
                    initialfile=os.path.basename(path),
                    initialdir=default_output_dir(),
                    title="Save model workbook as",
                )
                if not new:
                    return None
                path = new
                self.var_outfile.set(path)

    # -- run ----------------------------------------------------------------
    def run_model(self):
        self.btn_run.config(state="disabled")
        self.var_status.set("Running...")
        self.update_idletasks()
        try:
            analogy_df = self._collect_analogy()
            estimate_df = self._collect_estimate()
            overrides = self._collect_overrides()

            run_info = {
                "RunID": self.var_runid.get().strip()
                or SETTINGS["DefaultRunID"],
                "Program": self.var_program.get().strip()
                or SETTINGS["DefaultProgram"],
                "RunLabel": self.var_label.get().strip()
                or SETTINGS["DefaultRunLabel"],
                "BaseYear": self.var_baseyear.get().strip(),
            }

            projections_df, models_ctx = run_lot_cost_model(
                analogy_df, estimate_df, overrides
            )
            summary_df = generate_analyst_summary(models_ctx, run_info)
            chart_df = generate_fit_chart_data(models_ctx)

            risk_summary = risk_intervals = None
            if (
                risk is not None
                and risk.AVAILABLE
                and bool(self.var_do_risk.get())
            ):
                try:
                    self.risk_result = self._compute_risk()
                    self._show_risk(self.risk_result)
                    risk_summary = risk.summary_frame(self.risk_result)
                    risk_intervals = self.risk_result.intervals
                    self.var_risk_status.set(
                        f"Ran {self.risk_result.n_iter:,} iterations."
                    )
                except (ValueError, RuntimeError) as exc:
                    # A risk failure must not cost the analyst the main run.
                    self.var_risk_status.set("Did not run.")
                    messagebox.showwarning(
                        "Risk analysis skipped",
                        f"{exc}\n\nThe rest of the model ran normally.",
                    )

            path = self.var_outfile.get().strip()
            if not path:
                raise ValueError("Choose an output file first.")
            if not path.lower().endswith(".xlsx"):
                path += ".xlsx"
            saved = self._save_workbook(
                path,
                projections_df,
                summary_df,
                chart_df,
                risk_summary,
                risk_intervals,
            )

            self._show_results(summary_df)
            if saved:
                self.var_outfile.set(saved)
                self.var_status.set(f"Saved: {saved}")
                self.lbl_result.config(
                    text=(
                        f"Run complete. Workbook saved to:\n{saved}\n"
                        "Sheets: Analyst_Summary, Estimate_Projections, "
                        "Fit_Chart_Data (with 3 embedded charts)."
                    )
                )
                if messagebox.askyesno(
                    "Run complete",
                    f"Model ran successfully.\n\nSaved to:\n{saved}\n\n"
                    "Open the workbook now?",
                ):
                    try:
                        os.startfile(saved)
                    except Exception:
                        pass
            else:
                self.var_status.set("Run complete, workbook not saved.")
                self.lbl_result.config(
                    text="Run complete, but the workbook was not saved."
                )

        except ValueError as exc:
            messagebox.showerror("Check your input", str(exc))
            self.var_status.set("Input error.")
        except Exception as exc:
            messagebox.showerror(
                "Model error",
                f"{type(exc).__name__}: {exc}\n\n"
                f"{traceback.format_exc(limit=3)}",
            )
            self.var_status.set("Run failed.")
        finally:
            self.btn_run.config(state="normal")

    def _show_results(self, summary_df: pd.DataFrame):
        self.tree.delete(*self.tree.get_children())
        for _, row in summary_df.iterrows():
            vals = [
                str(row["Item"]),
                str(row["Value"]),
                str(row["LC"]),
                str(row["Rate"]),
                str(row["LC+Rate"]),
            ]
            tag = "sel" if vals[0] == "SELECTED" else ""
            self.tree.insert("", "end", values=vals, tags=(tag,))
        self.nb.select(self.tab_results)


# ============================================================================
# 8. EXECUTION BLOCK
# ============================================================================
if __name__ == "__main__":
    # Crisper text on high-DPI Windows displays.
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    try:
        app = LotCostApp()
    except tk.TclError as exc:
        print(f"Could not start the GUI: {exc}", file=sys.stderr)
        print("A desktop session is required to run this tool.", file=sys.stderr)
        sys.exit(1)

    app.mainloop()
