"""Shared fixtures. Nothing here needs a display: the tests exercise the model
and the workbook writer directly, never the Tk window."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def analogy_df() -> pd.DataFrame:
    """The bundled demo lots, in the shape run_lot_cost_model expects.

    Kept in step with EXAMPLE_ANALOGY in the main module; a test asserts they
    have not drifted apart.
    """
    return pd.DataFrame(
        {
            "Lot": [1, 2, 3, 4, 5, 6],
            "Lot FY": [2015, 2016, 2017, 2018, 2019, 2020],
            "Qty": [5.0, 9.0, 14.0, 22.0, 34.0, 50.0],
            "AUC ($K)": [857.91, 645.57, 531.74, 437.51, 380.10, 332.21],
        }
    )


@pytest.fixture(scope="session")
def estimate_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Lot": [1, 2, 3, 4, 5, 6],
            "Lot FY": [2028, 2029, 2030, 2031, 2032, 2033],
            "Qty": [12.0, 20.0, 30.0, 40.0, 25.0, 10.0],
            "Complexity": [1.15] * 6,
        }
    )


@pytest.fixture(scope="session")
def analogy_arrays(analogy_df):
    return (
        analogy_df["Qty"].to_numpy(dtype=float),
        analogy_df["AUC ($K)"].to_numpy(dtype=float),
    )


@pytest.fixture(scope="session")
def cfg() -> dict:
    return {
        "CostUnitScale": 1.0,
        "TotalScale": 1000.0,
        "FitPriorUnits": 0,
        "FcstPriorUnits": 0,
        "DefaultCF": 1.0,
    }
