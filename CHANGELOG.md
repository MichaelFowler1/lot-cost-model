# Changelog

All notable changes to Lot Cost Model are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries say what moved and by how much. A number that changes is not a
housekeeping detail here, because someone may have put the old one in a budget.

## [Unreleased]

### Changed

- The library pin moves from cost-core 1.0.0 to 1.0.1. No number moves with
  it: 1.0.1 changes nothing but the library's own test suite, which 1.0.0's
  shipped in a form that failed off Windows, and the version it reports. The engine, the roll-up and the workbook
  writers are byte for byte what 1.0.0 installed, and the one visible
  difference is the `cost_core version` row of the Analyst_Summary sheet.

## [3.0.0] - 2026-09-10

The estimating engine, the WBS roll-up and the Excel workbook writers have
left this repository for [cost-core](https://github.com/MichaelFowler1/cost-risk-toolkit),
and this tool is now the window onto it. There was a private copy of the
engine here and a slightly different one there; now there is one.

### Your numbers move

Upgrading changes estimates, and every change comes from one of two
corrections in the library rather than from anything in the window. Re-run
anything you have briefed and compare before you rely on a 2.x figure.

- **Fitted coefficients move by about a part in a hundred billion.** The
  library fits through a shared estimator in place of the normal equations
  this tool carried, and on these designs the normal equations lose four or
  five digits: against an exact solve in sixty digits they sat 1.1e-11 out
  and the shared estimator sits 1.5e-14. Model selection does not change on
  any of the 36 cases the library pins, and the analyst summary prints
  identically.
- **A projected lot cost can move by a cent** where the old value sat right
  on a half-cent rounding boundary. One cell in the library's pinned corpus
  does.
- **A run with its convergence tolerance tightened well below the default can
  now report converged where it reported not converged.** The old solver
  cycled just above a 1e-14 tolerance and ran out its iterations; the new one
  converges. The fitted slope is the same to twelve digits.
- **The unbiased refits on the method comparison move by up to about 1.5e-6**
  relative, because the shared estimator's MUPE and ZMPE stop on their own
  inner tolerance. The ZMPE row was never reproducible better than about 1e-7
  against itself.
- **Program percentiles on tab 6 fall, and the P80 reserve falls by about a
  quarter.** Each fitted element used to be summarised as a lognormal before
  being correlated with the others, and that summary ran high through the
  shoulder of the distribution, which is where the P80 lives. The roll-up now
  correlates each element's own simulated draws. On the library's reference
  programmes the P80 falls 0.27% to 0.29% and the reserve to P80 falls from
  2,168,822 to 1,599,303 on one of them. Each element's percentiles inside the
  program now equal its percentiles on its own, exactly, where they used to
  sit about half a percent out.

### Changed

- **cost_core is required.** It used to be optional, lighting up only the
  risk halves of tabs 5 and 6. It is now where every number comes from, so
  the tool does not start without it, and it checks at startup that it has a
  1.x release rather than failing later on a renamed column.
- **The tool needs scipy**, which arrives with cost_core. It supplies the t,
  chi-square and normal quantiles the intervals and the Monte Carlo are built
  on. matplotlib is not needed.
- **The one-file archive vendors the library.** `tools/build_pyz.py` copies
  the installed cost_core into `lot-cost-model.pyz`, so the archive is still a
  single file. It runs with numpy, pandas, openpyxl and scipy installed and
  nothing else. It is built compressed now, which makes it about 210KB rather
  than three times that.
- The Analyst_Summary sheet carries a `cost_core version` row beside the tool
  version, because the two now move independently: a change to the window
  touches no number, and a change to the library touches no window.
- The Python floor stays at 3.9, and CI runs 3.9 through 3.14 in one job with
  the library installed, plus a job that builds the archive and runs it from a
  bare environment.
- Saved run files keep their format. Four real run files now sit in
  `tests/run_fixtures`, and a test loads every file it finds there, so a
  format change that breaks an old file fails the suite.

### Removed

- `wbs.py`. The roll-up lives in `cost_core.program`, and the window imports
  it under the same name, so nothing about tab 6 looks different.
- The engine half of `lot_cost_model.py`, about two thousand lines: the model
  fits, the analyst summary, the fit chart data and both workbook writers.
- The without-cost_core mode, its install hints and the CI job that checked
  it. There is nothing left to run without the library.

## [2.1.0] - 2026-08-21

- Projections satisfy the fitted equation. Rate and LC+Rate used to be
  projected on the wrong variable and without their rate term, which
  overstated them. `LegacyRateOmission` reproduces the old behaviour for
  reconciling a pre-2.1.0 workbook, and the old `ToolMatchProjection` setting
  is refused rather than ignored.
- Runs can be saved and reopened, and every workbook says which build made it.

[Unreleased]: https://github.com/MichaelFowler1/lot-cost-model/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/MichaelFowler1/lot-cost-model/releases/tag/v3.0.0
[2.1.0]: https://github.com/MichaelFowler1/lot-cost-model/commits/main
