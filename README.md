# Lot Cost Model

[![tests](https://github.com/MichaelFowler1/lot-cost-model/actions/workflows/tests.yml/badge.svg)](https://github.com/MichaelFowler1/lot-cost-model/actions/workflows/tests.yml)

A desktop tool for fitting learning curves to historical production lots and
projecting unit costs for future buys. You type your lots into a window, click
Run, and get an Excel workbook back with the fit statistics, the lot-by-lot
projections, and three charts.

It fits three competing models, tells you which one it picked and why, and can
put a prediction interval and a Monte Carlo around the answer.

![Data entry window](docs/screenshot-input.png)

## What it does

Give it a set of historical lots (fiscal year, quantity, average unit cost) and
a buy profile you want costed (fiscal year, quantity, complexity factor). It
then fits:

- **LC**, a plain learning curve on the lot midpoint
- **Rate**, cost against lot quantity only
- **LC+Rate**, both terms together

The learning curve fit isn't a simple regression. Lot midpoint depends on the
learning slope, and the slope depends on the midpoint, so the solver iterates
until the exponent stops moving. That's the same goal-seek loop you'd set up by
hand in Excel, just automated.

Model selection runs on the t-statistic of the rate coefficient with AICc as a
tiebreaker. If the rate term isn't significant, you get the plain learning
curve. If your lot quantities are too uniform to say anything about rate, the
rate models get gated off entirely and the summary tells you so instead of
quietly handing you a garbage coefficient.

## Requirements

Python 3.10 or newer, plus:

```
pip install numpy pandas openpyxl
```

The interface uses tkinter, which already ships with Python, so there's nothing
else to install.

### Optional: risk analysis

Tab 5 adds prediction intervals and a Monte Carlo of the whole buy. That part
leans on `cost_core` from the
[cost-risk-toolkit](https://github.com/MichaelFowler1/cost-risk-toolkit),
which needs Python 3.11 or newer:

```
pip install git+https://github.com/MichaelFowler1/cost-risk-toolkit.git
```

Skip it and everything else still works. The tab just shows you how to install
it, and the workbook comes out with its usual three sheets.

## Running it

```
python lot_cost_model.py
```

Fill in tabs 1 and 2, set your run info on tab 3, then click Run Model. Results
land on tab 4 and the workbook saves wherever the path box points.

Both grids take a paste straight from Excel. Copy your columns, click the first
cell, press Ctrl+V, and it fills the rows and adds more as needed. Tabs, commas,
and plain spaces all work as separators.

Each grid also has a Load Example button if you just want to see it run.

## A few things worth knowing

Leave the unit cost blank on an analogy lot to make it quantity-only. Those
units still count toward cumulative production and push later lots further down
the curve, but the lot itself doesn't get fit. Handy when you know a buy
happened but the cost is no good.

Leave a complexity factor blank and it carries the previous lot's value
forward, so you only type it when it changes.

You need at least three analogy lots with both a quantity and a cost. Four
before the rate models will even be attempted.

## Output

| Sheet | What's on it |
|---|---|
| `Analyst_Summary` | Side by side comparison of all three models, with the selection called out |
| `Estimate_Projections` | Row per forecast lot: midpoint, unit cost, lot cost before and after complexity, plus every fit statistic |
| `Fit_Chart_Data` | How each model did against the historical lots, with residuals, and three scatter charts |
| `Risk_Summary` | Fitted parameters, the buy total with its interval, the simulated percentiles, and the assumptions behind them |
| `Risk_Intervals` | Row per forecast lot with low and high bounds, and a chart of the band |
| `Risk_SCurve` | The simulated buy total at every percentile, with the S-curve charted and P50 and P80 marked |

The last three appear only when `cost_core` is installed and the risk analysis
ran.

![Results tab](docs/screenshot-results.png)

## Risk and prediction intervals

The deterministic tabs give you a point estimate. They say nothing about how
much confidence it deserves, which is the first thing anybody reviewing an
estimate asks. Tab 5 answers it.

![Risk tab](docs/screenshot-risk.png)

Rather than write the statistics a second time, this hands the same lots to
`cost_core` and reports what comes back: a prediction interval on every
forecast lot, and a Monte Carlo of the total buy with P50, P80 and P90. The
simulation propagates two things, parameter uncertainty in the fitted slope and
T1, which dominates on a short series, and lot-to-lot scatter, which is what
makes the answer a prediction about a real lot rather than a statement about
where the line sits. Future lots are correlated at 0.30 by default because
consecutive lots share a workforce and a schedule, and pretending otherwise
lets the shocks cancel and understates the spread of the whole buy.

The handoff is deliberately thin. `cost_core` is not asked to fit anything of
its own: `projection_intervals` and `simulate_buy` take the very objects
`run_lot_cost_model` already returned, so the intervals and the distribution
describe **this tool's own selected model**, on its own lot positions, with the
complexity factors already applied.

That matters for a reason worth stating plainly. The number under the
distribution is the same number on the projections sheet, identical by
construction rather than by luck, so the P80 cannot quietly belong to a
slightly different estimate than the one being briefed. There is no separate
theory or fitting method to choose here, because there is no separate fit.

The cost is that you lose an independent second opinion. An earlier version did
refit through `cost_core` with a different estimator, which gave a genuine
cross-check but meant two point estimates that had to be reconciled. Agreement
by construction was the better trade.

### The S-curve

Five percentiles in a table is a thin way to show a distribution, so the whole
thing gets charted. Cost runs along the bottom and cumulative probability up
the side, which is the orientation you read a P80 off. P50 and P80 are marked
and labelled with their cost.

![Cost S-curve](docs/scurve.png)

Read it as: pick a number on the bottom axis, and the curve tells you the
chance the buy comes in at or below it.

### What it does not cover

Schedule risk, requirement changes, and rate changes the history never saw.
It's production cost risk conditional on the program continuing as it has been,
which is a narrower claim than a full risk model.

The complexity factor scales the interval as a certain multiplier. Its own
uncertainty isn't modelled.

And if you're pricing from unit 1, you're using the curve as an analogy for a
different program. Whether the slope carries across is a judgement about
product, process, rate and contractor. Nothing in the data can confirm it, and
the extra error it introduces is in none of the intervals. The tab says so on
every run.

## Tests

```
pip install -r requirements-dev.txt
pytest
```

71 tests with `cost_core` installed, 35 without (the risk ones skip). CI runs
both, because "works when the optional dependency is missing" is a claim worth
checking rather than asserting.

A few of them exist for a specific reason. The most important assert that the
risk numbers describe the same thing the projections sheet does: the same
selected model, one interval row per forecast lot, and a simulated
distribution whose point estimate matches the sheet's buy total. If those ever
drift apart the tool would be showing a distribution around a number it is not
displaying, which is the failure worth catching early.

That check earned its place. An earlier bridge reached into a private
`cost_core` hook, and CI caught the toolkit removing it within minutes of the
first run.

Others read chart features back out of the workbook XML. openpyxl raises
nothing when you assign to an attribute a class does not have, so a chart can
come out perfectly valid with the data labels silently missing, which is
exactly what shipped for a while. Those assertions parse the XML rather than
matching strings, since openpyxl serialises differently depending on whether
lxml happens to be installed.

## About the example data

The numbers behind the Load Example buttons are made up. I generated them from
a 90% learning curve with a $1,000K first unit and a bit of random scatter, so
the tool has something realistic to chew on without shipping anybody's real
program data.

It doubles as a sanity check. Run the example and the model recovers a T1 of
about $1,011 and a slope near 89.7%, which is close to the truth it was built
from.

## License

MIT. See [LICENSE](LICENSE).
