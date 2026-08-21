# Lot Cost Model

A desktop tool for fitting learning curves to historical production lots and
projecting unit costs for future buys. You type your lots into a window, click
Run, and get an Excel workbook back with the fit statistics, the lot-by-lot
projections, and three charts.

It fits three competing models and tells you which one it picked and why.

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

Python 3.9 or newer, plus:

```
pip install numpy pandas openpyxl
```

The interface uses tkinter, which already ships with Python, so there's nothing
else to install.

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

Three sheets:

| Sheet | What's on it |
|---|---|
| `Analyst_Summary` | Side by side comparison of all three models, with the selection called out |
| `Estimate_Projections` | Row per forecast lot: midpoint, unit cost, lot cost before and after complexity, plus every fit statistic |
| `Fit_Chart_Data` | How each model did against the historical lots, with residuals, and three scatter charts |

![Results tab](docs/screenshot-results.png)

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
