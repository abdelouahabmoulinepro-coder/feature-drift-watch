# Feature Drift Watch

A dependency-free command-line tool for comparing a reference dataset with a newer batch. It measures numeric distribution changes with Population Stability Index (PSI), categorical changes with Jensen-Shannon divergence, and missing-rate shifts. It can also report columns added or removed between the two CSV schemas.

## Business use

Data and ML teams can use this as a lightweight monitoring step after a pipeline refresh or before scoring a new batch. It helps identify which selected features changed enough to investigate. It does not decide whether a model is still accurate.

## Quick start

Requires Python 3.10+; no third-party packages.

```bash
python drift_watch.py demo --output demo-output
python -m unittest -v
```

Compare your own CSV files:

```bash
python drift_watch.py compare reference.csv current.csv \
  --numeric tenure_days order_value_eur \
  --categorical channel country \
  --psi-threshold 0.20 --js-threshold 0.10 \
  --missing-delta-threshold 0.10 --output drift-report
```

The report folder contains `drift.json` with per-feature metrics and `report.md` for review. Empty cells are treated as missing. Selected numeric values must parse as finite numbers.

## Metrics

- **PSI**: numeric features are divided into up to ten equal-frequency bins using cut points calculated from the reference values only. Current values are assigned to those same reference bins. A small pseudocount keeps empty bins finite.
- **Jensen-Shannon divergence**: categorical features are compared over the union of categories present in either batch; the result uses base-2 logarithms and ranges from zero to one.
- **Missing-rate delta**: current missing share minus reference missing share, reported for every selected feature.
- **Schema changes**: the CLI lists added and removed columns separately from feature metrics.

Default alert cutoffs are PSI 0.20, categorical divergence 0.10, and an absolute missing-rate change of 10 percentage points. They are configurable screening thresholds, not universal statistical or operational standards. An alert is raised if either the distribution threshold or missingness threshold is crossed. A feature with no non-missing values in one batch has no distribution metric, but missingness is still reported.

## Reproducible demo result

The fixed-seed example compares 500 fictional reference rows with 500 fictional current rows. It flags `order_value_eur` (PSI **1.4483**, missing rate up 4 points), while `tenure_days` (PSI 0.1609) and `channel` (Jensen-Shannon 0.0855) remain below the configured cutoffs. These results come only from synthetic data.

## Limits

These descriptive statistics are sensitive to binning, sample size, category granularity, and feature selection. They are not significance tests and do not establish that model quality has changed. A drift alert should prompt inspection of upstream data, segment composition, and actual outcome metrics. PSI on a constant reference feature uses explicit boundaries around that value so newly observed values register as drift.

All sample data are generated locally with a fixed seed. No external or private data are included.

## License

MIT. See [LICENSE](LICENSE).
