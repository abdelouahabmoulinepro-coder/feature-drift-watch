# Feature drift report

- Reference rows: **500**; current rows: **500**.
- Status: **drift_alert**; alerted features: **order_value_eur**.

| Feature | Metric | Value | Missing rate change | Alert |
|---|---|---:|---:|---|
| tenure_days | PSI | 0.1609 | +0.0% | no |
| order_value_eur | PSI | 1.4483 | +4.0% | yes |
| channel | Jensen-Shannon | 0.0855 | +0.0% | no |

PSI uses quantile cut points learned from the reference set only. Categorical divergence uses the union of observed categories.

## Limits

These are descriptive thresholds, not significance tests or proof of model degradation. PSI and Jensen-Shannon values depend on binning, sample size, category granularity, and the selected variables. Review upstream changes and model quality metrics before acting.
