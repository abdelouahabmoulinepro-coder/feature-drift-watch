"""Compare tabular feature distributions without using the recent batch to bin PSI."""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import random
from pathlib import Path


def load_csv(path: Path, numeric: list[str], categorical: list[str]) -> tuple[list[dict], list[str]]:
    if len(set(numeric + categorical)) != len(numeric) + len(categorical):
        raise ValueError("Colonnes numériques et catégorielles sans doublons")
    if not numeric and not categorical:
        raise ValueError("Sélectionnez au moins une variable")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("CSV sans en-tête")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Noms de colonnes dupliqués")
        requested = numeric + categorical
        missing = [col for col in requested if col not in reader.fieldnames]
        if missing:
            raise ValueError("Variables absentes: " + ", ".join(missing))
        rows = []
        for line, raw in enumerate(reader, 2):
            row = {col: None for col in numeric}
            for col in numeric:
                value = (raw.get(col) or "").strip()
                if value:
                    try:
                        number = float(value)
                    except ValueError:
                        raise ValueError(f"Ligne {line}: {col} doit être numérique") from None
                    if not math.isfinite(number):
                        raise ValueError(f"Ligne {line}: {col} doit être fini")
                    row[col] = number
            for col in categorical:
                value = (raw.get(col) or "").strip()
                row[col] = value if value else None
            rows.append(row)
    if not rows:
        raise ValueError(f"Aucune observation dans {path}")
    return rows, list(reader.fieldnames)


def reference_edges(values: list[float], bins: int = 10) -> list[float]:
    """Unique internal quantile cut points estimated only from reference values."""
    if not values or bins < 2:
        return []
    ordered = sorted(values)
    if ordered[0] == ordered[-1]:
        # Preserve an explicit boundary around a constant reference feature.
        return [ordered[0], math.nextafter(ordered[0], math.inf)]
    cuts = set()
    for i in range(1, bins):
        index = math.ceil(i * len(ordered) / bins) - 1
        if index + 1 < len(ordered):
            cuts.add((ordered[index] + ordered[index + 1]) / 2)
    return sorted(cuts)


def _proportions(counts: list[int], epsilon: float) -> list[float]:
    total = sum(counts)
    if total == 0:
        return []
    denom = total + epsilon * len(counts)
    return [(count + epsilon) / denom for count in counts]


def population_stability_index(reference: list[float], current: list[float], bins: int = 10,
                               epsilon: float = 1e-6) -> tuple[float | None, list[float]]:
    if not reference or not current:
        return None, reference_edges(reference, bins)
    edges = reference_edges(reference, bins)
    ref_counts = [0] * (len(edges) + 1)
    cur_counts = [0] * (len(edges) + 1)
    for value in reference:
        ref_counts[bisect.bisect_right(edges, value)] += 1
    for value in current:
        cur_counts[bisect.bisect_right(edges, value)] += 1
    p, q = _proportions(ref_counts, epsilon), _proportions(cur_counts, epsilon)
    psi = sum((new - old) * math.log(new / old) for old, new in zip(p, q))
    return psi, edges


def jensen_shannon(reference: list[str], current: list[str]) -> float | None:
    if not reference or not current:
        return None
    categories = sorted(set(reference) | set(current))
    rc = [reference.count(cat) for cat in categories]
    cc = [current.count(cat) for cat in categories]
    p, q = _proportions(rc, 0), _proportions(cc, 0)
    midpoint = [(a + b) / 2 for a, b in zip(p, q)]
    def kl(a, b):
        return sum(x * math.log2(x / y) for x, y in zip(a, b) if x > 0)
    return (kl(p, midpoint) + kl(q, midpoint)) / 2


def compare(reference: list[dict], current: list[dict], numeric: list[str], categorical: list[str],
            psi_threshold: float = .2, js_threshold: float = .1,
            missing_delta_threshold: float = .1, bins: int = 10) -> dict:
    if not reference or not current:
        raise ValueError("Les deux jeux doivent contenir au moins une ligne")
    if len(set(numeric + categorical)) != len(numeric) + len(categorical) or not (numeric or categorical):
        raise ValueError("Sélectionnez des variables sans doublons")
    if not 0 <= psi_threshold or not 0 <= js_threshold <= 1 or not 0 <= missing_delta_threshold <= 1 or bins < 2:
        raise ValueError("Seuils non valides")
    numeric_results, categorical_results = {}, {}
    for col in numeric:
        ref_all = [row.get(col) for row in reference]
        cur_all = [row.get(col) for row in current]
        ref_missing = sum(value is None for value in ref_all) / len(ref_all)
        cur_missing = sum(value is None for value in cur_all) / len(cur_all)
        ref_values = [value for value in ref_all if value is not None]
        cur_values = [value for value in cur_all if value is not None]
        psi, edges = population_stability_index(ref_values, cur_values, bins)
        drifted = (psi is not None and psi >= psi_threshold) or abs(cur_missing - ref_missing) >= missing_delta_threshold
        numeric_results[col] = {"psi": psi, "reference_edges": edges,
            "reference_non_missing": len(ref_values), "current_non_missing": len(cur_values),
            "reference_missing_rate": ref_missing, "current_missing_rate": cur_missing,
            "missing_rate_delta": cur_missing - ref_missing, "alert": drifted}
    for col in categorical:
        ref_all = [row.get(col) for row in reference]
        cur_all = [row.get(col) for row in current]
        ref_missing = sum(value is None for value in ref_all) / len(ref_all)
        cur_missing = sum(value is None for value in cur_all) / len(cur_all)
        ref_values = [value for value in ref_all if value is not None]
        cur_values = [value for value in cur_all if value is not None]
        js = jensen_shannon(ref_values, cur_values)
        drifted = (js is not None and js >= js_threshold) or abs(cur_missing - ref_missing) >= missing_delta_threshold
        categorical_results[col] = {"jensen_shannon_divergence": js,
            "reference_non_missing": len(ref_values), "current_non_missing": len(cur_values),
            "reference_missing_rate": ref_missing, "current_missing_rate": cur_missing,
            "missing_rate_delta": cur_missing - ref_missing, "alert": drifted}
    alerts = [name for name, result in {**numeric_results, **categorical_results}.items() if result["alert"]]
    return {"reference_rows": len(reference), "current_rows": len(current),
        "numeric_threshold_psi": psi_threshold, "categorical_threshold_js": js_threshold,
        "missing_rate_delta_threshold": missing_delta_threshold,
        "numeric": numeric_results, "categorical": categorical_results,
        "alerted_features": alerts, "status": "drift_alert" if alerts else "no_alert"}


def write_report(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "drift.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Feature drift report", "", f"- Reference rows: **{result['reference_rows']}**; current rows: **{result['current_rows']}**.",
        f"- Status: **{result['status']}**; alerted features: **{', '.join(result['alerted_features']) if result['alerted_features'] else 'none'}**.", "",
        "| Feature | Metric | Value | Missing rate change | Alert |", "|---|---|---:|---:|---|"]
    for col, item in result["numeric"].items():
        val = "n/a" if item["psi"] is None else f"{item['psi']:.4f}"
        lines.append(f"| {col} | PSI | {val} | {item['missing_rate_delta']:+.1%} | {'yes' if item['alert'] else 'no'} |")
    for col, item in result["categorical"].items():
        val = "n/a" if item["jensen_shannon_divergence"] is None else f"{item['jensen_shannon_divergence']:.4f}"
        lines.append(f"| {col} | Jensen-Shannon | {val} | {item['missing_rate_delta']:+.1%} | {'yes' if item['alert'] else 'no'} |")
    lines.extend(["", "PSI uses quantile cut points learned from the reference set only. Categorical divergence uses the union of observed categories.", "",
        "## Limits", "",
        "These are descriptive thresholds, not significance tests or proof of model degradation. PSI and Jensen-Shannon values depend on binning, "
        "sample size, category granularity, and the selected variables. Review upstream changes and model quality metrics before acting.", ""])
    if result.get("added_columns") or result.get("removed_columns"):
        lines.extend(["## Schema changes", "", f"- Added: {', '.join(result['added_columns']) or 'none'}.",
                      f"- Removed: {', '.join(result['removed_columns']) or 'none'}.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def generate_demo(reference_path: Path, current_path: Path, seed: int = 20261115) -> None:
    rng = random.Random(seed)
    fields = ["tenure_days", "order_value_eur", "channel"]
    with reference_path.open("w", encoding="utf-8", newline="") as ref_stream, current_path.open("w", encoding="utf-8", newline="") as cur_stream:
        rw, cw = csv.DictWriter(ref_stream, fieldnames=fields), csv.DictWriter(cur_stream, fieldnames=fields)
        rw.writeheader(); cw.writeheader()
        for _ in range(500):
            rw.writerow({"tenure_days": max(1, int(rng.gauss(45, 12))), "order_value_eur": f"{max(5, rng.gauss(48, 14)):.2f}",
                         "channel": "web" if rng.random() < .72 else "store"})
        for i in range(500):
            cw.writerow({"tenure_days": max(1, int(rng.gauss(49, 14))),
                         "order_value_eur": "" if i % 25 == 0 else f"{max(5, rng.gauss(70, 22)):.2f}",
                         "channel": "web" if rng.random() < .43 else "store"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Surveille le décalage entre données de référence et données récentes")
    subs = parser.add_subparsers(dest="command", required=True)
    demo = subs.add_parser("demo", help="génère deux lots fictifs reproductibles")
    demo.add_argument("--output", type=Path, default=Path("demo-output"))
    run = subs.add_parser("compare", help="compare deux CSV")
    run.add_argument("reference", type=Path); run.add_argument("current", type=Path)
    run.add_argument("--numeric", nargs="*", default=[]); run.add_argument("--categorical", nargs="*", default=[])
    run.add_argument("--psi-threshold", type=float, default=.2); run.add_argument("--js-threshold", type=float, default=.1)
    run.add_argument("--missing-delta-threshold", type=float, default=.1); run.add_argument("--bins", type=int, default=10)
    run.add_argument("--output", type=Path, default=Path("drift-output"))
    args = parser.parse_args()
    if args.command == "demo":
        args.output.mkdir(parents=True, exist_ok=True)
        rp, cp = args.output / "reference.csv", args.output / "current.csv"
        generate_demo(rp, cp)
        numeric, categorical = ["tenure_days", "order_value_eur"], ["channel"]
        ref, _ = load_csv(rp, numeric, categorical); cur, _ = load_csv(cp, numeric, categorical)
        result = compare(ref, cur, numeric, categorical)
    else:
        ref, ref_cols = load_csv(args.reference, args.numeric, args.categorical)
        cur, cur_cols = load_csv(args.current, args.numeric, args.categorical)
        result = compare(ref, cur, args.numeric, args.categorical, args.psi_threshold, args.js_threshold,
                         args.missing_delta_threshold, args.bins)
        result["added_columns"] = sorted(set(cur_cols) - set(ref_cols))
        result["removed_columns"] = sorted(set(ref_cols) - set(cur_cols))
    write_report(result, args.output)
    print(f"Statut: {result['status']} | alertes: {len(result['alerted_features'])} | rapport: {args.output / 'report.md'}")


if __name__ == "__main__": main()
