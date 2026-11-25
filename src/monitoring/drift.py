"""Telemetry sensor drift detection using statistical KS-test and PSI monitoring."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def calculate_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """Calculates Population Stability Index (PSI) between reference and current samples."""
    # Filter out NaNs/Infs
    ref_clean = reference[np.isfinite(reference)]
    cur_clean = current[np.isfinite(current)]
    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    # Create quantile bins based on reference distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(ref_clean, quantiles)
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    # Ensure range covers all data
    bins[0] = -np.inf
    bins[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=bins)
    cur_counts, _ = np.histogram(cur_clean, bins=bins)

    # Convert to proportions with smoothing to avoid log(0)
    eps = 1e-4
    ref_props = (ref_counts + eps) / (len(ref_clean) + eps * len(ref_counts))
    cur_props = (cur_counts + eps) / (len(cur_clean) + eps * len(cur_counts))

    psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    return float(psi)


def detect_sensor_drift(
    current_data_path: str,
    reference_data_path: str = "data/reference/reference_features.parquet",
    report_output_html: str = "reports/drift_report.html",
    report_output_json: str = "reports/drift_metrics.json",
    config_path: str = "params.yaml",
) -> dict:
    """Computes sensor drift via Kolmogorov-Smirnov 2-sample test and PSI."""
    ref_path = Path(reference_data_path)
    cur_path = Path(current_data_path)

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference features not found at {ref_path}. Run training first.")
    if not cur_path.exists():
        raise FileNotFoundError(f"Current telemetry batch not found at {cur_path}.")

    ks_threshold = 0.05
    psi_threshold = 0.1
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            ks_threshold = cfg.get("monitoring", {}).get("drift_threshold_ks", 0.05)
            psi_threshold = cfg.get("monitoring", {}).get("psi_threshold", 0.1)

    ref_df = pd.read_parquet(ref_path)
    cur_df = pd.read_parquet(cur_path)

    common_cols = [c for c in ref_df.columns if c in cur_df.columns]
    metrics = {
        "drift_detected": False,
        "number_of_columns": len(common_cols),
        "columns_drifted": [],
        "metrics_by_column": {},
    }

    html_rows = []
    for col in common_cols:
        ref_vals = ref_df[col].to_numpy()
        cur_vals = cur_df[col].to_numpy()

        # Kolmogorov-Smirnov 2-sample test
        ks_res = stats.ks_2samp(ref_vals, cur_vals)
        ks_stat = float(ks_res.statistic)
        ks_pvalue = float(ks_res.pvalue)

        # PSI
        psi_val = calculate_psi(ref_vals, cur_vals)

        is_drifted = bool(ks_pvalue < ks_threshold or psi_val > psi_threshold)
        if is_drifted:
            metrics["columns_drifted"].append(col)

        col_metric = {
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(ks_pvalue, 6),
            "psi": round(psi_val, 4),
            "drift_detected": is_drifted,
        }
        metrics["metrics_by_column"][col] = col_metric

        badge_text = "DRIFT" if is_drifted else "STABLE"
        html_rows.append(
            f"<tr><td>{col}</td><td>{ks_stat:.4f}</td><td>{ks_pvalue:.4e}</td><td>{psi_val:.4f}</td>"
            f"<td><span style='color: {'#dc2626' if is_drifted else '#16a34a'}; font-weight: bold;'>{badge_text}</span></td></tr>"
        )

    metrics["drift_detected"] = len(metrics["columns_drifted"]) > 0

    # Write JSON report
    Path(report_output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(report_output_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # Write HTML report
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>SMART Telemetry Sensor Drift Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 24px; background: #f8fafc; color: #1e293b; }}
        h1 {{ margin-bottom: 8px; }}
        .summary {{ margin-bottom: 24px; font-size: 16px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f1f5f9; font-weight: 600; font-size: 14px; }}
    </style>
</head>
<body>
    <h1>SMART Telemetry Drift Analysis</h1>
    <div class="summary">
        Overall Status: <strong>{"DRIFT DETECTED" if metrics["drift_detected"] else "ALL SENSORS STABLE"}</strong> |
        Drifted Features: {len(metrics["columns_drifted"])} / {len(common_cols)}
    </div>
    <table>
        <thead>
            <tr><th>Sensor Feature</th><th>KS Statistic</th><th>KS p-value</th><th>PSI</th><th>Status</th></tr>
        </thead>
        <tbody>
            {"".join(html_rows)}
        </tbody>
    </table>
</body>
</html>"""
    with open(report_output_html, "w") as f:
        f.write(html_content)

    print(f"Drift report generated at: {report_output_html} and {report_output_json}")
    return metrics
