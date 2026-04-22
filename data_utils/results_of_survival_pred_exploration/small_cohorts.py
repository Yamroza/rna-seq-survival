"""
Aggregate small_cohorts results into a summary table (c_index and c_index_ipcw avg ± std).
Produces both a CSV and a formatted Excel file.

Usage:
    python aggregate_results.py --results-dir results --output-dir results/summary
"""

import os
import json
import argparse

import pandas as pd

# ---------------------------------------------------------------------------
# Filename patterns → model label
# ---------------------------------------------------------------------------

MODEL_PATTERNS = {
    "scgpt": "scGPT",
    "tgpt":  "tGPT",
}

def detect_model(filename: str) -> str:
    """Infer model name from filename."""
    lower = filename.lower()
    for key, label in MODEL_PATTERNS.items():
        if f"_{key}_" in lower or lower.startswith(key):
            return label
    return "Raw"


def extract_cohort(filename: str) -> str:
    """Extract cohort name (e.g. GBM, BRCA) from filename."""
    stem = filename.replace("Final_results_", "").replace(".json", "")
    # Cohort is the part after TCGA- up to the next dot or underscore
    if "TCGA-" in stem:
        after = stem.split("TCGA-")[1]
        cohort = after.split(".")[0].split("_")[0]
        return cohort
    return stem


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_json_file(path: str) -> dict:
    """Return Summary metrics from a result JSON file."""
    with open(path) as f:
        data = json.load(f)
    summary = data.get("Summary", {})
    return {
        "c_index_avg":      summary.get("c_index", {}).get("avg"),
        "c_index_std":      summary.get("c_index", {}).get("std"),
        "c_index_ipcw_avg": summary.get("c_index_ipcw", {}).get("avg"),
        "c_index_ipcw_std": summary.get("c_index_ipcw", {}).get("std"),
    }


def collect_results(results_dir: str) -> pd.DataFrame:
    """Walk results directory and collect all small_cohorts JSON results."""
    rows = []
    for subdir, dirs, files in os.walk(results_dir):
        if os.path.basename(subdir) != "small_cohorts":
            continue
        for fname in files:
            if not fname.endswith(".json") or not fname.startswith("Final_results_"):
                continue
            path = os.path.join(subdir, fname)
            metrics = parse_json_file(path)
            rows.append({
                "cohort": extract_cohort(fname),
                "data":  detect_model(fname),
                **metrics,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["cohort", "data"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt(avg, std, decimals=3) -> str:
    if avg is None or std is None:
        return "N/A"
    return f"{avg:.{decimals}f} ± {std:.{decimals}f}"


def build_display_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to cohort x data table with formatted strings."""
    records = []
    for _, row in df.iterrows():
        records.append({
            "Cohort":         row["cohort"],
            "Data":           row["data"],
            "C-index":        fmt(row["c_index_avg"], row["c_index_std"]),
            "C-index IPCW":   fmt(row["c_index_ipcw_avg"], row["c_index_ipcw_std"]),
        })
    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate small_cohorts JSON results into a summary table.")
    parser.add_argument("--results_dir",  default="results",        help="Root results directory.")
    parser.add_argument("--output_dir",   default="results/summary", help="Output directory for summary files.")
    parser.add_argument("--decimals",     type=int, default=3,       help="Decimal places in formatted values.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect and display
    df_raw = collect_results(args.results_dir)
    if df_raw.empty:
        print("[WARNING] No result files found. Check --results-dir.")
        return

    print(f"Found {len(df_raw)} result files across {df_raw['cohort'].nunique()} cohorts.\n")

    display_df = build_display_table(df_raw)
    print(display_df.to_string(index=False))

    # Save CSV (raw numeric values)
    csv_path = os.path.join(args.output_dir, "results_summary.csv")
    df_raw.round(args.decimals).to_csv(csv_path, index=False)
    print(f"\nRaw CSV saved:     {csv_path}")

if __name__ == "__main__":
    main()