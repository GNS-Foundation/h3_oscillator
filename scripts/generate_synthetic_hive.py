#!/usr/bin/env python3
"""
Generate synthetic Hive telemetry data for H3-Oscillator deployment prep.

Outputs Parquet files matching the production hive_inference_log and
hive_inference_stats schemas. Calibrated to match the real anchor.

Usage:
    python scripts/generate_synthetic_hive.py [options]

Examples:
    # Default: 12 weeks, ~25 cells, output to data/synthetic_hive/
    python scripts/generate_synthetic_hive.py

    # Short run for quick iteration: 2 weeks, output partitioned by date
    python scripts/generate_synthetic_hive.py --weeks 2 --partition-by-date

    # Reproducible with explicit seed
    python scripts/generate_synthetic_hive.py --seed 123 --weeks 8
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add src/ to path so synthetic module is importable
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from synthetic.hive_simulator import (
    HiveSimulator,
    make_default_cohort_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Hive telemetry data."
    )
    parser.add_argument(
        "--weeks", type=int, default=12,
        help="Simulation duration in weeks (default: 12)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data" / "synthetic_hive",
        help="Output directory (default: data/synthetic_hive)",
    )
    parser.add_argument(
        "--partition-by-date", action="store_true",
        help="Partition inference_log Parquet by date (recommended for large runs)",
    )
    parser.add_argument(
        "--start-date", type=str, default="2026-05-13",
        help="Simulation start date YYYY-MM-DD (default: 2026-05-13)",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Generate data and print summary but don't write Parquet files",
    )
    args = parser.parse_args()

    # Build config
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    config = make_default_cohort_config(
        weeks=args.weeks,
        start_time=start_dt,
        random_seed=args.seed,
    )

    print(f"=== Synthetic Hive Generator ===")
    print(f"  Duration:    {args.weeks} weeks")
    print(f"  Start date:  {args.start_date}")
    print(f"  Seed:        {args.seed}")
    print(f"  Cells:       {len(config.cells)} configured")
    print(f"  Output:      {args.output_dir if not args.summary_only else '(summary-only, no files)'}")
    print()

    # Run
    print("Running simulation...")
    t0 = time.time()
    sim = HiveSimulator(config)
    results = sim.generate()
    sim_elapsed = time.time() - t0

    meta = results["meta"]
    print(f"  → Generated {meta['n_events_total']:,} events across "
          f"{meta['n_cells_active']} active cells in {sim_elapsed:.1f}s")
    print(f"  → inference_log:   {results['inference_log'].shape[0]:,} rows × "
          f"{results['inference_log'].shape[1]} cols")
    print(f"  → inference_stats: {results['inference_stats'].shape[0]:,} rows × "
          f"{results['inference_stats'].shape[1]} cols")
    print()

    # Print per-cell summary
    print("=== Per-cell calibration summary ===")
    print(f"{'Cell':<17} {'Region':<24} {'jobs/hr':>8} {'tok/s':>7} {'lat_ms':>7} {'workers':>8}")
    print("-" * 80)
    for c in meta["cells"]:
        if c["n_events"] == 0:
            continue
        print(f"{c['h3_cell']:<17} {c['region_label']:<24} "
              f"{c['jobs_per_hour_mean']:>8.1f} {c['tokens_per_second_mean']:>7.1f} "
              f"{c['latency_ms_mean']:>7.0f} {c['n_unique_workers']:>8}")
    print()

    if args.summary_only:
        print("Summary-only mode; no files written.")
        return 0

    # Export
    print("Writing Parquet files...")
    t0 = time.time()
    written = sim.export_parquet(
        results,
        args.output_dir,
        partition_by_date=args.partition_by_date,
    )
    export_elapsed = time.time() - t0
    print(f"  → Export complete in {export_elapsed:.1f}s")
    print()
    print("Files written:")
    for k, v in written.items():
        if isinstance(v, list):
            print(f"  {k}: {len(v)} partitions in {Path(v[0]).parent.parent}")
        else:
            print(f"  {k}: {v}")

    print()
    print("Done. Load with e.g.:")
    print(f"  import pandas as pd")
    print(f"  log = pd.read_parquet('{args.output_dir}/inference_log')  # if partitioned")
    print(f"  stats = pd.read_parquet('{args.output_dir}/inference_stats.parquet')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
