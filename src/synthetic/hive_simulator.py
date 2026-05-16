"""
Synthetic Hive Inference Network Simulator
===========================================

Generates synthetic telemetry data matching the schema of real
hive_inference_log and hive_inference_stats tables, calibrated to
match observed statistics from the GEIANT Hive production anchor.

PURPOSE: bridge the gap between real Hive (currently 1 cell, machine-
driven load) and the multi-cell organic traffic needed to validate
H3-Oscillator forecasting. Used to develop and stress-test the
deployment pipeline before real cohort data arrives.

NOT a substitute for real-data validation. Explicitly synthetic.

Calibration target (from real anchor @hive-anchor-eu as of 2026-05-13):
    - ~60 jobs/hour, 24/7 (machine-driven, ~constant)
    - ~109 tok/s mean throughput
    - ~550 ms mean latency

Schema match: hive_inference_log (17 cols) and hive_inference_stats (9 cols)
exactly as returned by the Hive Supabase production database.
"""

from __future__ import annotations
import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass
class CellConfig:
    """Configuration for a single H3 cell in the simulated network."""
    h3_cell: str                       # 15-char hex H3 res-7 cell ID
    region_label: str                  # human-readable e.g. "Italy", "Bay Area"
    timezone_offset_hours: float       # UTC offset for local-time calculations

    # Traffic shape
    base_arrival_rate_per_hour: float  # baseline Poisson rate at trough
    diurnal_amplitude: float           # 0.0 = flat, 1.0 = strong day/night
    peak_hour_local: float             # 0-24, hour of peak traffic (local)
    weekend_factor: float              # multiplier for Sat/Sun (1.0 = same)

    # Worker characteristics
    n_workers: int                     # number of workers in this cell
    worker_uptime: float               # fraction online (0-1)
    mean_latency_ms: float
    latency_cv: float                  # coefficient of variation
    mean_tokens_per_second: float
    tps_cv: float                      # coefficient of variation

    # Activation
    activation_offset_days: int = 0    # days from sim start
    is_anchor: bool = False            # flat 24/7 traffic if True

    # Models served
    available_models: list = field(
        default_factory=lambda: ["tinyllama", "phi-3"]
    )


@dataclass
class SimulatorConfig:
    """Top-level simulator configuration."""
    start_time: datetime
    end_time: datetime
    cells: list                            # list of CellConfig
    random_seed: int = 42
    timestep_seconds: int = 60             # granularity of arrival sampling
    global_activity_amplitude: float = 0.1 # cross-cell correlation
    output_per_event: bool = True          # hive_inference_log rows
    output_aggregates: bool = True         # hive_inference_stats rows
    aggregate_interval_minutes: int = 60   # bucket size for stats
    epoch_duration_seconds: int = 3600     # H3-Oscillator epoch granularity


# =============================================================================
# Helper functions
# =============================================================================

def _rand_hex(rng: np.random.Generator, n_chars: int = 64) -> str:
    """Generate a random hex string of length n_chars (lowercase)."""
    n_bytes = (n_chars + 1) // 2
    return rng.bytes(n_bytes).hex()[:n_chars]


def _stable_pk(seed: str, prefix: str = "WK") -> str:
    """Deterministic worker/requester public key from a seed string."""
    h = hashlib.sha256(seed.encode()).hexdigest()[:32]
    return f"{prefix}_{h}"


def _local_hour(utc_dt: datetime, tz_offset: float) -> float:
    """Convert UTC datetime to local hour-of-day (0-24, fractional)."""
    local = utc_dt + timedelta(hours=tz_offset)
    return local.hour + local.minute / 60.0 + local.second / 3600.0


def _diurnal_multiplier(
    local_hour: float, peak_hour: float, amplitude: float
) -> float:
    """
    Multiplier for Poisson rate based on time-of-day.

    Returns a value in [1-amplitude, 1+amplitude], peaking at `peak_hour`
    and reaching minimum 12 hours opposite. Cosine wave.
    """
    if amplitude <= 0.0:
        return 1.0
    phase = 2 * np.pi * (local_hour - peak_hour) / 24.0
    return 1.0 + amplitude * np.cos(phase)


def _weekend_multiplier(utc_dt: datetime, weekend_factor: float) -> float:
    """Return 1.0 on weekdays, weekend_factor on Sat/Sun."""
    return weekend_factor if utc_dt.weekday() >= 5 else 1.0


# =============================================================================
# The simulator
# =============================================================================

class HiveSimulator:
    """
    Discrete-event-style simulator that emits per-job records and
    pre-aggregated hourly stats matching the production Hive schema.
    """

    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

        # Pre-assign deterministic worker_pks per cell
        self._cell_workers = {
            c.h3_cell: [
                _stable_pk(f"{c.h3_cell}_w{i}", prefix="WK")
                for i in range(c.n_workers)
            ]
            for c in config.cells
        }

        # Pre-build the global activity multiplier time series
        n_steps = int(
            (config.end_time - config.start_time).total_seconds()
            / config.timestep_seconds
        )
        if config.global_activity_amplitude > 0.0:
            # Slow random walk multiplier around 1.0
            walk = self.rng.normal(0.0, 0.05, size=n_steps).cumsum()
            walk -= walk.mean()
            walk *= config.global_activity_amplitude / (walk.std() + 1e-6)
            self._global_mult = 1.0 + walk
        else:
            self._global_mult = np.ones(n_steps)

    # ---- main entry point ---------------------------------------------------

    def generate(self) -> dict:
        """
        Run the simulation. Returns a dict with keys:
            'inference_log': DataFrame matching hive_inference_log schema
            'inference_stats': DataFrame matching hive_inference_stats schema
            'meta': summary stats per cell
        """
        all_events = []
        cfg = self.config
        n_steps = int(
            (cfg.end_time - cfg.start_time).total_seconds()
            / cfg.timestep_seconds
        )

        for cell in cfg.cells:
            cell_activation = cfg.start_time + timedelta(
                days=cell.activation_offset_days
            )
            cell_events = self._simulate_cell(cell, cell_activation, n_steps)
            all_events.extend(cell_events)

        if not all_events:
            log_df = self._empty_inference_log()
            stats_df = self._empty_inference_stats()
            return {
                "inference_log": log_df,
                "inference_stats": stats_df,
                "meta": {"n_events": 0, "cells": []},
            }

        log_df = pd.DataFrame(all_events)
        log_df = log_df.sort_values("created_at").reset_index(drop=True)

        # Force schema column order
        log_cols = [
            "id", "h3_cell", "epoch", "worker_pk", "job_id", "requester_pk",
            "model", "provider", "tokens_in", "tokens_out", "latency_ms",
            "prompt_hash", "response_hash", "job_hash", "stellar_tx",
            "cost_gns", "created_at",
        ]
        log_df = log_df[log_cols]

        # Aggregated hourly stats
        if cfg.output_aggregates:
            stats_df = self._aggregate(log_df)
        else:
            stats_df = self._empty_inference_stats()

        # Summary metadata
        meta = self._summarize(log_df, cfg)

        return {
            "inference_log": log_df,
            "inference_stats": stats_df,
            "meta": meta,
        }

    # ---- per-cell simulation ------------------------------------------------

    def _simulate_cell(
        self, cell: CellConfig, activation_time: datetime, n_steps: int
    ) -> list:
        """Generate all events for one cell."""
        cfg = self.config
        events = []
        step_seconds = cfg.timestep_seconds

        for step_idx in range(n_steps):
            t_now = cfg.start_time + timedelta(seconds=step_idx * step_seconds)

            if t_now < activation_time:
                continue

            # Compute current arrival rate
            base_per_sec = cell.base_arrival_rate_per_hour / 3600.0

            if cell.is_anchor:
                # Flat traffic, no diurnal
                rate_mult = 1.0
            else:
                local_h = _local_hour(t_now, cell.timezone_offset_hours)
                rate_mult = _diurnal_multiplier(
                    local_h, cell.peak_hour_local, cell.diurnal_amplitude
                )
                rate_mult *= _weekend_multiplier(t_now, cell.weekend_factor)

            rate_mult *= self._global_mult[step_idx]

            # Effective worker capacity (some workers offline)
            n_online = self.rng.binomial(cell.n_workers, cell.worker_uptime)
            if n_online == 0:
                continue
            worker_capacity_factor = n_online / max(cell.n_workers, 1)

            expected_jobs = (
                base_per_sec * step_seconds * rate_mult * worker_capacity_factor
            )
            n_jobs = self.rng.poisson(max(expected_jobs, 0.0))

            if n_jobs == 0:
                continue

            # Generate each job in this timestep
            for _ in range(n_jobs):
                # Random time within this timestep
                offset_sec = self.rng.uniform(0, step_seconds)
                created_at = t_now + timedelta(seconds=offset_sec)

                # Random worker from online set
                worker_idx = int(self.rng.integers(0, n_online))
                worker_pk = self._cell_workers[cell.h3_cell][worker_idx]

                # Model selection
                model = cell.available_models[
                    int(self.rng.integers(0, len(cell.available_models)))
                ]

                # Provider: 95% hive, 5% groq fallback (matches real)
                provider = "hive" if self.rng.random() < 0.95 else "groq"

                # Physics: latency = generation_time + overhead
                # where generation_time = tokens_out / effective_tps
                # So we sample (tokens_in, tokens_out, effective_tps) and
                # COMPUTE latency from them. This keeps the three quantities
                # consistent (calibration to anchor: lat 550ms, tps 109,
                # implies tokens_out avg ~55).

                # Token counts (log-normal)
                tokens_in = max(1, int(self.rng.lognormal(4.5, 0.7)))
                tokens_out = max(1, int(self.rng.lognormal(3.7, 0.6)))

                # Sample worker's effective TPS this job (gamma around cell mean)
                tps_mean = cell.mean_tokens_per_second
                tps_std = max(tps_mean * cell.tps_cv, 1e-3)
                tps_var = tps_std ** 2
                tps_shape = (tps_mean ** 2) / tps_var
                tps_scale = tps_var / tps_mean
                effective_tps = max(10.0, self.rng.gamma(tps_shape, tps_scale))

                # Queueing / startup overhead, gamma around small positive value
                # Cell's latency_cv controls overhead variance
                overhead_mean = 50.0
                overhead_std = max(overhead_mean * cell.latency_cv, 1.0)
                overhead_var = overhead_std ** 2
                ovh_shape = (overhead_mean ** 2) / overhead_var
                ovh_scale = overhead_var / overhead_mean
                overhead_ms = float(self.rng.gamma(ovh_shape, ovh_scale))

                # Latency = generation time + overhead
                latency_ms = (tokens_out / effective_tps) * 1000.0 + overhead_ms
                latency_ms = max(50.0, min(latency_ms, 30000.0))

                # Cost: simple model — proportional to tokens, with small markup
                cost_gns = round(
                    (tokens_in + tokens_out) * 0.0001
                    + self.rng.uniform(0, 0.001),
                    6,
                )

                # Hashes & IDs (random hex)
                job_id = _rand_hex(self.rng, 32)
                requester_pk = _stable_pk(
                    f"req_{cell.h3_cell}_{int(self.rng.integers(0, 50))}",
                    prefix="RQ",
                )
                prompt_hash = _rand_hex(self.rng, 64)
                response_hash = _rand_hex(self.rng, 64)
                job_hash = _rand_hex(self.rng, 64)
                stellar_tx = _rand_hex(self.rng, 64)

                # epoch = floor(unix_ts / epoch_duration)
                epoch = int(
                    created_at.timestamp() // cfg.epoch_duration_seconds
                )

                # Row ID — UUID-ish hex
                row_id = _rand_hex(self.rng, 32)

                events.append({
                    "id": row_id,
                    "h3_cell": cell.h3_cell,
                    "epoch": epoch,
                    "worker_pk": worker_pk,
                    "job_id": job_id,
                    "requester_pk": requester_pk,
                    "model": model,
                    "provider": provider,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "latency_ms": round(latency_ms, 2),
                    "prompt_hash": prompt_hash,
                    "response_hash": response_hash,
                    "job_hash": job_hash,
                    "stellar_tx": stellar_tx,
                    "cost_gns": cost_gns,
                    "created_at": created_at.replace(tzinfo=timezone.utc),
                })

        return events

    # ---- aggregation --------------------------------------------------------

    def _aggregate(self, log_df: pd.DataFrame) -> pd.DataFrame:
        """Roll up per-event log into hive_inference_stats (hourly buckets)."""
        if log_df.empty:
            return self._empty_inference_stats()

        df = log_df.copy()
        # Bucket to hour
        df["hour"] = df["created_at"].dt.floor("h")

        grouped = df.groupby(
            ["h3_cell", "hour", "model", "provider"], as_index=False
        ).agg(
            total_jobs=("id", "count"),
            total_tokens_in=("tokens_in", "sum"),
            total_tokens_out=("tokens_out", "sum"),
            avg_latency_ms=("latency_ms", "mean"),
            total_cost_gns=("cost_gns", "sum"),
        )

        # Round floats
        grouped["avg_latency_ms"] = grouped["avg_latency_ms"].round(2)
        grouped["total_cost_gns"] = grouped["total_cost_gns"].round(6)

        cols = [
            "h3_cell", "hour", "model", "provider",
            "total_jobs", "total_tokens_in", "total_tokens_out",
            "avg_latency_ms", "total_cost_gns",
        ]
        return grouped[cols].sort_values(["hour", "h3_cell"]).reset_index(drop=True)

    # ---- empty frames -------------------------------------------------------

    @staticmethod
    def _empty_inference_log() -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "id", "h3_cell", "epoch", "worker_pk", "job_id", "requester_pk",
            "model", "provider", "tokens_in", "tokens_out", "latency_ms",
            "prompt_hash", "response_hash", "job_hash", "stellar_tx",
            "cost_gns", "created_at",
        ])

    @staticmethod
    def _empty_inference_stats() -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "h3_cell", "hour", "model", "provider",
            "total_jobs", "total_tokens_in", "total_tokens_out",
            "avg_latency_ms", "total_cost_gns",
        ])

    # ---- summary stats ------------------------------------------------------

    def _summarize(
        self, log_df: pd.DataFrame, cfg: SimulatorConfig
    ) -> dict:
        """Return per-cell summary stats for sanity checking calibration.

        Preserves the input order of cells (anchor first, then cohort, then
        viral) and includes region labels.
        """
        summary = {
            "n_events_total": len(log_df),
            "n_cells_active": log_df["h3_cell"].nunique() if len(log_df) else 0,
            "sim_duration_days": (
                cfg.end_time - cfg.start_time
            ).total_seconds() / 86400.0,
            "cells": [],
        }
        if log_df.empty:
            return summary

        # Iterate through config.cells in original order, not alphabetic
        for cell_cfg in cfg.cells:
            group = log_df[log_df["h3_cell"] == cell_cfg.h3_cell]
            if group.empty:
                summary["cells"].append({
                    "h3_cell": cell_cfg.h3_cell,
                    "region_label": cell_cfg.region_label,
                    "n_events": 0,
                    "jobs_per_hour_mean": 0.0,
                    "tokens_per_second_mean": 0.0,
                    "latency_ms_mean": 0.0,
                    "latency_ms_p50": 0.0,
                    "latency_ms_p95": 0.0,
                    "cost_gns_total": 0.0,
                    "n_unique_workers": 0,
                })
                continue

            duration_hours = max(
                (group["created_at"].max() - group["created_at"].min())
                .total_seconds() / 3600.0,
                1e-6,
            )
            summary["cells"].append({
                "h3_cell": cell_cfg.h3_cell,
                "region_label": cell_cfg.region_label,
                "n_events": int(len(group)),
                "jobs_per_hour_mean": float(len(group) / duration_hours),
                "tokens_per_second_mean": float(
                    (group["tokens_out"] / (group["latency_ms"] / 1000.0))
                    .mean()
                ),
                "latency_ms_mean": float(group["latency_ms"].mean()),
                "latency_ms_p50": float(group["latency_ms"].median()),
                "latency_ms_p95": float(group["latency_ms"].quantile(0.95)),
                "cost_gns_total": float(group["cost_gns"].sum()),
                "n_unique_workers": int(group["worker_pk"].nunique()),
            })
        return summary

    # ---- export -------------------------------------------------------------

    def export_parquet(
        self,
        results: dict,
        output_dir: Path,
        partition_by_date: bool = True,
    ) -> dict:
        """
        Write generated data to Parquet files. Returns dict of written paths.

        Layout:
            output_dir/inference_log/dt=YYYY-MM-DD/part.parquet  (if partitioned)
            output_dir/inference_log.parquet                      (if not)
            output_dir/inference_stats.parquet
            output_dir/meta.json
        """
        import json
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written = {}

        log_df = results["inference_log"]
        stats_df = results["inference_stats"]

        # inference_log
        if partition_by_date and not log_df.empty:
            log_root = output_dir / "inference_log"
            log_root.mkdir(exist_ok=True)
            log_df = log_df.copy()
            log_df["__date"] = log_df["created_at"].dt.strftime("%Y-%m-%d")
            for date_str, subset in log_df.groupby("__date"):
                day_dir = log_root / f"dt={date_str}"
                day_dir.mkdir(exist_ok=True)
                path = day_dir / "part.parquet"
                subset.drop(columns=["__date"]).to_parquet(path, index=False)
                written.setdefault("inference_log_partitions", []).append(str(path))
        else:
            path = output_dir / "inference_log.parquet"
            log_df.to_parquet(path, index=False)
            written["inference_log"] = str(path)

        # inference_stats (always single file)
        stats_path = output_dir / "inference_stats.parquet"
        stats_df.to_parquet(stats_path, index=False)
        written["inference_stats"] = str(stats_path)

        # meta
        meta_path = output_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(results["meta"], f, indent=2, default=str)
        written["meta"] = str(meta_path)

        return written


# =============================================================================
# Default cohort builder
# =============================================================================

def make_default_cohort_config(
    weeks: int = 12,
    start_time: Optional[datetime] = None,
    random_seed: int = 42,
) -> SimulatorConfig:
    """
    Build a realistic cohort growth scenario matching expected Hive expansion:

    - Italy anchor (matches real @hive-anchor-eu) from day 0
    - Bay Area cluster (3 cells) starting week 1 (Mauricio + neighbors)
    - US East/Midwest (5 cells) starting weeks 2-3 (Heather, Pete + viral)
    - Random viral expansion (~2 cells/week) after week 4
    Target: ~20-25 active cells by end of `weeks`.

    Calibration: Italy anchor matches real stats (~60 jobs/hr, ~109 tok/s,
    ~550 ms latency, flat 24/7).
    """
    if start_time is None:
        start_time = datetime(2026, 5, 13, 0, 0, 0, tzinfo=timezone.utc)
    end_time = start_time + timedelta(weeks=weeks)

    # Placeholder H3 res-7 cell IDs (15-char hex starting with '87').
    # Valid H3 IDs are lowercase hex only (0-9, a-f). The region_code is
    # mixed into the hash but the output is pure hex.
    def _make_cell(region_code: str, idx: int) -> str:
        h = hashlib.md5(f"{region_code}_{idx}".encode()).hexdigest()[:13]
        return f"87{h}"

    cells = []

    # ---- Italy anchor (matches real) ----
    # Real anchor in production is at H3 res-6 (`861e8050fffffff`); at res-7
    # this would be one of its 7 children. Use a plausible 15-char res-7 ID.
    cells.append(CellConfig(
        h3_cell="871e8050affffff",  # plausible Italy res-7 cell, 15-char
        region_label="Italy (anchor)",
        timezone_offset_hours=1.0,  # CET
        base_arrival_rate_per_hour=60.0,  # matches real ~60 jobs/hr
        diurnal_amplitude=0.0,  # flat
        peak_hour_local=12.0,
        weekend_factor=1.0,
        n_workers=1,
        worker_uptime=0.99,
        mean_latency_ms=550.0,
        latency_cv=0.15,
        mean_tokens_per_second=109.0,
        tps_cv=0.12,
        activation_offset_days=0,
        is_anchor=True,
        available_models=["tinyllama", "phi-3"],
    ))

    # ---- Bay Area cluster ----
    for i in range(3):
        cells.append(CellConfig(
            h3_cell=_make_cell("BA", i),
            region_label=f"Bay Area #{i+1}",
            timezone_offset_hours=-8.0,  # PST
            base_arrival_rate_per_hour=8.0 + i * 4,  # 8, 12, 16 jobs/hr at trough
            diurnal_amplitude=0.75,
            peak_hour_local=14.0 + i * 0.5,  # slight stagger
            weekend_factor=0.5,
            n_workers=2 + i,  # 2, 3, 4 workers
            worker_uptime=0.85,
            mean_latency_ms=450.0 + i * 30,
            latency_cv=0.25,
            mean_tokens_per_second=120.0 - i * 10,
            tps_cv=0.20,
            activation_offset_days=7,  # week 1
            is_anchor=False,
        ))

    # ---- US East / Midwest ----
    east_midwest_specs = [
        ("USE", -5.0, 14),  # NYC area
        ("USE", -5.0, 15),  # Boston
        ("USE", -5.0, 16),  # DC
        ("USM", -6.0, 18),  # Chicago
        ("USM", -6.0, 19),  # Detroit
    ]
    for i, (region_code, tz, week_start_day) in enumerate(east_midwest_specs):
        cells.append(CellConfig(
            h3_cell=_make_cell(region_code, i),
            region_label=f"{region_code} #{i+1}",
            timezone_offset_hours=tz,
            base_arrival_rate_per_hour=6.0 + i * 3,
            diurnal_amplitude=0.7,
            peak_hour_local=15.0,
            weekend_factor=0.45,
            n_workers=2 + (i % 3),
            worker_uptime=0.82,
            mean_latency_ms=500.0 + i * 20,
            latency_cv=0.22,
            mean_tokens_per_second=100.0 + i * 5,
            tps_cv=0.18,
            activation_offset_days=week_start_day,
            is_anchor=False,
        ))

    # ---- Viral expansion: ~2 cells/week from week 4 onwards ----
    viral_rng = np.random.default_rng(random_seed + 1)
    n_viral_weeks = max(0, weeks - 4)
    n_viral_cells = int(n_viral_weeks * 2)

    viral_regions = [
        ("EUR", 1.0, 15.0),    # Western Europe
        ("EUR", 2.0, 14.0),    # Eastern Europe
        ("UK",  0.0, 14.0),    # UK
        ("ASA", 9.0, 11.0),    # Japan/Korea
        ("AUS", 11.0, 13.0),   # Australia
        ("BR",  -3.0, 15.0),   # Brazil
        ("CAN", -5.0, 14.0),   # Canada
    ]
    for vi in range(n_viral_cells):
        spec = viral_regions[vi % len(viral_regions)]
        region_code, tz, peak = spec
        cells.append(CellConfig(
            h3_cell=_make_cell(region_code, 100 + vi),
            region_label=f"Viral {region_code} #{vi+1}",
            timezone_offset_hours=tz,
            base_arrival_rate_per_hour=float(viral_rng.uniform(3.0, 12.0)),
            diurnal_amplitude=float(viral_rng.uniform(0.5, 0.9)),
            peak_hour_local=peak + float(viral_rng.uniform(-1.5, 1.5)),
            weekend_factor=float(viral_rng.uniform(0.35, 0.7)),
            n_workers=int(viral_rng.integers(1, 5)),
            worker_uptime=float(viral_rng.uniform(0.75, 0.95)),
            mean_latency_ms=float(viral_rng.uniform(400, 700)),
            latency_cv=0.25,
            mean_tokens_per_second=float(viral_rng.uniform(70, 140)),
            tps_cv=0.20,
            activation_offset_days=28 + (vi // 2) * 7,  # 2 per week starting w4
            is_anchor=False,
        ))

    return SimulatorConfig(
        start_time=start_time,
        end_time=end_time,
        cells=cells,
        random_seed=random_seed,
        timestep_seconds=60,
        global_activity_amplitude=0.15,
        output_per_event=True,
        output_aggregates=True,
        aggregate_interval_minutes=60,
        epoch_duration_seconds=3600,
    )


# =============================================================================
# Module-level convenience function
# =============================================================================

def run_default_simulation(
    weeks: int = 12,
    output_dir: Optional[Path] = None,
    random_seed: int = 42,
) -> dict:
    """
    One-shot convenience: build default cohort config, generate, optionally
    export to Parquet. Returns the results dict (with DataFrames in memory).
    """
    config = make_default_cohort_config(weeks=weeks, random_seed=random_seed)
    sim = HiveSimulator(config)
    results = sim.generate()
    if output_dir is not None:
        paths = sim.export_parquet(results, Path(output_dir))
        results["written_paths"] = paths
    return results


if __name__ == "__main__":
    # Quick smoke test
    print("Running quick smoke test (2 weeks, default cohort)...")
    results = run_default_simulation(weeks=2, random_seed=42)
    meta = results["meta"]
    print(f"Generated {meta['n_events_total']} events across "
          f"{meta['n_cells_active']} cells over "
          f"{meta['sim_duration_days']:.1f} days.")
    print("\nPer-cell summary (first 5):")
    for cell_summary in meta["cells"][:5]:
        print(f"  {cell_summary['h3_cell'][:16]}... "
              f"jobs/hr={cell_summary['jobs_per_hour_mean']:.1f}, "
              f"lat_ms={cell_summary['latency_ms_mean']:.0f}, "
              f"workers={cell_summary['n_unique_workers']}")
