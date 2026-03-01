"""Demo script: run Phase 1 acquisition orchestration without hardware.

This demo uses a fake adapter so users can validate orchestration flow on any PC.
Run from repository root:
    python demo_tests/phase1_acquisition_sweep_demo.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eis import execute_sweep, load_and_validate_config
from eis.models.measurement_models import ExcitationConfig, HardwareConfig, PreflightCheckResult


class FakeAdapter:
    """Simple fake adapter implementing the methods used by execute_sweep."""

    def run_preflight_check(self, **kwargs) -> PreflightCheckResult:
        samples = int(kwargs["samples_per_channel"])
        return PreflightCheckResult(
            sample_rate_sps=float(kwargs["sample_rate_sps"]),
            samples_per_channel=samples,
            measured_shape=(2, samples),
            message="Fake preflight passed.",
        )

    def measure_sine_point(self, **kwargs) -> np.ndarray:
        periods = int(kwargs["n_periods"])
        sample_count = periods * 12
        t = np.arange(sample_count, dtype=np.float64)
        ch1 = 0.1 * np.sin(2.0 * np.pi * t / sample_count)
        ch2 = 0.2 * np.sin(2.0 * np.pi * t / sample_count + 0.1)
        return np.vstack([ch1, ch2])


def _progress_bar_text(completed: int, total: int, width: int = 28) -> str:
    """Build fixed-width text progress bar."""

    filled = int(round((completed / total) * width)) if total else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def main() -> None:
    """Run a short sweep with fake backend and print progress updates."""

    sweep = load_and_validate_config(REPO_ROOT / "config_examples" / "config_phase1_example.xlsx")
    short_sweep = type(sweep)(
        source_path=sweep.source_path,
        sheet_name=sweep.sheet_name,
        points=sweep.points[:4],
    )

    adapter = FakeAdapter()
    hardware = HardwareConfig(ai_channels=("ai0", "ai7"))
    excitation = ExcitationConfig(amplitude_v=0.2, offset_v=0.0)

    def on_progress(p) -> None:
        bar = _progress_bar_text(p.completed_steps, p.total_steps)
        print(
            f"{bar} {p.completed_steps}/{p.total_steps} "
            f"f={p.frequency_hz:.2f} Hz repeat={p.repeat_index}"
        )

    result = execute_sweep(
        sweep=short_sweep,
        adapter=adapter,  # type: ignore[arg-type]
        hardware=hardware,
        excitation=excitation,
        repeats=2,
        run_preflight=True,
        progress_callback=on_progress,
    )

    print("\nSummary")
    print(f"Start: {result.started_at_utc_iso}")
    print(f"Finish: {result.finished_at_utc_iso}")
    print(f"Captures: {len(result.captures)}")
    print(f"Preflight: {result.preflight.message if result.preflight else 'skipped'}")
    print(f"First capture shape: {result.captures[0].raw_data.shape}")


if __name__ == "__main__":
    main()
