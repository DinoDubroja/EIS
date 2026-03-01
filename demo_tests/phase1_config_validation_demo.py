"""Demo script: load and validate Phase 1 Excel config example.

Run from repository root:
    python demo_tests/phase1_config_validation_demo.py
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eis import load_and_validate_config


def main() -> None:
    """Load example config and print a compact measurement summary."""

    config_path = REPO_ROOT / "config_examples" / "config_phase1_example.xlsx"
    sweep = load_and_validate_config(config_path)
    n=5 # number of points to print

    print("Phase 1 config validation demo")
    print(f"Source file: {sweep.source_path}")
    print(f"Sheet: {sweep.sheet_name}")
    print(f"Number of sweep points: {len(sweep.points)}")
    print(f"First {n} points:")

    for point in sweep.points[:n]:
        print(
            f"  row {point.row_number}: "
            f"f={point.frequency_hz:.4f} Hz, "
            f"sample_rate={point.sample_rate_sps:.0f} S/s, "
            f"periods={point.n_periods}, "
            f"Irms={point.current_rms:g}"
        )


if __name__ == "__main__":
    main()
