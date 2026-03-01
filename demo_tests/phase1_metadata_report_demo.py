"""Demo script: generate metadata bank + HTML/PDF reports from fake sweep run.

Outputs are written to `measurements/` (ignored by git).
Run:
    python demo_tests/phase1_metadata_report_demo.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eis import (
    ExcitationConfig,
    HardwareConfig,
    build_metadata_bank,
    create_run_folder_layout,
    execute_sweep,
    load_and_validate_config,
    write_description_file,
    write_metadata_bank_csv,
    write_metadata_bank_txt,
    write_metadata_report_html,
    write_metadata_report_pdf,
)
from eis.models.measurement_models import PreflightCheckResult

import numpy as np


class FakeAdapter:
    """Fake adapter to run demo without NI hardware."""

    def run_preflight_check(self, **kwargs) -> PreflightCheckResult:
        samples = int(kwargs["samples_per_channel"])
        return PreflightCheckResult(
            sample_rate_sps=float(kwargs["sample_rate_sps"]),
            samples_per_channel=samples,
            measured_shape=(2, samples),
            message="Fake preflight passed",
        )

    def measure_sine_point(self, **kwargs):
        n = int(kwargs["n_periods"]) * 10
        t = np.arange(n, dtype=np.float64)
        ch1 = 0.05 * np.sin(2 * np.pi * t / n)
        ch2 = 0.07 * np.sin(2 * np.pi * t / n + 0.2)
        return np.vstack([ch1, ch2])


def main() -> None:
    """Run fake sweep and write metadata bank/reports."""

    sweep = load_and_validate_config(REPO_ROOT / "config_examples" / "config_phase1_example.xlsx")
    sweep = type(sweep)(
        source_path=sweep.source_path,
        sheet_name=sweep.sheet_name,
        points=sweep.points[:3],
    )

    adapter = FakeAdapter()
    hardware = HardwareConfig(ai_channels=("ai0", "ai7"))
    excitation = ExcitationConfig(drive_mode="auto_from_current_rms", offset_v=0.0)

    run_result = execute_sweep(
        sweep=sweep,
        adapter=adapter,  # type: ignore[arg-type]
        hardware=hardware,
        excitation=excitation,
        repeats=2,
        run_preflight=True,
    )

    layout = create_run_folder_layout(
        base_output_dir=REPO_ROOT / "measurements",
        serial_number="DEMO_META_001",
        started_at_local=datetime.now(),
    )

    metadata_bank = build_metadata_bank(
        sweep=sweep,
        run_result=run_result,
        hardware=hardware,
        excitation=excitation,
        serial_number="DEMO_META_001",
        user_name="demo_user",
        description="Chunk 4 metadata report demo",
    )

    txt_path = write_metadata_bank_txt(metadata_bank, layout.root / "metadata_bank.txt")
    csv_path = write_metadata_bank_csv(metadata_bank, layout.root / "metadata_measurements.csv")
    html_path = write_metadata_report_html(metadata_bank, layout.root / "metadata_report.html")
    pdf_path = write_metadata_report_pdf(metadata_bank, layout.root / "metadata_report.pdf")
    write_description_file(metadata_bank["identity"]["description"], layout.root / "description.txt")

    print("Metadata demo completed")
    print(f"Run folder: {layout.root}")
    print(f"Metadata bank txt: {txt_path}")
    print(f"Metadata measurements csv: {csv_path}")
    print(f"Metadata report html: {html_path}")
    print(f"Metadata report pdf: {pdf_path}")


if __name__ == "__main__":
    main()
