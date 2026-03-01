"""Demo script: persist RAW/IMPEDANCE artifacts and generate metadata reports.

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
    ImpedancePointResult,
    build_artifact_link_payload,
    build_metadata_bank,
    create_run_folder_layout,
    execute_sweep,
    load_impedance_rows_from_run,
    load_and_validate_config,
    persist_run_artifacts,
    write_description_file,
    write_metadata_bank_csv,
    write_metadata_bank_txt,
    write_metadata_report_html,
    write_metadata_report_pdf,
)
from eis.models.measurement_models import PreflightCheckResult, SweepRunResult

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


def _build_demo_impedance_results(run_result: SweepRunResult) -> tuple[ImpedancePointResult, ...]:
    """Create deterministic placeholder impedance results for storage demo."""

    results: list[ImpedancePointResult] = []
    for capture in run_result.captures:
        z_real = 100.0 + 0.5 * capture.row_number + 0.2 * capture.repeat_index
        z_imag = -0.25 * capture.frequency_hz + 0.1 * capture.repeat_index
        z_mag = float(np.hypot(z_real, z_imag))
        z_phase_deg = float(np.degrees(np.arctan2(z_imag, z_real)))
        results.append(
            ImpedancePointResult(
                row_number=capture.row_number,
                repeat_index=capture.repeat_index,
                frequency_hz=capture.frequency_hz,
                z_real_ohm=z_real,
                z_imag_ohm=z_imag,
                z_magnitude_ohm=z_mag,
                z_phase_deg=z_phase_deg,
                extraction_method="demo_placeholder",
                notes="Demo value generated without full processing pipeline.",
            )
        )
    return tuple(results)


def main() -> None:
    """Run fake sweep and write run artifacts plus metadata/report outputs."""

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

    demo_serial_number = f"DEMO_META_{datetime.now().strftime('%H%M%S')}"

    layout = create_run_folder_layout(
        base_output_dir=REPO_ROOT / "measurements",
        serial_number=demo_serial_number,
        started_at_local=datetime.now(),
    )

    impedance_results = _build_demo_impedance_results(run_result)
    persisted = persist_run_artifacts(
        layout=layout,
        run_result=run_result,
        impedance_results=impedance_results,
    )
    capture_artifacts, point_summaries = build_artifact_link_payload(persisted)

    metadata_bank = build_metadata_bank(
        sweep=sweep,
        run_result=run_result,
        hardware=hardware,
        excitation=excitation,
        serial_number=demo_serial_number,
        user_name="demo_user",
        description="Chunk 5 artifact + metadata report demo",
        capture_artifacts=capture_artifacts,
        point_summaries=point_summaries,
    )

    txt_path = write_metadata_bank_txt(metadata_bank, layout.root / "metadata_bank.txt")
    csv_path = write_metadata_bank_csv(metadata_bank, layout.root / "metadata_measurements.csv")
    html_path = write_metadata_report_html(metadata_bank, layout.reports / "metadata_report.html")

    # HTML is preferred default. Keep PDF generation optional for comparison phase.
    generate_pdf = False
    pdf_path = None
    if generate_pdf:
        pdf_path = write_metadata_report_pdf(metadata_bank, layout.reports / "metadata_report.pdf")

    description_path = write_description_file(
        metadata_bank["identity"]["description"],
        layout.root / "description.txt",
    )
    loaded_rows = load_impedance_rows_from_run(layout.root)

    print("Metadata demo completed")
    print(f"Run folder: {layout.root}")
    print(f"Persisted raw captures: {len(persisted.capture_artifacts)}")
    print(f"Persisted summary rows: {len(persisted.point_summaries)}")
    print(f"Loaded impedance rows from disk: {len(loaded_rows)}")
    print(f"Metadata bank txt: {txt_path}")
    print(f"Metadata measurements csv: {csv_path}")
    print(f"Metadata report html: {html_path}")
    print(f"Metadata report pdf: {pdf_path if pdf_path is not None else 'skipped'}")
    print(f"Description file: {description_path if description_path is not None else 'not generated'}")


if __name__ == "__main__":
    main()
