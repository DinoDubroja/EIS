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
    CaptureConditioningConfig,
    ExcitationConfig,
    HardwareConfig,
    ImpedanceProcessingConfig,
    build_artifact_link_payload,
    build_metadata_bank,
    compute_impedance_for_run,
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
        frequency_hz = float(kwargs["frequency_hz"])
        sample_rate_sps = float(kwargs["sample_rate_sps"])
        sample_count = int(round(float(kwargs["n_periods"]) * sample_rate_sps / frequency_hz))
        sample_count = max(64, sample_count)

        t = np.arange(sample_count, dtype=np.float64) / sample_rate_sps
        omega = 2.0 * np.pi * frequency_hz

        # Demo synthesis: nominal current waveform on CH1 via shunt voltage.
        i_peak_a = 1.6
        i_phase_rad = 0.15
        r_shunt_ohm = 0.008
        v_shunt = (r_shunt_ohm * i_peak_a) * np.sin(omega * t + i_phase_rad)

        # DUT channel follows known impedance phasor for deterministic processing demo.
        z_dut = 5.0 + 1.2j
        v_dut_peak = i_peak_a * abs(z_dut)
        v_dut_phase = i_phase_rad + float(np.angle(z_dut))
        v_dut = v_dut_peak * np.sin(omega * t + v_dut_phase)

        # Add small deterministic harmonic + dc terms so filter options are visible.
        v_shunt = v_shunt + 0.0008 * np.sin(3.0 * omega * t) + 0.0003
        v_dut = v_dut + 0.03 * np.sin(3.0 * omega * t + 0.2) - 0.005
        return np.vstack([v_shunt, v_dut])


def main() -> None:
    """Run fake sweep, process impedance, and write artifacts/reports."""

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
        conditioning=CaptureConditioningConfig(
            settle_discard_s=0.02,
            extra_periods_for_trim=1,
            alignment_search_periods=1,
        ),
    )

    demo_serial_number = f"DEMO_META_{datetime.now().strftime('%H%M%S')}"

    layout = create_run_folder_layout(
        base_output_dir=REPO_ROOT / "measurements",
        serial_number=demo_serial_number,
        started_at_local=datetime.now(),
    )

    impedance_results = compute_impedance_for_run(
        run_result=run_result,
        config=ImpedanceProcessingConfig(
            method="fft",
            filter_mode="lowpass",
            lowpass_cutoff_hz=2000.0,
            shunt_resistance_ohm=0.008,
        ),
    )
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
