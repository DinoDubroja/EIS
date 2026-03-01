"""Unit tests for metadata bank and report writer module."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eis.models.config_models import MeasurementPointConfig, SweepConfig
from eis.models.measurement_models import (
    ExcitationConfig,
    HardwareConfig,
    MeasurementCapture,
    PreflightCheckResult,
    SweepRunResult,
)
from eis.storage.metadata_writer import (
    build_metadata_bank,
    regenerate_reports_from_bank,
    write_description_file,
    write_metadata_bank_csv,
    write_metadata_bank_txt,
    write_metadata_report_html,
    write_metadata_report_pdf,
)


class TestStorageMetadataWriterUnit(unittest.TestCase):
    """Checks metadata persistence and report regeneration from data bank."""

    def _build_sample_inputs(self, root: Path):
        sweep = SweepConfig(
            source_path=root / "config_examples" / "config_phase1_example.xlsx",
            sheet_name="Sheet1",
            points=(
                MeasurementPointConfig(
                    row_number=2,
                    frequency_hz=12.54,
                    ch0_range_v=2.5,
                    ch1_range_v=2.5,
                    sample_rate_sps=250000.0,
                    n_periods=20,
                    current_rms=10.0,
                ),
            ),
        )
        capture = MeasurementCapture(
            row_number=2,
            repeat_index=1,
            frequency_hz=12.54,
            sample_rate_sps=250000.0,
            n_periods=20,
            current_rms=10.0,
            ao_amplitude_v_peak=1.4142,
            ao_offset_v=0.0,
            current_range_name="20A",
            transconductance_siemens=10.0,
            started_at_utc_iso=datetime.now(timezone.utc).isoformat(),
            duration_s=0.123,
            ai_channels=("ai0", "ai7"),
            ai_range_v=2.5,
            raw_data=np.zeros((2, 240), dtype=np.float64),
        )
        run_result = SweepRunResult(
            started_at_utc_iso=datetime.now(timezone.utc).isoformat(),
            finished_at_utc_iso=datetime.now(timezone.utc).isoformat(),
            repeats=1,
            captures=(capture,),
            preflight=PreflightCheckResult(
                sample_rate_sps=250000.0,
                samples_per_channel=256,
                measured_shape=(2, 256),
                message="ok",
            ),
        )
        hardware = HardwareConfig(ai_channels=("ai0", "ai7"))
        excitation = ExcitationConfig(drive_mode="auto_from_current_rms", offset_v=0.0)
        return sweep, run_result, hardware, excitation

    # Checks txt/csv/html/pdf writing and report regeneration from metadata bank.
    def test_metadata_bank_and_report_writers(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        sweep, run_result, hardware, excitation = self._build_sample_inputs(repo_root)

        bank = build_metadata_bank(
            sweep=sweep,
            run_result=run_result,
            hardware=hardware,
            excitation=excitation,
            serial_number="Z100N34",
            user_name="tester",
            description="unit test run",
            capture_artifacts=[
                {
                    "row_number": 2,
                    "repeat_index": 1,
                    "frequency_hz": 12.54,
                    "raw_csv_relpath": "RAW/row_0002_f12_54Hz/repeat_001_raw_ch1_ai0_ch2_ai7.csv",
                    "impedance_csv_relpath": "IMPEDANCE/impedance.csv",
                }
            ],
            point_summaries=[
                {
                    "row_number": 2,
                    "frequency_hz": 12.54,
                    "repeat_count": 1,
                    "summary_csv_relpath": "IMPEDANCE/summary_mean_std.csv",
                }
            ],
        )
        self.assertEqual(bank["schema_version"], "phase1_metadata_v2")
        self.assertEqual(bank["identity"]["serial_number"], "Z100N34")
        self.assertEqual(len(bank["captures"]), 1)
        self.assertEqual(
            bank["captures"][0]["raw_csv_relpath"],
            "RAW/row_0002_f12_54Hz/repeat_001_raw_ch1_ai0_ch2_ai7.csv",
        )
        self.assertEqual(
            bank["captures"][0]["impedance_csv_relpath"],
            "IMPEDANCE/impedance.csv",
        )
        self.assertEqual(bank["artifacts"]["point_summary_count"], 1)

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            txt_path = write_metadata_bank_txt(bank, base / "metadata_bank.txt")
            csv_path = write_metadata_bank_csv(bank, base / "metadata_measurements.csv")
            html_path = write_metadata_report_html(bank, base / "metadata_report.html")
            pdf_path = write_metadata_report_pdf(bank, base / "metadata_report.pdf")
            desc_path = write_description_file("unit description", base / "description.txt")
            desc_none = write_description_file(None, base / "description_missing.txt")

            self.assertTrue(txt_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertTrue(html_path.exists())
            self.assertTrue(pdf_path.exists())
            self.assertTrue(desc_path.exists())
            self.assertIsNone(desc_none)
            self.assertFalse((base / "description_missing.txt").exists())
            self.assertTrue(pdf_path.stat().st_size > 0)

            loaded = json.loads(txt_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["identity"]["user_name"], "tester")

            regen_html, regen_pdf = regenerate_reports_from_bank(
                metadata_bank_txt_path=txt_path,
                html_output_path=base / "regen_report.html",
                pdf_output_path=base / "regen_report.pdf",
            )
            self.assertTrue(regen_html.exists())
            self.assertTrue(regen_pdf.exists())
            self.assertTrue(regen_pdf.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
