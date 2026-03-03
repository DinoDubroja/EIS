"""Unit tests for notebook frontend measurement-run wrappers."""

from __future__ import annotations

from datetime import datetime
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eis.frontend.measurement_runs import (
    RunSaveOptions,
    run_measure_process_save,
    run_preflight_only,
)
from eis.models.config_models import MeasurementPointConfig, SweepConfig
from eis.models.measurement_models import ExcitationConfig, HardwareConfig, PreflightCheckResult
from eis.processing import ImpedanceProcessingConfig


class _FakeAdapter:
    """Deterministic adapter stub for unit tests without NI hardware."""

    def __init__(self) -> None:
        self.close_calls = 0

    def run_preflight_check(self, **kwargs) -> PreflightCheckResult:
        samples_per_channel = int(kwargs["samples_per_channel"])
        return PreflightCheckResult(
            sample_rate_sps=float(kwargs["sample_rate_sps"]),
            samples_per_channel=samples_per_channel,
            measured_shape=(2, samples_per_channel),
            message="fake preflight ok",
        )

    def measure_sine_point(self, **kwargs) -> np.ndarray:
        frequency_hz = float(kwargs["frequency_hz"])
        sample_rate_sps = float(kwargs["sample_rate_sps"])
        n_periods = int(kwargs["n_periods"])
        sample_count = int(round(n_periods * sample_rate_sps / frequency_hz))
        sample_count = max(sample_count, 64)

        t = np.arange(sample_count, dtype=np.float64) / sample_rate_sps
        omega = 2.0 * np.pi * frequency_hz
        ai0 = 0.03 * np.sin(omega * t + 0.12)
        ai7 = 6.20 * np.sin(omega * t + 0.73)
        return np.vstack([ai0, ai7])

    def close(self) -> None:
        self.close_calls += 1


class TestFrontendMeasurementRunsUnit(unittest.TestCase):
    """Checks high-level wrappers used by the Phase 1 measurement notebook."""

    def _build_sweep(self) -> SweepConfig:
        point = MeasurementPointConfig(
            row_number=1,
            frequency_hz=80.0,
            ch0_range_v=10.0,
            ch1_range_v=10.0,
            sample_rate_sps=20_000.0,
            n_periods=10,
            current_rms=10.0,
        )
        return SweepConfig(
            source_path=Path("config_examples/config_phase1_example.xlsx"),
            sheet_name="Sheet1",
            points=(point,),
        )

    # Checks preflight-only wrapper auto-sizes sample count with settle discard.
    def test_run_preflight_only(self) -> None:
        sweep = self._build_sweep()
        fake = _FakeAdapter()
        result = run_preflight_only(
            sweep=sweep,
            hardware=HardwareConfig(),
            excitation=ExcitationConfig(),
            sample_rate_sps=None,
            samples_per_channel=None,
            settle_discard_s=0.15,
            adapter=fake,  # explicit adapter should not be auto-closed
        )
        expected_samples = int(round(0.15 * 20_000.0)) + max(64, int(round(0.02 * 20_000.0)))
        self.assertEqual(result.samples_per_channel, expected_samples)
        self.assertEqual(result.measured_shape, (2, expected_samples))
        self.assertEqual(fake.close_calls, 0)

    # Checks run/process/save wrapper returns expected artifacts and paths.
    def test_run_measure_process_save(self) -> None:
        sweep = self._build_sweep()
        fake = _FakeAdapter()
        save_options = RunSaveOptions(
            write_metadata_bank_txt=True,
            write_metadata_bank_csv=True,
            write_metadata_report_html=True,
            write_metadata_report_pdf=False,
            write_description_file=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            bundle = run_measure_process_save(
                sweep=sweep,
                hardware=HardwareConfig(),
                excitation=ExcitationConfig(
                    drive_mode="auto_from_current_rms",
                    manual_current_range="20A",
                ),
                processing=ImpedanceProcessingConfig(method="fft"),
                base_output_dir=Path(tmp),
                serial_number="UNIT_TEST_SERIAL",
                user_name="tester",
                description="demo description",
                repeats=1,
                run_preflight_during_sweep=False,
                conditioning=None,
                save_options=save_options,
                started_at_local=datetime(2026, 3, 3, 12, 10, 0),
                adapter=fake,
            )

            self.assertEqual(len(bundle.run_result.captures), 1)
            self.assertEqual(len(bundle.impedance_results), 1)
            self.assertEqual(len(bundle.persisted_artifacts.capture_artifacts), 1)
            self.assertTrue((bundle.layout.root / "RAW").exists())
            self.assertTrue((bundle.layout.root / "IMPEDANCE" / "impedance.csv").exists())
            self.assertTrue((bundle.layout.reports / "metadata_report.html").exists())
            self.assertTrue((bundle.layout.root / "description.txt").exists())
            self.assertGreaterEqual(len(bundle.saved_paths), 4)
            self.assertIn((1, 1), bundle.capture_frequency_map)
            self.assertEqual(fake.close_calls, 0)


if __name__ == "__main__":
    unittest.main()
