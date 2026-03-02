"""Unit tests for sweep orchestration with repeats and progress callbacks."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from eis.acquisition.sweep_controller import execute_sweep
from eis.models.config_models import MeasurementPointConfig, SweepConfig
from eis.models.measurement_models import (
    ExcitationConfig,
    HardwareConfig,
    PreflightCheckResult,
    SweepProgress,
)


class _FakeAdapter:
    """Fake adapter implementing run_preflight_check + measure_sine_point."""

    def __init__(self) -> None:
        self.preflight_calls: list[dict[str, object]] = []
        self.measure_calls: list[dict[str, object]] = []

    def run_preflight_check(self, **kwargs) -> PreflightCheckResult:
        self.preflight_calls.append(kwargs)
        return PreflightCheckResult(
            sample_rate_sps=float(kwargs["sample_rate_sps"]),
            samples_per_channel=int(kwargs["samples_per_channel"]),
            measured_shape=(2, int(kwargs["samples_per_channel"])),
            message="fake preflight ok",
        )

    def measure_sine_point(self, **kwargs) -> np.ndarray:
        self.measure_calls.append(kwargs)
        sample_count = int(kwargs["n_periods"]) * 5
        return np.zeros((2, sample_count), dtype=np.float64)


class TestAcquisitionSweepControllerUnit(unittest.TestCase):
    """Checks sweep loop behavior, preflight, and progress updates."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]

    def _build_sweep(self) -> SweepConfig:
        return SweepConfig(
            source_path=self.repo_root / "config_examples" / "config_phase1_example.xlsx",
            sheet_name="Sheet1",
            points=(
                MeasurementPointConfig(
                    row_number=2,
                    frequency_hz=10.0,
                    ch0_range_v=2.5,
                    ch1_range_v=2.5,
                    sample_rate_sps=250000.0,
                    n_periods=20,
                    current_rms=10.0,
                ),
                MeasurementPointConfig(
                    row_number=3,
                    frequency_hz=20.0,
                    ch0_range_v=2.5,
                    ch1_range_v=2.5,
                    sample_rate_sps=250000.0,
                    n_periods=20,
                    current_rms=10.0,
                ),
            ),
        )

    # Checks repeats, progress callback count, and preflight call in a normal run.
    def test_execute_sweep_runs_all_points_and_repeats(self) -> None:
        adapter = _FakeAdapter()
        progress_events: list[SweepProgress] = []

        result = execute_sweep(
            sweep=self._build_sweep(),
            adapter=adapter,  # type: ignore[arg-type]
            hardware=HardwareConfig(ai_channels=("ai0", "ai7")),
            excitation=ExcitationConfig(amplitude_v=0.2),
            repeats=3,
            run_preflight=True,
            progress_callback=progress_events.append,
        )

        self.assertEqual(len(adapter.preflight_calls), 1)
        self.assertEqual(len(adapter.measure_calls), 6)
        self.assertEqual(len(result.captures), 6)
        self.assertIsNotNone(result.preflight)
        self.assertEqual(result.repeats, 3)
        self.assertAlmostEqual(float(adapter.preflight_calls[0]["ao_test_voltage"]), 1.0)
        self.assertAlmostEqual(
            float(adapter.preflight_calls[0]["expected_current_channel_voltage_v"]),
            0.08,
        )
        self.assertAlmostEqual(
            float(adapter.preflight_calls[0]["current_channel_tolerance_v"]),
            0.01,
        )
        self.assertAlmostEqual(float(adapter.preflight_calls[0]["settle_discard_s"]), 0.15)

        self.assertEqual(len(progress_events), 6)
        self.assertEqual(progress_events[-1].completed_steps, 6)
        self.assertEqual(progress_events[-1].total_steps, 6)

    # Checks preflight can be skipped and override sample rate works when enabled.
    def test_execute_sweep_preflight_controls(self) -> None:
        adapter = _FakeAdapter()

        result_no_preflight = execute_sweep(
            sweep=self._build_sweep(),
            adapter=adapter,  # type: ignore[arg-type]
            hardware=HardwareConfig(ai_channels=("ai0", "ai7")),
            excitation=ExcitationConfig(amplitude_v=0.1),
            repeats=1,
            run_preflight=False,
        )
        self.assertEqual(len(adapter.preflight_calls), 0)
        self.assertIsNone(result_no_preflight.preflight)

        execute_sweep(
            sweep=self._build_sweep(),
            adapter=adapter,  # type: ignore[arg-type]
            hardware=HardwareConfig(ai_channels=("ai0", "ai7")),
            excitation=ExcitationConfig(amplitude_v=0.1),
            repeats=1,
            run_preflight=True,
            preflight_sample_rate_sps=12345.0,
        )
        self.assertEqual(len(adapter.preflight_calls), 1)
        self.assertAlmostEqual(float(adapter.preflight_calls[0]["sample_rate_sps"]), 12345.0)

    # Checks preflight current/range overrides propagate to computed AO/shunt targets.
    def test_execute_sweep_preflight_current_and_range_override(self) -> None:
        adapter = _FakeAdapter()

        execute_sweep(
            sweep=self._build_sweep(),
            adapter=adapter,  # type: ignore[arg-type]
            hardware=HardwareConfig(ai_channels=("ai0", "ai7")),
            excitation=ExcitationConfig(drive_mode="auto_from_current_rms"),
            repeats=1,
            run_preflight=True,
            preflight_test_current_rms_a=3.0,
            preflight_manual_current_range="20A",
        )

        self.assertEqual(len(adapter.preflight_calls), 1)
        self.assertAlmostEqual(float(adapter.preflight_calls[0]["ao_test_voltage"]), 0.3)
        self.assertAlmostEqual(
            float(adapter.preflight_calls[0]["expected_current_channel_voltage_v"]),
            0.024,
        )


if __name__ == "__main__":
    unittest.main()
