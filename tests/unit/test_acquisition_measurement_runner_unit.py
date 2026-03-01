"""Unit tests for single-point acquisition runner."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from eis.acquisition.measurement_runner import run_measurement_point
from eis.models.config_models import MeasurementPointConfig
from eis.models.measurement_models import CaptureConditioningConfig, ExcitationConfig, HardwareConfig


class _FakeAdapter:
    """Simple fake adapter capturing method call arguments."""

    def __init__(self, response: np.ndarray) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def measure_sine_point(self, **kwargs) -> np.ndarray:
        self.calls.append(kwargs)
        return self.response


class _DynamicFakeAdapter:
    """Fake adapter producing sample count proportional to requested periods."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def measure_sine_point(self, **kwargs) -> np.ndarray:
        self.calls.append(kwargs)
        sample_count = int(kwargs["n_periods"]) * 10
        t = np.arange(sample_count, dtype=np.float64)
        ch1 = np.sin(2.0 * np.pi * t / sample_count)
        ch2 = np.sin(2.0 * np.pi * t / sample_count + 0.2)
        return np.vstack([ch1, ch2])


class TestAcquisitionMeasurementRunnerUnit(unittest.TestCase):
    """Checks measurement runner output model and guard checks."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._repo_root = Path(__file__).resolve().parents[2]

    # Checks runner maps one config row to one capture and uses larger AI range.
    def test_run_measurement_point_uses_max_channel_range(self) -> None:
        adapter = _FakeAdapter(np.zeros((2, 10), dtype=np.float64))
        point = MeasurementPointConfig(
            row_number=7,
            frequency_hz=53.4,
            ch0_range_v=1.0,
            ch1_range_v=2.5,
            sample_rate_sps=250000.0,
            n_periods=20,
            current_rms=10.0,
        )
        hardware = HardwareConfig(ai_channels=("ai0", "ai7"))
        excitation = ExcitationConfig(
            drive_mode="auto_from_current_rms",
            offset_v=0.0,
        )

        capture = run_measurement_point(
            adapter=adapter,
            point=point,
            hardware=hardware,
            excitation=excitation,
            repeat_index=2,
        )

        self.assertEqual(capture.row_number, 7)
        self.assertEqual(capture.repeat_index, 2)
        self.assertEqual(capture.ai_channels, ("ai0", "ai7"))
        self.assertAlmostEqual(capture.ai_range_v, 2.5)
        self.assertEqual(capture.current_range_name, "20A")
        self.assertAlmostEqual(float(capture.transconductance_siemens), 10.0)
        self.assertTrue(capture.ao_amplitude_v_peak > 0.0)
        self.assertEqual(capture.raw_data.shape[0], 2)
        self.assertTrue(1 <= capture.raw_data.shape[1] <= 10)
        self.assertTrue(capture.duration_s >= 0.0)
        self.assertIn("T", capture.started_at_utc_iso)
        self.assertTrue(capture.acquired_periods >= point.n_periods)
        self.assertEqual(capture.periodic_window_samples, capture.raw_data.shape[1])

        self.assertEqual(len(adapter.calls), 1)
        self.assertAlmostEqual(float(adapter.calls[0]["ai_range_v"]), 2.5)
        self.assertEqual(int(adapter.calls[0]["n_periods"]), capture.acquired_periods)

    # Checks runner rejects data with unexpected channel dimension.
    def test_run_measurement_point_rejects_wrong_channel_count(self) -> None:
        adapter = _FakeAdapter(np.zeros((1, 8), dtype=np.float64))
        point = MeasurementPointConfig(
            row_number=2,
            frequency_hz=100.0,
            ch0_range_v=2.5,
            ch1_range_v=2.5,
            sample_rate_sps=200000.0,
            n_periods=10,
            current_rms=5.0,
        )
        hardware = HardwareConfig(ai_channels=("ai0", "ai7"))
        excitation = ExcitationConfig(amplitude_v=0.1, offset_v=0.0)

        with self.assertRaises(RuntimeError):
            run_measurement_point(
                adapter=adapter,
                point=point,
                hardware=hardware,
                excitation=excitation,
                repeat_index=1,
            )

    # Checks fixed amplitude mode is respected and no range metadata is added.
    def test_run_measurement_point_fixed_amplitude_mode(self) -> None:
        adapter = _FakeAdapter(np.zeros((2, 8), dtype=np.float64))
        point = MeasurementPointConfig(
            row_number=2,
            frequency_hz=100.0,
            ch0_range_v=2.5,
            ch1_range_v=2.5,
            sample_rate_sps=200000.0,
            n_periods=10,
            current_rms=5.0,
        )
        hardware = HardwareConfig(ai_channels=("ai0", "ai7"))
        excitation = ExcitationConfig(
            drive_mode="fixed_ao_amplitude",
            amplitude_v=0.25,
            offset_v=0.0,
        )

        capture = run_measurement_point(
            adapter=adapter,
            point=point,
            hardware=hardware,
            excitation=excitation,
            repeat_index=1,
        )
        self.assertAlmostEqual(capture.ao_amplitude_v_peak, 0.25)
        self.assertIsNone(capture.current_range_name)
        self.assertIsNone(capture.transconductance_siemens)

    # Checks settle+trim conditioning acquires extra data and returns requested window length.
    def test_run_measurement_point_applies_settle_and_trim_conditioning(self) -> None:
        adapter = _DynamicFakeAdapter()
        point = MeasurementPointConfig(
            row_number=2,
            frequency_hz=10.0,
            ch0_range_v=2.5,
            ch1_range_v=2.5,
            sample_rate_sps=1000.0,
            n_periods=20,
            current_rms=5.0,
        )
        hardware = HardwareConfig(ai_channels=("ai0", "ai7"))
        excitation = ExcitationConfig(amplitude_v=0.1, offset_v=0.0)
        conditioning = CaptureConditioningConfig(
            settle_discard_s=0.05,  # 0.5 period at 10 Hz -> ceil to 1 period
            extra_periods_for_trim=1,
            alignment_search_periods=1,
        )

        capture = run_measurement_point(
            adapter=adapter,  # type: ignore[arg-type]
            point=point,
            hardware=hardware,
            excitation=excitation,
            conditioning=conditioning,
            repeat_index=1,
        )

        self.assertEqual(capture.acquired_periods, 22)
        self.assertEqual(int(adapter.calls[0]["n_periods"]), 22)
        self.assertEqual(capture.raw_data.shape[1], 200)
        self.assertEqual(capture.periodic_window_samples, 200)
        self.assertTrue(capture.discarded_settle_samples > 0)


if __name__ == "__main__":
    unittest.main()
