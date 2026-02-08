"""Unit tests for waveform helper functions in USB6451/waveforms.py."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


def _load_waveforms_module():
    """Load waveforms module from source path."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "USB6451" / "waveforms.py"
    spec = importlib.util.spec_from_file_location("waveforms_module_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestWaveformsUnit(unittest.TestCase):
    """Unit tests for waveform generation helpers."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.waveforms = _load_waveforms_module()

    # Checks sine period shape and value range for valid inputs.
    def test_sine_period_shape_and_range(self) -> None:
        data = self.waveforms.sine_period(amplitude=2.0, offset=1.0, samples_per_period=1000)
        self.assertEqual(data.shape, (1000,))
        self.assertTrue(np.isclose(np.max(data), 3.0, atol=1e-12))
        self.assertTrue(np.isclose(np.min(data), -1.0, atol=1e-12))

    # Checks sine rejects negative amplitude.
    def test_sine_period_rejects_negative_amplitude(self) -> None:
        with self.assertRaises(ValueError):
            self.waveforms.sine_period(amplitude=-0.1, offset=0.0, samples_per_period=100)

    # Checks ramp starts at requested value and remains below stop when endpoint is excluded.
    def test_ramp_period_values_with_excluded_endpoint(self) -> None:
        data = self.waveforms.ramp_period(
            start=-1.0, stop=1.0, samples_per_period=5, include_endpoint=False
        )
        self.assertEqual(data.shape, (5,))
        self.assertTrue(np.isclose(data[0], -1.0, atol=1e-12))
        self.assertTrue(data[-1] < 1.0)

    # Checks staircase repeats each level for the requested number of samples.
    def test_staircase_period_repeats_levels(self) -> None:
        data = self.waveforms.staircase_period(levels=[-1.0, 0.0, 1.0], samples_per_level=2)
        self.assertTrue(np.array_equal(data, np.array([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0])))

    # Checks staircase rejects empty level list.
    def test_staircase_period_rejects_empty_levels(self) -> None:
        with self.assertRaises(ValueError):
            self.waveforms.staircase_period(levels=[], samples_per_level=2)

    # Checks triangle waveform keeps requested sample count and stays in expected voltage range.
    def test_triangle_period_shape_and_range(self) -> None:
        data = self.waveforms.triangle_period(
            amplitude=1.5, offset=0.5, samples_per_period=100, symmetry=0.4
        )
        self.assertEqual(data.shape, (100,))
        self.assertTrue(np.max(data) <= 2.0 + 1e-12)
        self.assertTrue(np.min(data) >= -1.0 - 1e-12)

    # Checks triangle rejects invalid symmetry values.
    def test_triangle_period_rejects_invalid_symmetry(self) -> None:
        with self.assertRaises(ValueError):
            self.waveforms.triangle_period(
                amplitude=1.0, offset=0.0, samples_per_period=100, symmetry=1.0
            )

    # Checks square waveform uses duty cycle to split high and low sample counts.
    def test_square_period_duty_counts(self) -> None:
        data = self.waveforms.square_period(
            amplitude=1.0, offset=0.0, samples_per_period=10, duty=0.3
        )
        self.assertEqual(np.sum(np.isclose(data, 1.0)), 3)
        self.assertEqual(np.sum(np.isclose(data, -1.0)), 7)

    # Checks square rejects invalid duty values.
    def test_square_period_rejects_invalid_duty(self) -> None:
        with self.assertRaises(ValueError):
            self.waveforms.square_period(
                amplitude=1.0, offset=0.0, samples_per_period=10, duty=0.0
            )

    # Checks helper rejects period length above the default DAQ limit.
    def test_waveforms_reject_too_many_samples(self) -> None:
        with self.assertRaises(ValueError):
            self.waveforms.sine_period(
                amplitude=1.0,
                offset=0.0,
                samples_per_period=self.waveforms.DEFAULT_MAX_PERIOD_SAMPLES + 1,
            )

    # Checks helpers reject values outside configured voltage limits.
    def test_waveforms_reject_values_outside_voltage_limits(self) -> None:
        with self.assertRaises(ValueError):
            self.waveforms.square_period(
                amplitude=6.0,
                offset=0.0,
                samples_per_period=10,
                duty=0.5,
                min_voltage=-5.0,
                max_voltage=5.0,
            )


if __name__ == "__main__":
    unittest.main()
