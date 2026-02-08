"""Unit tests for USB6451 class behavior.

These tests avoid hardware access and NI driver dependency.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


def _install_fake_nidaqmx() -> None:
    """Install a small nidaqmx stub for unit tests."""
    if "nidaqmx" in sys.modules and "nidaqmx.constants" in sys.modules:
        return

    fake_nidaqmx = types.ModuleType("nidaqmx")
    fake_constants = types.ModuleType("nidaqmx.constants")

    class DaqError(Exception):
        """Stub DAQ error."""

    class Task:
        """Stub task type used only for type references in unit tests."""

    class AcquisitionType:
        CONTINUOUS = "CONTINUOUS"

    class RegenerationMode:
        DONT_ALLOW_REGENERATION = "DONT_ALLOW_REGENERATION"

    fake_nidaqmx.DaqError = DaqError
    fake_nidaqmx.Task = Task
    fake_constants.AcquisitionType = AcquisitionType
    fake_constants.RegenerationMode = RegenerationMode

    sys.modules["nidaqmx"] = fake_nidaqmx
    sys.modules["nidaqmx.constants"] = fake_constants


def _load_usb6451_module():
    """Load USB6451 module from source file path."""
    _install_fake_nidaqmx()
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "USB6451" / "USB6451.py"
    module_name = "usb6451_module_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestUSB6451Unit(unittest.TestCase):
    """Unit tests for non-hardware behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_usb6451_module()
        cls.USB6451 = cls.mod.USB6451

    def setUp(self) -> None:
        self.dev = self.USB6451()

    # Checks public config preview returns requested/actual values without hardware use.
    def test_get_config_returns_expected_requested_and_actual_values(self) -> None:
        config = self.dev.get_continuous_sine_output_config(
            device="Dev1",
            ao_channel="ao0",
            frequency=17.0,
            amplitude=1.0,
            offset=0.0,
            sample_rate=10_000.0,
            samples_per_period=None,
            min_voltage=-10.0,
            max_voltage=10.0,
        )
        self.assertEqual(config.requested_frequency, 17.0)
        self.assertEqual(config.samples_per_period, 588)
        self.assertAlmostEqual(config.actual_frequency, 10_000.0 / 588)

    # Checks periodic waveform config computes expected output frequency.
    def test_periodic_waveform_config_returns_expected_frequency(self) -> None:
        config, data = self.dev._validate_and_prepare_periodic_waveform(
            period_samples=[0.0, 1.0, 0.0, -1.0],
            sample_rate=4000.0,
            device="Dev1",
            ao_channel="ao0",
            min_voltage=-10.0,
            max_voltage=10.0,
        )
        self.assertEqual(config.samples_per_period, 4)
        self.assertAlmostEqual(config.actual_frequency, 1000.0)
        self.assertTrue(np.array_equal(data, np.array([0.0, 1.0, 0.0, -1.0])))

    # Checks periodic waveform rejects values outside min/max voltage limits.
    def test_periodic_waveform_rejects_values_outside_voltage_limits(self) -> None:
        with self.assertRaises(ValueError):
            self.dev._validate_and_prepare_periodic_waveform(
                period_samples=[0.0, 6.0, -6.0],
                sample_rate=1000.0,
                device="Dev1",
                ao_channel="ao0",
                min_voltage=-5.0,
                max_voltage=5.0,
            )

    # Checks periodic waveform rejects empty sample list.
    def test_periodic_waveform_rejects_empty_sample_list(self) -> None:
        with self.assertRaises(ValueError):
            self.dev._validate_and_prepare_periodic_waveform(
                period_samples=[],
                sample_rate=1000.0,
                device="Dev1",
                ao_channel="ao0",
                min_voltage=-10.0,
                max_voltage=10.0,
            )

    # Checks periodic waveform rejects period longer than regenerative sample limit.
    def test_periodic_waveform_rejects_too_many_samples_for_regen(self) -> None:
        with self.assertRaises(ValueError):
            self.dev._validate_and_prepare_periodic_waveform(
                period_samples=[0.0] * (self.dev.MAX_REGENERATIVE_PERIOD_SAMPLES + 1),
                sample_rate=1000.0,
                device="Dev1",
                ao_channel="ao0",
                min_voltage=-10.0,
                max_voltage=10.0,
            )

    # Checks that automatic sample-count calculation gives expected values.
    def test_validate_computes_expected_samples_per_period(self) -> None:
        config = self.dev._validate_and_build_config(
            device="Dev1",
            ao_channel="ao0",
            frequency=10.0,
            amplitude=1.0,
            offset=0.0,
            sample_rate=10_000.0,
            samples_per_period=None,
            min_voltage=-10.0,
            max_voltage=10.0,
        )
        self.assertEqual(config.samples_per_period, 1000)
        self.assertAlmostEqual(config.actual_frequency, 10.0)

    # Checks that zero or negative frequency is rejected.
    def test_validate_rejects_zero_or_negative_frequency(self) -> None:
        with self.assertRaises(ValueError):
            self.dev._validate_and_build_config(
                device="Dev1",
                ao_channel="ao0",
                frequency=0.0,
                amplitude=1.0,
                offset=0.0,
                sample_rate=10_000.0,
                samples_per_period=None,
                min_voltage=-10.0,
                max_voltage=10.0,
            )

    # Checks that waveform amplitude/offset outside voltage limits is rejected.
    def test_validate_rejects_waveform_outside_voltage_limits(self) -> None:
        with self.assertRaises(ValueError):
            self.dev._validate_and_build_config(
                device="Dev1",
                ao_channel="ao0",
                frequency=10.0,
                amplitude=6.0,
                offset=0.0,
                sample_rate=10_000.0,
                samples_per_period=None,
                min_voltage=-5.0,
                max_voltage=5.0,
            )

    # Checks that provided samples_per_period must match requested frequency.
    def test_validate_rejects_samples_per_period_frequency_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            self.dev._validate_and_build_config(
                device="Dev1",
                ao_channel="ao0",
                frequency=10.0,
                amplitude=1.0,
                offset=0.0,
                sample_rate=10_000.0,
                samples_per_period=800,
                min_voltage=-10.0,
                max_voltage=10.0,
            )

    # Checks a fresh object reports output as not running.
    def test_new_instance_not_running(self) -> None:
        self.assertFalse(self.dev.is_output_running())

    # Checks stop_output is safe to call even when no task exists.
    def test_stop_output_without_task_is_safe(self) -> None:
        self.dev.stop_output()
        self.assertFalse(self.dev.is_output_running())


if __name__ == "__main__":
    unittest.main()
