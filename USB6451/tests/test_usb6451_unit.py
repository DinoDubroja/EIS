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

    class _FakeOutStream:
        def __init__(self) -> None:
            self.regen_mode = None

    class _FakeStartTrigger:
        def __init__(self) -> None:
            self.term = "/Dev1/ai/StartTrigger"
            self.source = None

        def cfg_dig_edge_start_trig(self, trigger_source):
            self.source = trigger_source

    class _FakeTriggers:
        def __init__(self) -> None:
            self.start_trigger = _FakeStartTrigger()

    class _FakeTiming:
        def __init__(self) -> None:
            self.samp_clk_rate = 0.0

        def cfg_samp_clk_timing(self, rate, sample_mode=None, samps_per_chan=None):
            self.samp_clk_rate = float(rate)

    class _FakeAOChannels:
        def __init__(self) -> None:
            self.channels = []

        def add_ao_voltage_chan(self, physical_channel, min_val=None, max_val=None):
            self.channels.append(
                {"physical_channel": physical_channel, "min_val": min_val, "max_val": max_val}
            )

    class _FakeAIChannels:
        def __init__(self) -> None:
            self.channels = []

        def add_ai_voltage_chan(
            self,
            physical_channel,
            min_val=None,
            max_val=None,
            terminal_config=None,
        ):
            self.channels.append(
                {
                    "physical_channel": physical_channel,
                    "min_val": min_val,
                    "max_val": max_val,
                    "terminal_config": terminal_config,
                }
            )

    class Task:
        """Stub task with basic AO/AI/timing/read/write behavior."""

        def __init__(self) -> None:
            self.out_stream = _FakeOutStream()
            self.triggers = _FakeTriggers()
            self.timing = _FakeTiming()
            self.ao_channels = _FakeAOChannels()
            self.ai_channels = _FakeAIChannels()
            self.started = False
            self.closed = False

        def write(self, data, auto_start=False):
            if auto_start:
                self.start()
            arr = np.asarray(data)
            if arr.ndim == 0:
                return 1
            return int(arr.shape[-1])

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

        def close(self):
            self.closed = True

        def read(self, number_of_samples_per_channel=1, timeout=10.0):
            n = int(number_of_samples_per_channel)
            ch = len(self.ai_channels.channels)
            if ch <= 1:
                return [float(i) for i in range(n)]
            return [[float(i + 1000 * c) for i in range(n)] for c in range(ch)]

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

    # Checks continuous input can start and returns configured sample rate.
    def test_start_continuous_input_returns_actual_sample_rate(self) -> None:
        actual = self.dev.start_continuous_input(
            device="Dev1",
            ai_channels=("ai0", "ai1"),
            sample_rate=20_000.0,
            min_voltage=-10.0,
            max_voltage=10.0,
        )
        self.assertAlmostEqual(actual, 20_000.0)
        self.assertTrue(self.dev.is_input_running())

    # Checks input read chunk returns channels x samples matrix for multi-channel read.
    def test_read_input_chunk_shape_multi_channel(self) -> None:
        self.dev.start_continuous_input(
            device="Dev1",
            ai_channels=("ai0", "ai1"),
            sample_rate=10_000.0,
            min_voltage=-10.0,
            max_voltage=10.0,
        )
        data = self.dev.read_input_chunk(samples_per_channel=5)
        self.assertEqual(data.shape, (2, 5))

    # Checks read_input_chunk rejects calls when input task is not running.
    def test_read_input_chunk_requires_running_task(self) -> None:
        with self.assertRaises(RuntimeError):
            self.dev.read_input_chunk(samples_per_channel=5)

    # Checks stop_input is safe to call even when no input task exists.
    def test_stop_input_without_task_is_safe(self) -> None:
        self.dev.stop_input()
        self.assertFalse(self.dev.is_input_running())

    # Checks input validation rejects too many channels.
    def test_start_continuous_input_rejects_too_many_channels(self) -> None:
        channels = tuple(f"ai{i}" for i in range(self.dev.MAX_AI_CHANNELS + 1))
        with self.assertRaises(ValueError):
            self.dev.start_continuous_input(
                device="Dev1",
                ai_channels=channels,
                sample_rate=10_000.0,
                min_voltage=-10.0,
                max_voltage=10.0,
            )

    # Checks synchronized periodic IO start returns config and running state.
    def test_start_continuous_sync_periodic_io_returns_config(self) -> None:
        config = self.dev.start_continuous_sync_periodic_io(
            period_samples=[0.0, 1.0, 0.0, -1.0],
            sample_rate=20_000.0,
            device="Dev1",
            ao_channel="ao0",
            ai_channels=("ai0", "ai1"),
        )
        self.assertTrue(self.dev.is_sync_running())
        self.assertEqual(config.samples_per_period, 4)
        self.assertAlmostEqual(config.output_frequency, 5_000.0)
        self.assertAlmostEqual(config.actual_sample_rate, 20_000.0)

    # Checks synchronized read returns channels x samples matrix.
    def test_read_sync_input_chunk_shape_multi_channel(self) -> None:
        self.dev.start_continuous_sync_periodic_io(
            period_samples=[0.0, 1.0, 0.0, -1.0],
            sample_rate=10_000.0,
            device="Dev1",
            ao_channel="ao0",
            ai_channels=("ai0", "ai1"),
        )
        data = self.dev.read_sync_input_chunk(samples_per_channel=5)
        self.assertEqual(data.shape, (2, 5))

    # Checks synchronized read rejects calls when sync tasks are not running.
    def test_read_sync_input_chunk_requires_running_task(self) -> None:
        with self.assertRaises(RuntimeError):
            self.dev.read_sync_input_chunk(samples_per_channel=5)

    # Checks synchronized stop is safe to call when no sync tasks exist.
    def test_stop_sync_io_without_task_is_safe(self) -> None:
        self.dev.stop_sync_io()
        self.assertFalse(self.dev.is_sync_running())

    # Checks synchronized start wires AO start trigger to AI start trigger terminal.
    def test_sync_start_trigger_wiring(self) -> None:
        self.dev.start_continuous_sync_periodic_io(
            period_samples=[0.0, 1.0, 0.0, -1.0],
            sample_rate=10_000.0,
            device="Dev1",
            ao_channel="ao0",
            ai_channels=("ai0",),
        )
        self.assertEqual(
            self.dev._sync_ao_task.triggers.start_trigger.source,
            self.dev._sync_ai_task.triggers.start_trigger.term,
        )

    # Checks synchronized start rejects sample rates above AO limit.
    def test_start_continuous_sync_periodic_io_rejects_ao_sample_rate_limit(self) -> None:
        with self.assertRaises(ValueError):
            self.dev.start_continuous_sync_periodic_io(
                period_samples=[0.0, 1.0, 0.0, -1.0],
                sample_rate=self.dev.MAX_AO_SAMPLE_RATE + 1,
                device="Dev1",
                ao_channel="ao0",
                ai_channels=("ai0",),
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
