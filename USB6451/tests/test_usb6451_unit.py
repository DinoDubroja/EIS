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
            self.sample_mode = None
            self.samps_per_chan = None

        def cfg_samp_clk_timing(self, rate, sample_mode=None, samps_per_chan=None):
            self.samp_clk_rate = float(rate)
            self.sample_mode = sample_mode
            self.samps_per_chan = samps_per_chan

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

        created_tasks = []

        def __init__(self) -> None:
            self.out_stream = _FakeOutStream()
            self.triggers = _FakeTriggers()
            self.timing = _FakeTiming()
            self.ao_channels = _FakeAOChannels()
            self.ai_channels = _FakeAIChannels()
            self.started = False
            self.closed = False
            self.write_calls = []
            Task.created_tasks.append(self)

        def write(self, data, auto_start=False):
            if auto_start:
                self.start()
            arr = np.asarray(data)
            self.write_calls.append(arr.copy())
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
        FINITE = "FINITE"

    class RegenerationMode:
        DONT_ALLOW_REGENERATION = "DONT_ALLOW_REGENERATION"

    class TerminalConfiguration:
        DIFFERENTIAL = "DIFFERENTIAL"
        RSE = "RSE"
        NRSE = "NRSE"
        PSEUDODIFFERENTIAL = "PSEUDODIFFERENTIAL"

    fake_nidaqmx.DaqError = DaqError
    fake_nidaqmx.Task = Task
    fake_constants.AcquisitionType = AcquisitionType
    fake_constants.RegenerationMode = RegenerationMode
    fake_constants.TerminalConfiguration = TerminalConfiguration

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
        nidaq_task = sys.modules["nidaqmx"].Task
        nidaq_task.created_tasks.clear()

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

    # Checks non-regen sine start disables regeneration and starts AO task.
    def test_start_continuous_sine_output_non_regen_starts_and_disables_regen(self) -> None:
        actual = self.dev.start_continuous_sine_output_non_regen(
            device="Dev1",
            ao_channel="ao0",
            frequency=17.0,
            amplitude=1.0,
            offset=0.0,
            sample_rate=10_000.0,
            chunk_samples=250,
            min_voltage=-10.0,
            max_voltage=10.0,
        )
        self.assertAlmostEqual(actual, 10_000.0)
        self.assertTrue(self.dev.is_output_running())
        self.assertEqual(
            self.dev._ao_task.out_stream.regen_mode,
            self.mod.RegenerationMode.DONT_ALLOW_REGENERATION,
        )
        self.assertEqual(int(self.dev._ao_task.write_calls[0].shape[0]), 250)

    # Checks non-regen sine chunk writer appends data and advances internal phase.
    def test_write_sine_chunk_non_regen_writes_and_advances_phase(self) -> None:
        self.dev.start_continuous_sine_output_non_regen(
            device="Dev1",
            ao_channel="ao0",
            frequency=17.0,
            amplitude=1.0,
            offset=0.0,
            sample_rate=10_000.0,
            chunk_samples=100,
        )
        phase_before = self.dev._non_regen_phase
        written = self.dev.write_sine_chunk_non_regen(chunk_samples=120)
        self.assertEqual(written, 120)
        self.assertEqual(len(self.dev._ao_task.write_calls), 2)
        self.assertEqual(int(self.dev._ao_task.write_calls[1].shape[0]), 120)
        self.assertNotEqual(self.dev._non_regen_phase, phase_before)

    # Checks non-regen sine chunk writer rejects calls before non-regen start.
    def test_write_sine_chunk_non_regen_requires_active_output(self) -> None:
        with self.assertRaises(RuntimeError):
            self.dev.write_sine_chunk_non_regen(chunk_samples=100)

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

    # Checks input_mode="differential" is passed to NI AI channel config.
    def test_start_continuous_input_applies_differential_mode(self) -> None:
        self.dev.start_continuous_input(
            device="Dev1",
            ai_channels=("ai0",),
            sample_rate=10_000.0,
            min_voltage=-10.0,
            max_voltage=10.0,
            input_mode="differential",
        )
        term_cfg = self.dev._ai_task.ai_channels.channels[0]["terminal_config"]
        self.assertEqual(term_cfg, self.mod.TerminalConfiguration.DIFFERENTIAL)

    # Checks differential mode rejects more than 8 AI channels (USB-6451 limit).
    def test_start_continuous_input_rejects_too_many_differential_channels(self) -> None:
        channels = tuple(f"ai{i}" for i in range(self.dev.MAX_AI_DIFF_CHANNELS + 1))
        with self.assertRaises(ValueError):
            self.dev.start_continuous_input(
                device="Dev1",
                ai_channels=channels,
                sample_rate=10_000.0,
                min_voltage=-10.0,
                max_voltage=10.0,
                input_mode="differential",
            )

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

    # Checks finite AI read returns channels x samples array.
    def test_measure_input_finite_shape_multi_channel(self) -> None:
        data = self.dev.measure_input_finite(
            samples_per_channel=6,
            sample_rate=15_000.0,
            device="Dev1",
            ai_channels=("ai0", "ai1"),
            min_voltage=-10.0,
            max_voltage=10.0,
        )
        self.assertEqual(data.shape, (2, 6))

    # Checks finite AI rejects unknown input mode names.
    def test_measure_input_finite_rejects_unknown_input_mode(self) -> None:
        with self.assertRaises(ValueError):
            self.dev.measure_input_finite(
                samples_per_channel=6,
                sample_rate=15_000.0,
                device="Dev1",
                ai_channels=("ai0",),
                min_voltage=-10.0,
                max_voltage=10.0,
                input_mode="unsupported",
            )

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

    # Checks synchronized start applies differential input mode to AI channels.
    def test_start_continuous_sync_periodic_io_applies_differential_mode(self) -> None:
        self.dev.start_continuous_sync_periodic_io(
            period_samples=[0.0, 1.0, 0.0, -1.0],
            sample_rate=20_000.0,
            device="Dev1",
            ao_channel="ao0",
            ai_channels=("ai0",),
            input_mode="differential",
        )
        term_cfg = self.dev._sync_ai_task.ai_channels.channels[0]["terminal_config"]
        self.assertEqual(term_cfg, self.mod.TerminalConfiguration.DIFFERENTIAL)

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

    # Checks finite synchronized AO+AI measurement returns channels x samples.
    def test_measure_sync_finite_shape_multi_channel(self) -> None:
        data = self.dev.measure_sync_finite(
            output_samples=[0.0, 1.0, 0.0, -1.0, 0.5, -0.5],
            sample_rate=10_000.0,
            device="Dev1",
            ao_channel="ao0",
            ai_channels=("ai0", "ai1"),
        )
        self.assertEqual(data.shape, (2, 6))

    # Checks finite synchronized method wires AO trigger source to AI trigger terminal.
    def test_measure_sync_finite_trigger_wiring(self) -> None:
        self.dev.measure_sync_finite(
            output_samples=[0.0, 1.0, 0.0, -1.0],
            sample_rate=10_000.0,
            device="Dev1",
            ao_channel="ao0",
            ai_channels=("ai0",),
        )
        created = sys.modules["nidaqmx"].Task.created_tasks
        ai_task = created[-2]
        ao_task = created[-1]
        self.assertEqual(ao_task.triggers.start_trigger.source, ai_task.triggers.start_trigger.term)

    # Checks sine-period finite measurement returns expected sample count for N periods.
    def test_measure_sine_periods_returns_expected_shape(self) -> None:
        data = self.dev.measure_sine_periods(
            periods=3,
            frequency=10.0,
            amplitude=1.0,
            offset=0.0,
            sample_rate=10_000.0,
            samples_per_period=1000,
            device="Dev1",
            ao_channel="ao0",
            ai_channels=("ai0", "ai1"),
        )
        self.assertEqual(data.shape, (2, 3000))

    # Checks automatic integer-divider path keeps exact period-repeat sample count.
    def test_measure_sine_periods_integer_divider_auto_shape(self) -> None:
        data = self.dev.measure_sine_periods(
            periods=3,
            frequency=10.0,
            amplitude=1.0,
            offset=0.0,
            sample_rate=10_000.0,
            samples_per_period=None,
            device="Dev1",
            ao_channel="ao0",
            ai_channels=("ai0", "ai1"),
        )
        self.assertEqual(data.shape, (2, 3000))

    # Checks automatic non-integer-divider path uses rounded continuous-phase sample count.
    def test_measure_sine_periods_non_integer_divider_auto_shape(self) -> None:
        data = self.dev.measure_sine_periods(
            periods=3,
            frequency=17.0,
            amplitude=1.0,
            offset=0.0,
            sample_rate=10_000.0,
            samples_per_period=None,
            device="Dev1",
            ao_channel="ao0",
            ai_channels=("ai0", "ai1"),
        )
        self.assertEqual(data.shape, (2, 1765))

    # Checks sine-period finite measurement rejects non-positive period count.
    def test_measure_sine_periods_rejects_non_positive_periods(self) -> None:
        with self.assertRaises(ValueError):
            self.dev.measure_sine_periods(
                periods=0,
                frequency=10.0,
                amplitude=1.0,
                offset=0.0,
                sample_rate=10_000.0,
                device="Dev1",
                ao_channel="ao0",
                ai_channels=("ai0",),
            )

    # Checks synchronized connection preflight returns expected shape summary.
    def test_validate_sync_connection_returns_expected_shape(self) -> None:
        result = self.dev.validate_sync_connection(
            device="Dev1",
            ao_channel="ao0",
            ai_channels=("ai0", "ai7"),
            sample_rate=20_000.0,
            samples_per_channel=64,
            ao_test_voltage=1.0,
            settle_discard_s=0.0,
            expected_current_channel_voltage_v=31.5,
            current_channel_tolerance_v=1.0,
            current_channel_index=0,
            input_mode="differential",
        )
        self.assertEqual(result.device, "Dev1")
        self.assertEqual(result.ao_channel, "ao0")
        self.assertEqual(result.ai_channels, ("ai0", "ai7"))
        self.assertEqual(result.samples_per_channel, 64)
        self.assertEqual(result.measured_shape, (2, 64))

    # Checks synchronized connection preflight rejects AO test voltage outside limits.
    def test_validate_sync_connection_rejects_test_voltage_outside_limits(self) -> None:
        with self.assertRaises(ValueError):
            self.dev.validate_sync_connection(
                device="Dev1",
                ao_channel="ao0",
                ai_channels=("ai0",),
                sample_rate=20_000.0,
                samples_per_channel=64,
                ao_test_voltage=11.0,
                ao_min_voltage=-10.0,
                ao_max_voltage=10.0,
            )

    # Checks preflight rejects settle discard windows longer than captured record.
    def test_validate_sync_connection_rejects_too_large_settle_discard(self) -> None:
        with self.assertRaises(ValueError):
            self.dev.validate_sync_connection(
                device="Dev1",
                ao_channel="ao0",
                ai_channels=("ai0",),
                sample_rate=20_000.0,
                samples_per_channel=64,
                ao_test_voltage=0.0,
                settle_discard_s=0.01,
            )

    # Checks preflight fails when current-channel shunt mean is outside tolerance.
    def test_validate_sync_connection_fails_shunt_tolerance_check(self) -> None:
        with self.assertRaises(RuntimeError):
            self.dev.validate_sync_connection(
                device="Dev1",
                ao_channel="ao0",
                ai_channels=("ai0", "ai7"),
                sample_rate=20_000.0,
                samples_per_channel=64,
                ao_test_voltage=1.0,
                settle_discard_s=0.0,
                expected_current_channel_voltage_v=0.0,
                current_channel_tolerance_v=0.1,
            )

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

    # Checks stop_output clears non-regen state when non-regen output is active.
    def test_stop_output_clears_non_regen_state(self) -> None:
        self.dev.start_continuous_sine_output_non_regen(
            device="Dev1",
            ao_channel="ao0",
            frequency=17.0,
            amplitude=1.0,
            offset=0.0,
            sample_rate=10_000.0,
            chunk_samples=100,
        )
        self.dev.stop_output()
        self.assertFalse(self.dev._non_regen_sine_active)


if __name__ == "__main__":
    unittest.main()
