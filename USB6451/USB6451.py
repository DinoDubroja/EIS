"""High-level control helpers for NI USB-6451 DAQ tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, RegenerationMode
try:
    from USB6451 import waveforms
except ImportError:
    import waveforms


@dataclass(frozen=True)
class ContinuousSineConfig:
    """Validated settings used to start continuous sine output.

    Inputs:
        device: Device name from NI MAX, for example ``"Dev1"``.
        ao_channel: Analog output channel without device prefix, for example ``"ao0"``.
        requested_frequency: Requested sine frequency in hertz (Hz).
        actual_frequency: Exact generated sine frequency in hertz (Hz).
        amplitude: Sine peak amplitude in volts (V).
        offset: DC offset in volts (V).
        sample_rate: Sample clock rate in samples/second (S/s).
        min_voltage: Minimum channel voltage in volts (V).
        max_voltage: Maximum channel voltage in volts (V).
        samples_per_period: Number of samples used for one sine period.
    Output:
        Immutable config object.
    """

    device: str
    ao_channel: str
    requested_frequency: float
    actual_frequency: float
    amplitude: float
    offset: float
    sample_rate: float
    min_voltage: float
    max_voltage: float
    samples_per_period: int


@dataclass(frozen=True)
class ContinuousPeriodicConfig:
    """Validated settings used to start continuous periodic output.

    Inputs:
        device: Device name from NI MAX, for example ``"Dev1"``.
        ao_channel: Analog output channel without device prefix, for example ``"ao0"``.
        sample_rate: Sample clock rate in samples/second (S/s).
        min_voltage: Minimum channel voltage in volts (V).
        max_voltage: Maximum channel voltage in volts (V).
        samples_per_period: Number of user-provided samples in one period.
        actual_frequency: Generated period frequency in hertz (Hz).
    Output:
        Immutable config object.
    """

    device: str
    ao_channel: str
    sample_rate: float
    min_voltage: float
    max_voltage: float
    samples_per_period: int
    actual_frequency: float


@dataclass(frozen=True)
class ContinuousInputConfig:
    """Validated settings used to start continuous analog input acquisition.

    Inputs:
        device: Device name from NI MAX, for example ``"Dev1"``.
        ai_channels: Tuple of AI channel names without device prefix.
        sample_rate: Requested sample clock rate in samples/second (S/s).
        min_voltage: Minimum input voltage in volts (V).
        max_voltage: Maximum input voltage in volts (V).
        actual_sample_rate: Actual sample rate configured by DAQmx.
    Output:
        Immutable config object.
    """

    device: str
    ai_channels: tuple[str, ...]
    sample_rate: float
    min_voltage: float
    max_voltage: float
    actual_sample_rate: float


@dataclass(frozen=True)
class ContinuousSyncPeriodicConfig:
    """Validated settings used to start synchronized continuous AO+AI operation.

    Inputs:
        device: Device name from NI MAX, for example ``"Dev1"``.
        ao_channel: AO channel name without device prefix.
        ai_channels: Tuple of AI channel names without device prefix.
        sample_rate: Requested shared sample clock in samples/second (S/s).
        actual_sample_rate: Actual shared sample rate configured by DAQmx.
        ao_min_voltage: AO lower voltage limit in volts (V).
        ao_max_voltage: AO upper voltage limit in volts (V).
        ai_min_voltage: AI lower voltage limit in volts (V).
        ai_max_voltage: AI upper voltage limit in volts (V).
        samples_per_period: Number of AO samples in one repeated period.
        output_frequency: AO period frequency in hertz (Hz).
    Output:
        Immutable config object.
    """

    device: str
    ao_channel: str
    ai_channels: tuple[str, ...]
    sample_rate: float
    actual_sample_rate: float
    ao_min_voltage: float
    ao_max_voltage: float
    ai_min_voltage: float
    ai_max_voltage: float
    samples_per_period: int
    output_frequency: float


class USB6451:
    """High-level wrapper for NI USB-6451 analog I/O operations."""

    # USB-6451 AO output FIFO size (manual): 16,383 samples shared among channels used.
    MAX_REGENERATIVE_PERIOD_SAMPLES = 16_383
    # USB-6451 AO spec from manual: max update rate 250 kS/s (all channels).
    MAX_AO_SAMPLE_RATE = 250_000.0
    # USB-6451 AI spec from manual: up to 1 MS/s/ch simultaneous sampling.
    MAX_AI_SAMPLE_RATE = 1_000_000.0
    # USB-6451 AI channels from manual: 16 single-ended or 8 differential.
    MAX_AI_CHANNELS = 16

    def __init__(self) -> None:
        """Create a new controller.

        Input:
            None.
        Output:
            New object with no active DAQ task.
        """

        self._ao_task: Optional[nidaqmx.Task] = None
        self._ai_task: Optional[nidaqmx.Task] = None
        self._sync_ao_task: Optional[nidaqmx.Task] = None
        self._sync_ai_task: Optional[nidaqmx.Task] = None
        self._ai_channel_count = 0
        self._sync_ai_channel_count = 0
        self._last_config: Optional[ContinuousSineConfig | ContinuousPeriodicConfig] = None
        self._last_input_config: Optional[ContinuousInputConfig] = None
        self._last_sync_config: Optional[ContinuousSyncPeriodicConfig] = None

    def start_continuous_sine_output(
        self,
        *,
        device: str = "Dev1",
        ao_channel: str = "ao0",
        frequency: float = 10.0,
        amplitude: float = 1.0,
        offset: float = 0.0,
        sample_rate: float = 10_000.0,
        samples_per_period: Optional[int] = None,
        min_voltage: float = -10.0,
        max_voltage: float = 10.0,
        allow_regen: bool = True,
    ) -> float:
        """Start continuous sine generation on one analog output channel.

        Inputs:
            device: Device name (for example ``"Dev1"``).
            ao_channel: AO channel name (for example ``"ao0"``).
            frequency: Requested sine frequency in hertz (Hz). Must be > 0.
            amplitude: Sine peak amplitude in volts (V). Must be >= 0.
            offset: DC offset in volts (V).
            sample_rate: Output sample clock in samples/second (S/s). Must be > 0.
            samples_per_period: Optional number of samples in one sine period.
                If omitted, it is computed from ``sample_rate / frequency``.
                If provided, it must be >= 8 and match the requested frequency.
            min_voltage: Lower output limit in volts (V).
            max_voltage: Upper output limit in volts (V).
            allow_regen: If ``True``, DAQ replays one written period continuously.
        Output:
            Exact generated frequency in hertz (Hz).
            If you only want to preview requested vs actual settings without starting
            output, use ``get_continuous_sine_output_config``.
        Raises:
            ValueError: Invalid input or waveform exceeds voltage limits.
            nidaqmx.DaqError: DAQ configuration/start failed.
        """

        config = self.get_continuous_sine_output_config(
            device=device,
            ao_channel=ao_channel,
            frequency=frequency,
            amplitude=amplitude,
            offset=offset,
            sample_rate=sample_rate,
            samples_per_period=samples_per_period,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
        )

        self.stop_output()

        period_data = waveforms.sine_period(
            amplitude=config.amplitude,
            offset=config.offset,
            samples_per_period=config.samples_per_period,
            min_voltage=config.min_voltage,
            max_voltage=config.max_voltage,
            max_samples_per_period=self.MAX_REGENERATIVE_PERIOD_SAMPLES,
        )

        physical_channel = f"{config.device}/{config.ao_channel}"
        task = nidaqmx.Task()
        try:
            task.ao_channels.add_ao_voltage_chan(
                physical_channel,
                min_val=config.min_voltage,
                max_val=config.max_voltage,
            )

            if not allow_regen:
                task.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION

            task.timing.cfg_samp_clk_timing(
                rate=config.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=config.samples_per_period,
            )
            task.write(period_data, auto_start=False)
            task.start()
        except Exception:
            task.close()
            raise

        self._ao_task = task
        self._last_config = config
        return config.actual_frequency

    def get_continuous_sine_output_config(
        self,
        *,
        device: str = "Dev1",
        ao_channel: str = "ao0",
        frequency: float = 10.0,
        amplitude: float = 1.0,
        offset: float = 0.0,
        sample_rate: float = 10_000.0,
        samples_per_period: Optional[int] = None,
        min_voltage: float = -10.0,
        max_voltage: float = 10.0,
    ) -> ContinuousSineConfig:
        """Validate requested sine settings and return exact realizable config.

        Inputs:
            device: Device name (for example ``"Dev1"``).
            ao_channel: AO channel name (for example ``"ao0"``).
            frequency: Requested sine frequency in hertz (Hz). Must be > 0.
            amplitude: Sine peak amplitude in volts (V). Must be >= 0.
            offset: DC offset in volts (V).
            sample_rate: Output sample clock in samples/second (S/s). Must be > 0.
            samples_per_period: Optional number of samples in one sine period.
                If omitted, it is computed from ``sample_rate / frequency``.
                If provided, it must be >= 8 and match the requested frequency.
            min_voltage: Lower output limit in volts (V).
            max_voltage: Upper output limit in volts (V).
        Output:
            ``ContinuousSineConfig`` containing requested values and exact values
            that this class can generate.
        Raises:
            ValueError: Invalid input or waveform exceeds voltage limits.
        """

        return self._validate_and_build_config(
            device=device,
            ao_channel=ao_channel,
            frequency=frequency,
            amplitude=amplitude,
            offset=offset,
            sample_rate=sample_rate,
            samples_per_period=samples_per_period,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
        )

    def stop_output(self) -> None:
        """Stop and release the active analog output task.

        Input:
            None.
        Output:
            None. Safe to call when no task is running.
        """

        if self._ao_task is None:
            return

        task = self._ao_task
        self._ao_task = None
        try:
            task.stop()
        except nidaqmx.DaqError:
            pass
        finally:
            task.close()

    def start_continuous_input(
        self,
        *,
        device: str = "Dev1",
        ai_channels: str | Sequence[str] = ("ai0",),
        sample_rate: float = 10_000.0,
        min_voltage: float = -10.0,
        max_voltage: float = 10.0,
        terminal_config=None,
    ) -> float:
        """Start continuous analog input acquisition using the internal clock.

        This follows NI's continuous AI example pattern:
        `analog_in/cont_voltage_acq_int_clk.py` (start, then read chunks in loop).

        Inputs:
            device: Device name (for example ``"Dev1"``).
            ai_channels: One channel name (``"ai0"``) or a sequence (``("ai0", "ai1")``).
            sample_rate: Requested AI sample clock in samples/second (S/s). Must be > 0
                and <= ``MAX_AI_SAMPLE_RATE``.
            min_voltage: Lower input limit in volts (V).
            max_voltage: Upper input limit in volts (V).
            terminal_config: Optional NI terminal configuration value to pass through to
                `add_ai_voltage_chan`.
        Output:
            Actual configured sample rate in samples/second (S/s).
        Raises:
            ValueError: Invalid inputs.
            nidaqmx.DaqError: DAQ configuration/start failed.
        """

        channels = self._normalize_ai_channels(ai_channels)
        self._validate_input_limits(
            device=device,
            ai_channels=channels,
            sample_rate=sample_rate,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
        )

        self.stop_input()

        task = nidaqmx.Task()
        try:
            for ch in channels:
                physical_channel = ch if "/" in ch else f"{device}/{ch}"
                if terminal_config is None:
                    task.ai_channels.add_ai_voltage_chan(
                        physical_channel,
                        min_val=min_voltage,
                        max_val=max_voltage,
                    )
                else:
                    task.ai_channels.add_ai_voltage_chan(
                        physical_channel,
                        min_val=min_voltage,
                        max_val=max_voltage,
                        terminal_config=terminal_config,
                    )

            task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
            )
            actual_sample_rate = float(task.timing.samp_clk_rate)
            task.start()
        except Exception:
            task.close()
            raise

        self._ai_task = task
        self._ai_channel_count = len(channels)
        self._last_input_config = ContinuousInputConfig(
            device=device.strip(),
            ai_channels=channels,
            sample_rate=sample_rate,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
            actual_sample_rate=actual_sample_rate,
        )
        return actual_sample_rate

    def read_input_chunk(
        self,
        *,
        samples_per_channel: int = 1000,
        timeout: float = 10.0,
    ) -> np.ndarray:
        """Read one chunk from the running continuous AI task.

        Inputs:
            samples_per_channel: Number of samples to read per channel. Must be >= 1.
            timeout: Read timeout in seconds.
        Output:
            ``numpy.ndarray`` with shape ``(channels, samples_per_channel)``.
        Raises:
            RuntimeError: If input task is not running.
            ValueError: If `samples_per_channel` is invalid.
            nidaqmx.DaqError: DAQ read failure.
        """

        if self._ai_task is None:
            raise RuntimeError("Input task is not running. Call start_continuous_input() first.")
        if samples_per_channel < 1:
            raise ValueError("samples_per_channel must be >= 1.")

        raw = self._ai_task.read(
            number_of_samples_per_channel=samples_per_channel,
            timeout=timeout,
        )
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            if self._ai_channel_count <= 1:
                return arr.reshape(1, -1)
            return arr.reshape(-1, 1)
        return arr

    def stop_input(self) -> None:
        """Stop and release the active analog input task.

        Input:
            None.
        Output:
            None. Safe to call when no input task is running.
        """

        if self._ai_task is None:
            return

        task = self._ai_task
        self._ai_task = None
        self._ai_channel_count = 0
        try:
            task.stop()
        except nidaqmx.DaqError:
            pass
        finally:
            task.close()

    def start_continuous_sync_periodic_io(
        self,
        *,
        period_samples: Sequence[float],
        sample_rate: float,
        device: str = "Dev1",
        ao_channel: str = "ao0",
        ai_channels: str | Sequence[str] = ("ai0",),
        ao_min_voltage: float = -10.0,
        ao_max_voltage: float = 10.0,
        ai_min_voltage: float = -10.0,
        ai_max_voltage: float = 10.0,
        ai_terminal_config=None,
    ) -> ContinuousSyncPeriodicConfig:
        """Start synchronized continuous periodic AO output and AI acquisition.

        This follows NI synchronization patterns shown in:
        - `examples/synchronization/multi_function/ai_ao_sync.py`
        - `examples/playrec.py`

        Inputs:
            period_samples: One AO waveform period in volts (V), repeated continuously.
            sample_rate: Shared AO/AI sample clock in samples/second (S/s).
            device: Device name (for example ``"Dev1"``).
            ao_channel: AO channel name (for example ``"ao0"``).
            ai_channels: AI channel name or sequence (for example ``("ai0", "ai1")``).
            ao_min_voltage: AO lower voltage limit in volts (V).
            ao_max_voltage: AO upper voltage limit in volts (V).
            ai_min_voltage: AI lower voltage limit in volts (V).
            ai_max_voltage: AI upper voltage limit in volts (V).
            ai_terminal_config: Optional NI terminal config passed to AI channels.
        Output:
            ``ContinuousSyncPeriodicConfig`` containing requested and actual settings.
        Raises:
            ValueError: Invalid inputs or limits.
            nidaqmx.DaqError: DAQ configuration/start failure.
        """

        channels = self._normalize_ai_channels(ai_channels)
        self._validate_input_limits(
            device=device,
            ai_channels=channels,
            sample_rate=sample_rate,
            min_voltage=ai_min_voltage,
            max_voltage=ai_max_voltage,
        )
        if sample_rate > self.MAX_AO_SAMPLE_RATE:
            raise ValueError(
                "sample_rate exceeds USB-6451 AO limit: "
                f"{sample_rate:g} > {self.MAX_AO_SAMPLE_RATE:g} S/s."
            )

        ao_config, period_data = self._validate_and_prepare_periodic_waveform(
            period_samples=period_samples,
            sample_rate=sample_rate,
            device=device,
            ao_channel=ao_channel,
            min_voltage=ao_min_voltage,
            max_voltage=ao_max_voltage,
        )

        self.stop_sync_io()
        self.stop_output()
        self.stop_input()

        ai_task = nidaqmx.Task()
        ao_task = nidaqmx.Task()
        try:
            for ch in channels:
                physical_channel = ch if "/" in ch else f"{device}/{ch}"
                if ai_terminal_config is None:
                    ai_task.ai_channels.add_ai_voltage_chan(
                        physical_channel,
                        min_val=ai_min_voltage,
                        max_val=ai_max_voltage,
                    )
                else:
                    ai_task.ai_channels.add_ai_voltage_chan(
                        physical_channel,
                        min_val=ai_min_voltage,
                        max_val=ai_max_voltage,
                        terminal_config=ai_terminal_config,
                    )

            ai_task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
            )

            ao_task.ao_channels.add_ao_voltage_chan(
                f"{device}/{ao_channel}",
                min_val=ao_min_voltage,
                max_val=ao_max_voltage,
            )
            ao_task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=ao_config.samples_per_period,
            )

            # NI pattern: AO waits on AI start trigger terminal.
            ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                ai_task.triggers.start_trigger.term
            )

            ao_task.write(period_data, auto_start=False)

            # NI pattern: start AO first, then AI.
            ao_task.start()
            ai_task.start()
            actual_sample_rate = float(ai_task.timing.samp_clk_rate)
        except Exception:
            try:
                ai_task.close()
            finally:
                ao_task.close()
            raise

        config = ContinuousSyncPeriodicConfig(
            device=device.strip(),
            ao_channel=ao_channel.strip(),
            ai_channels=channels,
            sample_rate=sample_rate,
            actual_sample_rate=actual_sample_rate,
            ao_min_voltage=ao_min_voltage,
            ao_max_voltage=ao_max_voltage,
            ai_min_voltage=ai_min_voltage,
            ai_max_voltage=ai_max_voltage,
            samples_per_period=ao_config.samples_per_period,
            output_frequency=ao_config.actual_frequency,
        )
        self._sync_ai_task = ai_task
        self._sync_ao_task = ao_task
        self._sync_ai_channel_count = len(channels)
        self._last_sync_config = config
        return config

    def read_sync_input_chunk(
        self,
        *,
        samples_per_channel: int = 1000,
        timeout: float = 10.0,
    ) -> np.ndarray:
        """Read one chunk from the synchronized AI task.

        Inputs:
            samples_per_channel: Number of samples per channel. Must be >= 1.
            timeout: Read timeout in seconds.
        Output:
            ``numpy.ndarray`` with shape ``(channels, samples_per_channel)``.
        Raises:
            RuntimeError: If synchronized task is not running.
            ValueError: If `samples_per_channel` is invalid.
            nidaqmx.DaqError: DAQ read failure.
        """

        if self._sync_ai_task is None:
            raise RuntimeError(
                "Synchronized input task is not running. "
                "Call start_continuous_sync_periodic_io() first."
            )
        if samples_per_channel < 1:
            raise ValueError("samples_per_channel must be >= 1.")

        raw = self._sync_ai_task.read(
            number_of_samples_per_channel=samples_per_channel,
            timeout=timeout,
        )
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            if self._sync_ai_channel_count <= 1:
                return arr.reshape(1, -1)
            return arr.reshape(-1, 1)
        return arr

    def stop_sync_io(self) -> None:
        """Stop and release active synchronized AO+AI tasks.

        Input:
            None.
        Output:
            None. Safe to call when sync tasks are not running.
        """

        ao_task = self._sync_ao_task
        ai_task = self._sync_ai_task
        self._sync_ao_task = None
        self._sync_ai_task = None
        self._sync_ai_channel_count = 0

        if ai_task is not None:
            try:
                ai_task.stop()
            except nidaqmx.DaqError:
                pass
            finally:
                ai_task.close()

        if ao_task is not None:
            try:
                ao_task.stop()
            except nidaqmx.DaqError:
                pass
            finally:
                ao_task.close()

    def is_sync_running(self) -> bool:
        """Return synchronized AO+AI running state.

        Input:
            None.
        Output:
            ``True`` when both sync tasks are active, otherwise ``False``.
        """

        return self._sync_ai_task is not None and self._sync_ao_task is not None

    def is_input_running(self) -> bool:
        """Return input running state.

        Input:
            None.
        Output:
            ``True`` when this object has an active AI task, otherwise ``False``.
        """

        return self._ai_task is not None

    def is_output_running(self) -> bool:
        """Return output running state.

        Input:
            None.
        Output:
            ``True`` when this object has an active AO task, otherwise ``False``.
        """

        return self._ao_task is not None

    def close(self) -> None:
        """Release all DAQ resources held by this object.

        Input:
            None.
        Output:
            None.
        """

        self.stop_output()
        self.stop_input()
        self.stop_sync_io()

    def start_continuous_periodic_output(
        self,
        *,
        period_samples: Sequence[float],
        sample_rate: float,
        device: str = "Dev1",
        ao_channel: str = "ao0",
        min_voltage: float = -10.0,
        max_voltage: float = 10.0,
    ) -> float:
        """Start continuous periodic output from user-provided one-period samples.

        Inputs:
            period_samples: One full waveform period as voltage samples in volts (V).
                Each value is output in order and then repeated continuously.
            sample_rate: Output sample clock in samples/second (S/s). Must be > 0.
            device: Device name (for example ``"Dev1"``).
            ao_channel: AO channel name (for example ``"ao0"``).
            min_voltage: Lower output limit in volts (V).
            max_voltage: Upper output limit in volts (V).
        Output:
            Exact generated waveform frequency in hertz (Hz), equal to
            ``sample_rate / len(period_samples)``.
            This mode is always regenerative because the provided samples are
            explicitly one periodic waveform cycle.
        Raises:
            ValueError: Invalid input, waveform exceeds voltage limits, or period
                length exceeds ``MAX_REGENERATIVE_PERIOD_SAMPLES``.
            nidaqmx.DaqError: DAQ configuration/start failed.
        """

        config, period_data = self._validate_and_prepare_periodic_waveform(
            period_samples=period_samples,
            sample_rate=sample_rate,
            device=device,
            ao_channel=ao_channel,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
        )

        self.stop_output()

        physical_channel = f"{config.device}/{config.ao_channel}"
        task = nidaqmx.Task()
        try:
            task.ao_channels.add_ao_voltage_chan(
                physical_channel,
                min_val=config.min_voltage,
                max_val=config.max_voltage,
            )

            task.timing.cfg_samp_clk_timing(
                rate=config.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=config.samples_per_period,
            )
            task.write(period_data, auto_start=False)
            task.start()
        except Exception:
            task.close()
            raise

        self._ao_task = task
        self._last_config = None
        return config.actual_frequency

    def _validate_and_build_config(
        self,
        *,
        device: str,
        ao_channel: str,
        frequency: float,
        amplitude: float,
        offset: float,
        sample_rate: float,
        samples_per_period: Optional[int],
        min_voltage: float,
        max_voltage: float,
    ) -> ContinuousSineConfig:
        """Validate inputs and build internal config.

        Inputs:
            Same as ``start_continuous_sine_output`` except ``allow_regen``.
        Output:
            ``ContinuousSineConfig`` with exact generated frequency.
        Raises:
            ValueError: Invalid input values.
        """

        if not device.strip():
            raise ValueError("device must not be empty.")
        if not ao_channel.strip():
            raise ValueError("ao_channel must not be empty.")
        if frequency <= 0:
            raise ValueError("frequency must be > 0.")
        if amplitude < 0:
            raise ValueError("amplitude must be >= 0.")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0.")
        if sample_rate > self.MAX_AO_SAMPLE_RATE:
            raise ValueError(
                "sample_rate exceeds USB-6451 AO limit: "
                f"{sample_rate:g} > {self.MAX_AO_SAMPLE_RATE:g} S/s."
            )
        if min_voltage >= max_voltage:
            raise ValueError("min_voltage must be smaller than max_voltage.")

        high = offset + amplitude
        low = offset - amplitude
        if high > max_voltage or low < min_voltage:
            raise ValueError(
                "Output waveform exceeds voltage limits: "
                f"[{low:.3f}, {high:.3f}] V is outside "
                f"[{min_voltage:.3f}, {max_voltage:.3f}] V."
            )

        if samples_per_period is None:
            computed_samples_per_period = int(round(sample_rate / frequency))
            if computed_samples_per_period < 8:
                raise ValueError(
                    "frequency is too high for this sample_rate. "
                    "Increase sample_rate or lower frequency."
                )
        else:
            computed_samples_per_period = samples_per_period
            if computed_samples_per_period < 8:
                raise ValueError("samples_per_period must be >= 8.")

        actual_frequency = sample_rate / computed_samples_per_period
        if samples_per_period is not None:
            mismatch = abs(actual_frequency - frequency)
            if mismatch > max(1e-9, 1e-6 * frequency):
                raise ValueError(
                    "samples_per_period does not match requested frequency at this sample_rate. "
                    f"Requested {frequency:.9g} Hz but configuration generates "
                    f"{actual_frequency:.9g} Hz."
                )

        return ContinuousSineConfig(
            device=device.strip(),
            ao_channel=ao_channel.strip(),
            requested_frequency=frequency,
            actual_frequency=actual_frequency,
            amplitude=amplitude,
            offset=offset,
            sample_rate=sample_rate,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
            samples_per_period=computed_samples_per_period,
        )

    def _validate_and_prepare_periodic_waveform(
        self,
        *,
        period_samples: Sequence[float],
        sample_rate: float,
        device: str,
        ao_channel: str,
        min_voltage: float,
        max_voltage: float,
    ) -> tuple[ContinuousPeriodicConfig, np.ndarray]:
        """Validate user periodic waveform samples and return config + normalized data.

        Inputs:
            period_samples: One full waveform period as voltage samples in volts (V).
            sample_rate: Output sample clock in samples/second (S/s). Must be > 0.
            device: Device name.
            ao_channel: AO channel name.
            min_voltage: Lower output limit in volts (V).
            max_voltage: Upper output limit in volts (V).
        Output:
            Tuple of:
            1) ``ContinuousPeriodicConfig`` with exact generated frequency.
            2) ``numpy.ndarray`` period samples as ``float64``.
        Raises:
            ValueError: Invalid input values or waveform outside voltage limits.
        """

        if not device.strip():
            raise ValueError("device must not be empty.")
        if not ao_channel.strip():
            raise ValueError("ao_channel must not be empty.")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0.")
        if min_voltage >= max_voltage:
            raise ValueError("min_voltage must be smaller than max_voltage.")

        period_data = np.asarray(period_samples, dtype=np.float64)
        if period_data.ndim != 1:
            raise ValueError("period_samples must be a one-dimensional list/array.")
        if period_data.size < 1:
            raise ValueError("period_samples must contain at least one sample.")
        if period_data.size > self.MAX_REGENERATIVE_PERIOD_SAMPLES:
            raise ValueError(
                "period_samples is too long for regenerative periodic mode: "
                f"{period_data.size} samples exceeds "
                f"{self.MAX_REGENERATIVE_PERIOD_SAMPLES} samples."
            )
        if not np.all(np.isfinite(period_data)):
            raise ValueError("period_samples must contain only finite numbers.")

        high = float(np.max(period_data))
        low = float(np.min(period_data))
        if high > max_voltage or low < min_voltage:
            raise ValueError(
                "Output waveform exceeds voltage limits: "
                f"[{low:.3f}, {high:.3f}] V is outside "
                f"[{min_voltage:.3f}, {max_voltage:.3f}] V."
            )

        samples_per_period = int(period_data.size)
        actual_frequency = sample_rate / samples_per_period
        config = ContinuousPeriodicConfig(
            device=device.strip(),
            ao_channel=ao_channel.strip(),
            sample_rate=sample_rate,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
            samples_per_period=samples_per_period,
            actual_frequency=actual_frequency,
        )
        return config, period_data

    @staticmethod
    def _normalize_ai_channels(ai_channels: str | Sequence[str]) -> tuple[str, ...]:
        """Normalize AI channel input to a non-empty tuple of channel strings."""
        if isinstance(ai_channels, str):
            channels = (ai_channels.strip(),)
        else:
            channels = tuple(str(ch).strip() for ch in ai_channels)
        channels = tuple(ch for ch in channels if ch)
        if not channels:
            raise ValueError("ai_channels must contain at least one channel name.")
        return channels

    def _validate_input_limits(
        self,
        *,
        device: str,
        ai_channels: tuple[str, ...],
        sample_rate: float,
        min_voltage: float,
        max_voltage: float,
    ) -> None:
        """Validate continuous AI limits using USB-6451 constraints."""
        if not device.strip():
            raise ValueError("device must not be empty.")
        if len(ai_channels) > self.MAX_AI_CHANNELS:
            raise ValueError(
                f"Too many ai_channels: {len(ai_channels)} exceeds {self.MAX_AI_CHANNELS}."
            )
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0.")
        if sample_rate > self.MAX_AI_SAMPLE_RATE:
            raise ValueError(
                "sample_rate exceeds USB-6451 AI limit: "
                f"{sample_rate:g} > {self.MAX_AI_SAMPLE_RATE:g} S/s."
            )
        if min_voltage >= max_voltage:
            raise ValueError("min_voltage must be smaller than max_voltage.")
