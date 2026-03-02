"""High-level control helpers for NI USB-6451 DAQ tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, RegenerationMode, TerminalConfiguration
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


@dataclass(frozen=True)
class SyncConnectionValidationResult:
    """Result summary from a short synchronized AO+AI connection validation run.

    Inputs:
        device: Device name from NI MAX, for example ``"Dev1"``.
        ao_channel: AO channel used during validation.
        ai_channels: Tuple of AI channels used during validation.
        sample_rate: Shared AO/AI sample clock in samples/second (S/s).
        samples_per_channel: Number of samples captured per AI channel.
        measured_shape: Returned AI array shape as ``(channels, samples)``.
        message: Human-readable status message suitable for logs/UI.
    Output:
        Immutable validation report.
    """

    device: str
    ao_channel: str
    ai_channels: tuple[str, ...]
    sample_rate: float
    samples_per_channel: int
    measured_shape: tuple[int, int]
    message: str


class USB6451:
    """High-level wrapper for NI USB-6451 analog I/O operations.

    Purpose:
        Provide readable, reusable APIs for synchronized and non-synchronized
        AO/AI workflows used by impedance measurement notebooks and scripts.
    Scope:
        - continuous AO generation (periodic and non-regenerative)
        - continuous and finite AI measurement
        - finite synchronized AO+AI capture
        - finite sine-period helper and connection preflight checks
    """

    # USB-6451 AO output FIFO size (manual): 16,383 samples shared among channels used.
    MAX_REGENERATIVE_PERIOD_SAMPLES = 16_383
    # USB-6451 AO spec from manual: max update rate 250 kS/s (all channels).
    MAX_AO_SAMPLE_RATE = 250_000.0
    # USB-6451 AI spec from manual: up to 1 MS/s/ch simultaneous sampling.
    MAX_AI_SAMPLE_RATE = 1_000_000.0
    # USB-6451 AI channels from manual: 16 single-ended or 8 differential.
    MAX_AI_CHANNELS = 16
    MAX_AI_DIFF_CHANNELS = 8

    def __init__(self) -> None:
        """Create a new controller.

        Purpose:
            Initialize USB-6451 task handles and internal state used by this API.

        Inputs:
            None.
        Output:
            New object with no active DAQ task.
        """

        # Long-lived task handles for continuous operation APIs.
        self._ao_task: Optional[nidaqmx.Task] = None
        self._ai_task: Optional[nidaqmx.Task] = None
        self._sync_ao_task: Optional[nidaqmx.Task] = None
        self._sync_ai_task: Optional[nidaqmx.Task] = None
        # Channel counts are tracked so readback shape stays deterministic.
        self._ai_channel_count = 0
        self._sync_ai_channel_count = 0
        # Cached configs simplify diagnostics after a run.
        self._last_config: Optional[ContinuousSineConfig | ContinuousPeriodicConfig] = None
        self._last_input_config: Optional[ContinuousInputConfig] = None
        self._last_sync_config: Optional[ContinuousSyncPeriodicConfig] = None
        # Phase tracking for non-regenerative sine streaming.
        self._non_regen_sine_active = False
        self._non_regen_phase = 0.0
        self._non_regen_frequency = 0.0
        self._non_regen_amplitude = 0.0
        self._non_regen_offset = 0.0
        self._non_regen_sample_rate = 0.0
        self._non_regen_min_voltage = -10.0
        self._non_regen_max_voltage = 10.0

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

        Purpose:
            Generate a steady sine signal on AO for stimulus or quick lab checks.

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
                For true non-regenerative streaming, use
                ``start_continuous_sine_output_non_regen(...)``.
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

        Purpose:
            Preview the exact realizable sine settings before starting output.

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

    def start_continuous_sine_output_non_regen(
        self,
        *,
        device: str = "Dev1",
        ao_channel: str = "ao0",
        frequency: float = 17.0,
        amplitude: float = 1.0,
        offset: float = 0.0,
        sample_rate: float = 10_000.0,
        chunk_samples: int = 1000,
        min_voltage: float = -10.0,
        max_voltage: float = 10.0,
    ) -> float:
        """Start continuous non-regenerative sine output.

        Purpose:
            Generate an exact requested sine frequency when ``sample_rate / frequency``
            is not an integer by using chunked non-regenerative streaming.

        Inputs:
            device: Device name (for example ``"Dev1"``).
            ao_channel: AO channel name (for example ``"ao0"``).
            frequency: Exact sine frequency in hertz (Hz). Must be > 0.
            amplitude: Sine peak amplitude in volts (V). Must be >= 0.
            offset: DC offset in volts (V).
            sample_rate: AO sample clock in samples/second (S/s). Must be > 0.
            chunk_samples: Number of samples written in first chunk. Must be >= 1.
            min_voltage: Lower output limit in volts (V).
            max_voltage: Upper output limit in volts (V).
        Output:
            Actual configured sample rate in samples/second (S/s).
        Raises:
            ValueError: Invalid inputs.
            nidaqmx.DaqError: DAQ configuration/start/write failure.

        Notes:
            Keep feeding data with ``write_sine_chunk_non_regen(...)`` while running.
        """

        if chunk_samples < 1:
            raise ValueError("chunk_samples must be >= 1.")

        validated_device, validated_channel = self._validate_sine_common(
            device=device,
            ao_channel=ao_channel,
            frequency=frequency,
            amplitude=amplitude,
            offset=offset,
            sample_rate=sample_rate,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
        )

        self.stop_output()

        physical_channel = f"{validated_device}/{validated_channel}"
        task = nidaqmx.Task()
        try:
            task.ao_channels.add_ao_voltage_chan(
                physical_channel,
                min_val=min_voltage,
                max_val=max_voltage,
            )
            task.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION
            task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
            )
            actual_sample_rate = float(task.timing.samp_clk_rate)
            first_chunk, next_phase = self._build_sine_chunk(
                frequency=frequency,
                amplitude=amplitude,
                offset=offset,
                sample_rate=actual_sample_rate,
                sample_count=chunk_samples,
                phase_in=0.0,
            )
            task.write(first_chunk, auto_start=False)
            task.start()
        except Exception:
            task.close()
            raise

        self._ao_task = task
        self._last_config = None
        self._non_regen_sine_active = True
        self._non_regen_phase = next_phase
        self._non_regen_frequency = frequency
        self._non_regen_amplitude = amplitude
        self._non_regen_offset = offset
        self._non_regen_sample_rate = actual_sample_rate
        self._non_regen_min_voltage = min_voltage
        self._non_regen_max_voltage = max_voltage
        return actual_sample_rate

    def write_sine_chunk_non_regen(
        self,
        *,
        chunk_samples: int = 1000,
    ) -> int:
        """Write one more sine chunk for active non-regenerative output.

        Purpose:
            Continue non-regenerative sine output with phase continuity.

        Inputs:
            chunk_samples: Number of new samples to generate and write. Must be >= 1.
        Output:
            Number of written samples.
        Raises:
            RuntimeError: Non-regenerative sine output is not active.
            ValueError: Invalid ``chunk_samples``.
            nidaqmx.DaqError: DAQ write failure.
        """

        if not self._non_regen_sine_active or self._ao_task is None:
            raise RuntimeError(
                "Non-regenerative sine output is not active. "
                "Call start_continuous_sine_output_non_regen() first."
            )
        if chunk_samples < 1:
            raise ValueError("chunk_samples must be >= 1.")

        chunk, next_phase = self._build_sine_chunk(
            frequency=self._non_regen_frequency,
            amplitude=self._non_regen_amplitude,
            offset=self._non_regen_offset,
            sample_rate=self._non_regen_sample_rate,
            sample_count=chunk_samples,
            phase_in=self._non_regen_phase,
        )
        written = self._ao_task.write(chunk, auto_start=False)
        self._non_regen_phase = next_phase
        return int(written)

    def stop_output(self) -> None:
        """Stop and release the active analog output task.

        Purpose:
            Safely stop AO task and clear non-regenerative streaming state.

        Inputs:
            None.
        Output:
            None. Safe to call when no task is running.
        """

        if self._ao_task is None:
            self._clear_non_regen_state()
            return

        task = self._ao_task
        self._ao_task = None
        self._clear_non_regen_state()
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
        input_mode: str = "default",
        terminal_config=None,
    ) -> float:
        """Start continuous analog input acquisition using the internal clock.

        Purpose:
            Start continuous AI capture for live monitoring or long recordings.

        Inputs:
            device: Device name (for example ``"Dev1"``).
            ai_channels: One channel name (``"ai0"``) or a sequence (``("ai0", "ai1")``).
            sample_rate: Requested AI sample clock in samples/second (S/s). Must be > 0
                and <= ``MAX_AI_SAMPLE_RATE``.
            min_voltage: Lower input limit in volts (V).
            max_voltage: Upper input limit in volts (V).
            input_mode: Simple AI wiring mode string. Allowed values:
                ``"default"``, ``"differential"``, ``"rse"``, ``"nrse"``,
                ``"pseudodifferential"``.
            terminal_config: Optional NI terminal configuration value to pass through to
                `add_ai_voltage_chan`. If provided, it overrides ``input_mode``.
        Output:
            Actual configured sample rate in samples/second (S/s).
        Raises:
            ValueError: Invalid inputs.
            nidaqmx.DaqError: DAQ configuration/start failed.

        Notes:
            Read blocks using ``read_input_chunk(...)`` after this call.
        """

        channels = self._normalize_ai_channels(ai_channels)
        resolved_terminal_config = self._resolve_terminal_config(
            input_mode=input_mode,
            terminal_config=terminal_config,
        )
        self._validate_input_limits(
            device=device,
            ai_channels=channels,
            sample_rate=sample_rate,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
            terminal_config=resolved_terminal_config,
        )

        self.stop_input()

        task = nidaqmx.Task()
        try:
            for ch in channels:
                physical_channel = ch if "/" in ch else f"{device}/{ch}"
                if resolved_terminal_config is None:
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
                        terminal_config=resolved_terminal_config,
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

    def measure_input_finite(
        self,
        *,
        samples_per_channel: int,
        sample_rate: float,
        device: str = "Dev1",
        ai_channels: str | Sequence[str] = ("ai0",),
        min_voltage: float = -10.0,
        max_voltage: float = 10.0,
        input_mode: str = "default",
        terminal_config=None,
        timeout: float = 10.0,
    ) -> np.ndarray:
        """Measure one finite AI block and return raw samples.

        Purpose:
            Acquire one fixed-size block from AI channels with shared settings.

        Inputs:
            samples_per_channel: Number of samples to acquire per channel. Must be >= 1.
            sample_rate: AI sample clock in samples/second (S/s). Must be > 0 and
                <= ``MAX_AI_SAMPLE_RATE``.
            device: Device name (for example ``"Dev1"``).
            ai_channels: One channel name or a sequence of channel names.
            min_voltage: Lower input limit in volts (V).
            max_voltage: Upper input limit in volts (V).
            input_mode: Simple AI wiring mode string. Allowed values:
                ``"default"``, ``"differential"``, ``"rse"``, ``"nrse"``,
                ``"pseudodifferential"``.
            terminal_config: Optional NI terminal configuration value to pass through to
                `add_ai_voltage_chan`. If provided, it overrides ``input_mode``.
            timeout: Read timeout in seconds.
        Output:
            ``numpy.ndarray`` with shape ``(channels, samples_per_channel)``.
        Raises:
            ValueError: Invalid inputs.
            nidaqmx.DaqError: DAQ configuration or read failed.
        """

        if samples_per_channel < 1:
            raise ValueError("samples_per_channel must be >= 1.")

        channels = self._normalize_ai_channels(ai_channels)
        resolved_terminal_config = self._resolve_terminal_config(
            input_mode=input_mode,
            terminal_config=terminal_config,
        )
        self._validate_input_limits(
            device=device,
            ai_channels=channels,
            sample_rate=sample_rate,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
            terminal_config=resolved_terminal_config,
        )

        task = nidaqmx.Task()
        try:
            for ch in channels:
                physical_channel = ch if "/" in ch else f"{device}/{ch}"
                if resolved_terminal_config is None:
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
                        terminal_config=resolved_terminal_config,
                    )

            task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=samples_per_channel,
            )
            task.start()
            raw = task.read(
                number_of_samples_per_channel=samples_per_channel,
                timeout=timeout,
            )
        finally:
            try:
                task.stop()
            except nidaqmx.DaqError:
                pass
            task.close()

        return self._reshape_read_data(raw=raw, channel_count=len(channels))

    def read_input_chunk(
        self,
        *,
        samples_per_channel: int = 1000,
        timeout: float = 10.0,
    ) -> np.ndarray:
        """Read one chunk from the running continuous AI task.

        Purpose:
            Fetch the next AI block from an active continuous input task.

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
        return self._reshape_read_data(raw=raw, channel_count=self._ai_channel_count)

    def stop_input(self) -> None:
        """Stop and release the active analog input task.

        Purpose:
            Safely stop and release continuous AI task resources.

        Inputs:
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
        input_mode: str = "default",
        ai_terminal_config=None,
    ) -> ContinuousSyncPeriodicConfig:
        """Start synchronized continuous periodic AO output and AI acquisition.

        Purpose:
            Run synchronized periodic AO stimulation and continuous AI capture
            using one shared sample clock.

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
            input_mode: Simple AI wiring mode string. Allowed values:
                ``"default"``, ``"differential"``, ``"rse"``, ``"nrse"``,
                ``"pseudodifferential"``.
            ai_terminal_config: Optional NI terminal config passed to AI channels.
                If provided, it overrides ``input_mode``.
        Output:
            ``ContinuousSyncPeriodicConfig`` containing requested and actual settings.
        Raises:
            ValueError: Invalid inputs or limits.
            nidaqmx.DaqError: DAQ configuration/start failure.

        Notes:
            NI trigger pattern is used: AO waits for AI start trigger.
        """

        channels = self._normalize_ai_channels(ai_channels)
        resolved_ai_terminal_config = self._resolve_terminal_config(
            input_mode=input_mode,
            terminal_config=ai_terminal_config,
        )
        self._validate_input_limits(
            device=device,
            ai_channels=channels,
            sample_rate=sample_rate,
            min_voltage=ai_min_voltage,
            max_voltage=ai_max_voltage,
            terminal_config=resolved_ai_terminal_config,
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
                if resolved_ai_terminal_config is None:
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
                        terminal_config=resolved_ai_terminal_config,
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

    def measure_sync_finite(
        self,
        *,
        output_samples: Sequence[float],
        sample_rate: float,
        device: str = "Dev1",
        ao_channel: str = "ao0",
        ai_channels: str | Sequence[str] = ("ai0",),
        ao_min_voltage: float = -10.0,
        ao_max_voltage: float = 10.0,
        ai_min_voltage: float = -10.0,
        ai_max_voltage: float = 10.0,
        input_mode: str = "default",
        ai_terminal_config=None,
        timeout: float = 10.0,
    ) -> np.ndarray:
        """Run one finite synchronized AO+AI measurement and return raw AI data.

        Purpose:
            Output one finite AO waveform and capture synchronized AI samples.

        Inputs:
            output_samples: Finite AO waveform to output once, in volts (V).
            sample_rate: Shared AO/AI sample clock in samples/second (S/s).
            device: Device name (for example ``"Dev1"``).
            ao_channel: AO channel name (for example ``"ao0"``).
            ai_channels: One AI channel name or a sequence of channel names.
            ao_min_voltage: AO lower voltage limit in volts (V).
            ao_max_voltage: AO upper voltage limit in volts (V).
            ai_min_voltage: AI lower voltage limit in volts (V).
            ai_max_voltage: AI upper voltage limit in volts (V).
            input_mode: Simple AI wiring mode string. Allowed values:
                ``"default"``, ``"differential"``, ``"rse"``, ``"nrse"``,
                ``"pseudodifferential"``.
            ai_terminal_config: Optional NI terminal config passed to AI channels.
                If provided, it overrides ``input_mode``.
            timeout: Read timeout in seconds.
        Output:
            ``numpy.ndarray`` with shape ``(channels, len(output_samples))``.
        Raises:
            ValueError: Invalid inputs.
            nidaqmx.DaqError: DAQ configuration/start/read failed.

        Notes:
            NI trigger pattern is used: AO waits for AI start trigger.
        """

        channels = self._normalize_ai_channels(ai_channels)
        resolved_ai_terminal_config = self._resolve_terminal_config(
            input_mode=input_mode,
            terminal_config=ai_terminal_config,
        )
        self._validate_input_limits(
            device=device,
            ai_channels=channels,
            sample_rate=sample_rate,
            min_voltage=ai_min_voltage,
            max_voltage=ai_max_voltage,
            terminal_config=resolved_ai_terminal_config,
        )
        if sample_rate > self.MAX_AO_SAMPLE_RATE:
            raise ValueError(
                "sample_rate exceeds USB-6451 AO limit: "
                f"{sample_rate:g} > {self.MAX_AO_SAMPLE_RATE:g} S/s."
            )
        if not device.strip():
            raise ValueError("device must not be empty.")
        if not ao_channel.strip():
            raise ValueError("ao_channel must not be empty.")
        if ao_min_voltage >= ao_max_voltage:
            raise ValueError("ao_min_voltage must be smaller than ao_max_voltage.")

        ao_data = np.asarray(output_samples, dtype=np.float64)
        if ao_data.ndim != 1:
            raise ValueError("output_samples must be a one-dimensional list/array.")
        if ao_data.size < 1:
            raise ValueError("output_samples must contain at least one sample.")
        if not np.all(np.isfinite(ao_data)):
            raise ValueError("output_samples must contain only finite numbers.")

        ao_high = float(np.max(ao_data))
        ao_low = float(np.min(ao_data))
        if ao_high > ao_max_voltage or ao_low < ao_min_voltage:
            raise ValueError(
                "Output waveform exceeds AO voltage limits: "
                f"[{ao_low:.3f}, {ao_high:.3f}] V is outside "
                f"[{ao_min_voltage:.3f}, {ao_max_voltage:.3f}] V."
            )

        sample_count = int(ao_data.size)
        ai_task = nidaqmx.Task()
        ao_task = nidaqmx.Task()
        try:
            for ch in channels:
                physical_channel = ch if "/" in ch else f"{device}/{ch}"
                if resolved_ai_terminal_config is None:
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
                        terminal_config=resolved_ai_terminal_config,
                    )

            ai_task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=sample_count,
            )

            ao_task.ao_channels.add_ao_voltage_chan(
                f"{device}/{ao_channel}",
                min_val=ao_min_voltage,
                max_val=ao_max_voltage,
            )
            ao_task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=sample_count,
            )

            # NI sync pattern: AO waits for AI start trigger terminal.
            ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                ai_task.triggers.start_trigger.term
            )
            ao_task.write(ao_data, auto_start=False)

            # NI start order: arm AO first, then start AI.
            ao_task.start()
            ai_task.start()
            raw = ai_task.read(
                number_of_samples_per_channel=sample_count,
                timeout=timeout,
            )
        finally:
            try:
                ai_task.stop()
            except nidaqmx.DaqError:
                pass
            try:
                ao_task.stop()
            except nidaqmx.DaqError:
                pass
            ai_task.close()
            ao_task.close()

        return self._reshape_read_data(raw=raw, channel_count=len(channels))

    def measure_sine_periods(
        self,
        *,
        periods: int,
        frequency: float = 10.0,
        amplitude: float = 1.0,
        offset: float = 0.0,
        sample_rate: float = 10_000.0,
        samples_per_period: Optional[int] = None,
        device: str = "Dev1",
        ao_channel: str = "ao0",
        ai_channels: str | Sequence[str] = ("ai0",),
        ao_min_voltage: float = -10.0,
        ao_max_voltage: float = 10.0,
        ai_min_voltage: float = -10.0,
        ai_max_voltage: float = 10.0,
        input_mode: str = "default",
        ai_terminal_config=None,
        timeout: float = 10.0,
    ) -> np.ndarray:
        """Output a sine on AO and measure a finite number of sine periods on AI.

        Purpose:
            Convenience method for sine-based synchronized measurements.

        Inputs:
            periods: Number of sine periods to output/measure. Must be >= 1.
            frequency: Requested sine frequency in hertz (Hz).
            amplitude: Sine peak amplitude in volts (V).
            offset: DC offset in volts (V).
            sample_rate: Shared AO/AI sample rate in samples/second (S/s).
            samples_per_period: Optional samples used for one sine period.
                If omitted, method selects:
                1) regenerative single-period replay when ``sample_rate / frequency``
                   is integer-like, or
                2) non-regenerative phase-continuous sample generation otherwise.
            device: Device name (for example ``"Dev1"``).
            ao_channel: AO channel name.
            ai_channels: One AI channel name or a sequence of names.
            ao_min_voltage: AO lower voltage limit in volts (V).
            ao_max_voltage: AO upper voltage limit in volts (V).
            ai_min_voltage: AI lower voltage limit in volts (V).
            ai_max_voltage: AI upper voltage limit in volts (V).
            input_mode: Simple AI wiring mode string. Allowed values:
                ``"default"``, ``"differential"``, ``"rse"``, ``"nrse"``,
                ``"pseudodifferential"``.
            ai_terminal_config: Optional NI terminal config passed to AI channels.
                If provided, it overrides ``input_mode``.
            timeout: Read timeout in seconds.
        Output:
            ``numpy.ndarray`` with shape ``(channels, N)``.
            ``N`` equals:
            1) ``periods * samples_per_period`` in regenerative path.
            2) ``round(periods * sample_rate / frequency)`` in non-regenerative path.
        Raises:
            ValueError: Invalid inputs.
            nidaqmx.DaqError: DAQ configuration/start/read failed.
        """

        if periods < 1:
            raise ValueError("periods must be >= 1.")

        validated_device, validated_channel = self._validate_sine_common(
            device=device,
            ao_channel=ao_channel,
            frequency=frequency,
            amplitude=amplitude,
            offset=offset,
            sample_rate=sample_rate,
            min_voltage=ao_min_voltage,
            max_voltage=ao_max_voltage,
        )

        if sample_rate / frequency < 8:
            raise ValueError(
                "frequency is too high for this sample_rate. "
                "Increase sample_rate or lower frequency."
            )

        if samples_per_period is not None:
            sine_config = self.get_continuous_sine_output_config(
                device=validated_device,
                ao_channel=validated_channel,
                frequency=frequency,
                amplitude=amplitude,
                offset=offset,
                sample_rate=sample_rate,
                samples_per_period=samples_per_period,
                min_voltage=ao_min_voltage,
                max_voltage=ao_max_voltage,
            )
            sine_period = waveforms.sine_period(
                amplitude=sine_config.amplitude,
                offset=sine_config.offset,
                samples_per_period=sine_config.samples_per_period,
                min_voltage=sine_config.min_voltage,
                max_voltage=sine_config.max_voltage,
                max_samples_per_period=self.MAX_REGENERATIVE_PERIOD_SAMPLES,
            )
            output_samples = np.tile(sine_period, periods)
            effective_sample_rate = sine_config.sample_rate
        else:
            ratio = sample_rate / frequency
            nearest = int(round(ratio))
            tolerance = max(1e-9, 1e-9 * ratio)

            if nearest >= 8 and abs(ratio - nearest) <= tolerance:
                # Integer-like divider: efficient regenerative one-period replay.
                sine_config = self.get_continuous_sine_output_config(
                    device=validated_device,
                    ao_channel=validated_channel,
                    frequency=frequency,
                    amplitude=amplitude,
                    offset=offset,
                    sample_rate=sample_rate,
                    samples_per_period=nearest,
                    min_voltage=ao_min_voltage,
                    max_voltage=ao_max_voltage,
                )
                sine_period = waveforms.sine_period(
                    amplitude=sine_config.amplitude,
                    offset=sine_config.offset,
                    samples_per_period=sine_config.samples_per_period,
                    min_voltage=sine_config.min_voltage,
                    max_voltage=sine_config.max_voltage,
                    max_samples_per_period=self.MAX_REGENERATIVE_PERIOD_SAMPLES,
                )
                output_samples = np.tile(sine_period, periods)
                effective_sample_rate = sine_config.sample_rate
            else:
                # Non-integer divider: generate phase-continuous AO samples.
                sample_count = int(round(periods * sample_rate / frequency))
                if sample_count < 1:
                    sample_count = 1
                output_samples, _ = self._build_sine_chunk(
                    frequency=frequency,
                    amplitude=amplitude,
                    offset=offset,
                    sample_rate=sample_rate,
                    sample_count=sample_count,
                    phase_in=0.0,
                )
                effective_sample_rate = sample_rate

        return self.measure_sync_finite(
            output_samples=output_samples,
            sample_rate=effective_sample_rate,
            device=validated_device,
            ao_channel=validated_channel,
            ai_channels=ai_channels,
            ao_min_voltage=ao_min_voltage,
            ao_max_voltage=ao_max_voltage,
            ai_min_voltage=ai_min_voltage,
            ai_max_voltage=ai_max_voltage,
            input_mode=input_mode,
            ai_terminal_config=ai_terminal_config,
            timeout=timeout,
        )

    def validate_sync_connection(
        self,
        *,
        device: str = "Dev1",
        ao_channel: str = "ao0",
        ai_channels: str | Sequence[str] = ("ai0", "ai7"),
        sample_rate: float = 20_000.0,
        samples_per_channel: int = 256,
        ao_test_voltage: float = 1.0,
        expected_current_channel_voltage_v: float | None = None,
        current_channel_tolerance_v: float = 0.01,
        current_channel_index: int = 0,
        settle_discard_s: float = 0.15,
        ao_min_voltage: float = -10.0,
        ao_max_voltage: float = 10.0,
        ai_min_voltage: float = -10.0,
        ai_max_voltage: float = 10.0,
        input_mode: str = "differential",
        ai_terminal_config=None,
        timeout: float = 10.0,
    ) -> SyncConnectionValidationResult:
        """Run a short synchronized AO+AI self-check before a measurement sweep.

        Purpose:
            Verify that synchronized AO+AI operation is functioning before
            starting a frequency sweep.

        Inputs:
            device: Device name (for example ``"Dev1"``).
            ao_channel: AO channel used for the preflight test.
            ai_channels: One AI channel name or a sequence of AI channels.
            sample_rate: Shared AO/AI sample clock in samples/second (S/s).
            samples_per_channel: Samples acquired per AI channel. Must be >= 1.
            ao_test_voltage: Constant AO level in volts (V) used for the test.
            expected_current_channel_voltage_v: Expected mean shunt-voltage level
                on the configured current channel after settling discard. If
                omitted, only synchronized shape check is performed.
            current_channel_tolerance_v: Allowed error around expected shunt
                voltage when expectation is provided.
            current_channel_index: Index of current/shunt channel in ``ai_channels``.
            settle_discard_s: Initial capture time in seconds discarded before
                checking measured means against expected shunt voltage.
            ao_min_voltage: AO lower limit in volts (V).
            ao_max_voltage: AO upper limit in volts (V).
            ai_min_voltage: AI lower limit in volts (V).
            ai_max_voltage: AI upper limit in volts (V).
            input_mode: Simple AI wiring mode string.
            ai_terminal_config: Optional native NI terminal config override.
            timeout: Read timeout in seconds.
        Output:
            ``SyncConnectionValidationResult`` with measured shape and summary.
        Raises:
            ValueError: Invalid input values.
            RuntimeError: Returned sample shape does not match requested shape.
            nidaqmx.DaqError: DAQ configuration/start/read failure.

        Notes:
            This method checks DAQ communication and synchronized timing path.
        """

        if samples_per_channel < 1:
            raise ValueError("samples_per_channel must be >= 1.")
        if settle_discard_s < 0:
            raise ValueError("settle_discard_s must be >= 0.")
        if current_channel_tolerance_v <= 0:
            raise ValueError("current_channel_tolerance_v must be > 0.")
        if current_channel_index < 0:
            raise ValueError("current_channel_index must be >= 0.")
        if ao_min_voltage >= ao_max_voltage:
            raise ValueError("ao_min_voltage must be smaller than ao_max_voltage.")
        if ao_test_voltage < ao_min_voltage or ao_test_voltage > ao_max_voltage:
            raise ValueError(
                "ao_test_voltage must be inside AO limits: "
                f"[{ao_min_voltage:.3f}, {ao_max_voltage:.3f}] V."
            )

        settle_discard_samples = int(round(settle_discard_s * sample_rate))
        if settle_discard_samples >= samples_per_channel:
            raise ValueError(
                "settle_discard_s is too large for samples_per_channel at chosen sample_rate. "
                f"discard_samples={settle_discard_samples}, samples_per_channel={samples_per_channel}."
            )

        channels = self._normalize_ai_channels(ai_channels)
        if current_channel_index >= len(channels):
            raise ValueError(
                "current_channel_index is outside ai_channels range: "
                f"index={current_channel_index}, channel_count={len(channels)}."
            )
        test_output = np.full(samples_per_channel, ao_test_voltage, dtype=np.float64)
        ai_data = self.measure_sync_finite(
            output_samples=test_output,
            sample_rate=sample_rate,
            device=device,
            ao_channel=ao_channel,
            ai_channels=channels,
            ao_min_voltage=ao_min_voltage,
            ao_max_voltage=ao_max_voltage,
            ai_min_voltage=ai_min_voltage,
            ai_max_voltage=ai_max_voltage,
            input_mode=input_mode,
            ai_terminal_config=ai_terminal_config,
            timeout=timeout,
        )

        expected_shape = (len(channels), samples_per_channel)
        measured_shape = (int(ai_data.shape[0]), int(ai_data.shape[1]))
        if measured_shape != expected_shape:
            raise RuntimeError(
                "Synchronized connection check returned unexpected data shape: "
                f"expected {expected_shape}, got {measured_shape}."
            )

        usable = ai_data[:, settle_discard_samples:]
        if usable.shape[1] < 1:
            raise RuntimeError(
                "No usable preflight samples after settling discard. "
                "Increase samples_per_channel or reduce settle_discard_s."
            )
        current_channel_mean_v = float(np.mean(usable[current_channel_index]))
        if expected_current_channel_voltage_v is not None:
            lower_bound = float(
                expected_current_channel_voltage_v - current_channel_tolerance_v
            )
            upper_bound = float(
                expected_current_channel_voltage_v + current_channel_tolerance_v
            )
            if not (lower_bound <= current_channel_mean_v <= upper_bound):
                raise RuntimeError(
                    "Synchronized AO+AI preflight failed current-channel shunt-voltage check: "
                    f"expected={expected_current_channel_voltage_v:.6g} V, "
                    f"tolerance=+/-{current_channel_tolerance_v:.6g} V, "
                    f"measured={current_channel_mean_v:.6g} V, "
                    f"channel_index={current_channel_index}."
                )
            status_text = (
                f"expected_shunt={expected_current_channel_voltage_v:.6g} V, "
                f"tolerance=+/-{current_channel_tolerance_v:.6g} V, "
                f"measured={current_channel_mean_v:.6g} V"
            )
        else:
            status_text = (
                f"no_shunt_expectation_provided, measured_current_channel={current_channel_mean_v:.6g} V"
            )

        return SyncConnectionValidationResult(
            device=device.strip(),
            ao_channel=ao_channel.strip(),
            ai_channels=channels,
            sample_rate=float(sample_rate),
            samples_per_channel=samples_per_channel,
            measured_shape=measured_shape,
            message=(
                "Synchronized AO+AI preflight PASS "
                f"for {device.strip()} with {len(channels)} AI channel(s): "
                f"ao_dc={ao_test_voltage:.6g} V, discard={settle_discard_s:.6g} s, "
                f"current_channel_index={current_channel_index}, {status_text}."
            ),
        )

    def read_sync_input_chunk(
        self,
        *,
        samples_per_channel: int = 1000,
        timeout: float = 10.0,
    ) -> np.ndarray:
        """Read one chunk from the synchronized AI task.

        Purpose:
            Fetch one AI block while synchronized continuous IO is running.

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
        return self._reshape_read_data(raw=raw, channel_count=self._sync_ai_channel_count)

    def stop_sync_io(self) -> None:
        """Stop and release active synchronized AO+AI tasks.

        Purpose:
            Stop and release both synchronized tasks in a safe order.

        Inputs:
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

        Purpose:
            Report whether synchronized AO and AI tasks are both active.

        Inputs:
            None.
        Output:
            ``True`` when both sync tasks are active, otherwise ``False``.
        """

        return self._sync_ai_task is not None and self._sync_ao_task is not None

    def is_input_running(self) -> bool:
        """Return input running state.

        Purpose:
            Report whether continuous AI task is currently active.

        Inputs:
            None.
        Output:
            ``True`` when this object has an active AI task, otherwise ``False``.
        """

        return self._ai_task is not None

    def is_output_running(self) -> bool:
        """Return output running state.

        Purpose:
            Report whether continuous AO task is currently active.

        Inputs:
            None.
        Output:
            ``True`` when this object has an active AO task, otherwise ``False``.
        """

        return self._ao_task is not None

    def close(self) -> None:
        """Release all DAQ resources held by this object.

        Purpose:
            Final cleanup helper that stops output, input, and synchronized tasks.

        Inputs:
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

        Purpose:
            Replay a custom one-period waveform continuously on AO.

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

        Purpose:
            Internal helper for exact sine configuration checks.

        Inputs:
            Same as ``start_continuous_sine_output`` except ``allow_regen``.
        Output:
            ``ContinuousSineConfig`` with exact generated frequency.
        Raises:
            ValueError: Invalid input values.
        """
        device, ao_channel = self._validate_sine_common(
            device=device,
            ao_channel=ao_channel,
            frequency=frequency,
            amplitude=amplitude,
            offset=offset,
            sample_rate=sample_rate,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
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
            device=device,
            ao_channel=ao_channel,
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

        Purpose:
            Internal helper for validating custom periodic AO waveforms.

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

    def _validate_sine_common(
        self,
        *,
        device: str,
        ao_channel: str,
        frequency: float,
        amplitude: float,
        offset: float,
        sample_rate: float,
        min_voltage: float,
        max_voltage: float,
    ) -> tuple[str, str]:
        """Validate common sine parameters and normalize device/channel names.

        Purpose:
            Internal guard for sine-based APIs before task configuration.

        Inputs:
            device: NI device name.
            ao_channel: AO channel name.
            frequency: Requested sine frequency in hertz (Hz).
            amplitude: Sine amplitude in volts (V).
            offset: Sine offset in volts (V).
            sample_rate: AO sample rate in samples/second (S/s).
            min_voltage: AO lower limit in volts (V).
            max_voltage: AO upper limit in volts (V).
        Output:
            Tuple ``(device, ao_channel)`` with trimmed names.
        Raises:
            ValueError: Any invalid value or out-of-limit waveform request.
        """

        normalized_device = device.strip()
        normalized_channel = ao_channel.strip()
        if not normalized_device:
            raise ValueError("device must not be empty.")
        if not normalized_channel:
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
        return normalized_device, normalized_channel

    @staticmethod
    def _build_sine_chunk(
        *,
        frequency: float,
        amplitude: float,
        offset: float,
        sample_rate: float,
        sample_count: int,
        phase_in: float,
    ) -> tuple[np.ndarray, float]:
        """Generate one phase-continuous sine chunk.

        Purpose:
            Internal chunk generator for non-regenerative streaming paths.

        Inputs:
            frequency: Sine frequency in hertz (Hz).
            amplitude: Sine amplitude in volts (V).
            offset: Sine offset in volts (V).
            sample_rate: Sample rate in samples/second (S/s).
            sample_count: Number of samples to generate.
            phase_in: Start phase in radians.
        Output:
            Tuple ``(samples, phase_out)`` where ``phase_out`` is used for the next chunk.
        """

        omega = 2.0 * np.pi * frequency / sample_rate
        phases = phase_in + omega * np.arange(sample_count, dtype=np.float64)
        data = offset + amplitude * np.sin(phases)
        phase_out = float((phase_in + omega * sample_count) % (2.0 * np.pi))
        return data.astype(np.float64, copy=False), phase_out

    @staticmethod
    def _normalize_ai_channels(ai_channels: str | Sequence[str]) -> tuple[str, ...]:
        """Normalize AI channel input.

        Purpose:
            Convert user channel input into a clean non-empty tuple.

        Inputs:
            ai_channels: One channel string or sequence of channel strings.
        Output:
            Tuple of normalized channel names.
        Raises:
            ValueError: No valid channel names remain after normalization.
        """
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
        terminal_config,
    ) -> None:
        """Validate AI limits against USB-6451 constraints.

        Purpose:
            Internal safety guard for AI channel count, rate, and voltage limits.

        Inputs:
            device: NI device name.
            ai_channels: Tuple of AI channels.
            sample_rate: AI sample rate in samples/second (S/s).
            min_voltage: AI lower limit in volts (V).
            max_voltage: AI upper limit in volts (V).
            terminal_config: NI terminal configuration (or ``None``).
        Output:
            None.
        Raises:
            ValueError: Any requested setting exceeds USB-6451 limits.
        """
        if not device.strip():
            raise ValueError("device must not be empty.")
        if len(ai_channels) > self.MAX_AI_CHANNELS:
            raise ValueError(
                f"Too many ai_channels: {len(ai_channels)} exceeds {self.MAX_AI_CHANNELS}."
            )
        if (
            terminal_config is not None
            and terminal_config == TerminalConfiguration.DIFFERENTIAL
            and len(ai_channels) > self.MAX_AI_DIFF_CHANNELS
        ):
            raise ValueError(
                "Too many ai_channels for differential mode: "
                f"{len(ai_channels)} exceeds {self.MAX_AI_DIFF_CHANNELS}."
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

    @staticmethod
    def _reshape_read_data(raw, channel_count: int) -> np.ndarray:
        """Convert DAQmx read output to deterministic array shape.

        Purpose:
            Normalize NI read return types to ``(channels, samples)`` arrays.

        Inputs:
            raw: Raw value returned by DAQmx ``read``.
            channel_count: Expected number of channels.
        Output:
            ``numpy.ndarray`` shaped as ``(channels, samples)``.
        """
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            if channel_count <= 1:
                return arr.reshape(1, -1)
            return arr.reshape(-1, 1)
        return arr

    def _clear_non_regen_state(self) -> None:
        """Reset non-regenerative sine tracking state.

        Purpose:
            Return internal chunk/phase state to defaults after stop/cleanup.

        Inputs:
            None.
        Output:
            None.
        """

        self._non_regen_sine_active = False
        self._non_regen_phase = 0.0
        self._non_regen_frequency = 0.0
        self._non_regen_amplitude = 0.0
        self._non_regen_offset = 0.0
        self._non_regen_sample_rate = 0.0
        self._non_regen_min_voltage = -10.0
        self._non_regen_max_voltage = 10.0

    @staticmethod
    def _resolve_terminal_config(*, input_mode: str, terminal_config):
        """Map a simple input mode string to NI terminal configuration.

        Purpose:
            Translate user-friendly AI wiring mode names to NI constants.

        Inputs:
            input_mode: One of ``"default"``, ``"differential"``, ``"rse"``,
                ``"nrse"``, ``"pseudodifferential"``.
            terminal_config: Optional native NI terminal configuration object.
        Output:
            Terminal configuration object accepted by DAQmx, or ``None`` for default.
            If ``terminal_config`` is provided, it is returned unchanged.
        Raises:
            ValueError: Unknown input_mode.
        """
        if terminal_config is not None:
            return terminal_config

        mode = str(input_mode).strip().lower()
        if mode in ("", "default"):
            return None

        mapping = {
            "differential": TerminalConfiguration.DIFFERENTIAL,
            "diff": TerminalConfiguration.DIFFERENTIAL,
            "rse": TerminalConfiguration.RSE,
            "nrse": TerminalConfiguration.NRSE,
            "pseudodifferential": TerminalConfiguration.PSEUDODIFFERENTIAL,
            "pseudo": TerminalConfiguration.PSEUDODIFFERENTIAL,
        }
        if mode not in mapping:
            raise ValueError(
                "input_mode must be one of: "
                "default, differential, rse, nrse, pseudodifferential."
            )
        return mapping[mode]
