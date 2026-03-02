"""Typed data models for acquisition orchestration and metadata capture.

These dataclasses define stable contracts between:
- acquisition execution
- storage/report generation
- future processing/plotting layers

Keeping these models explicit helps technicians and engineers inspect run
records directly and supports robust test coverage for each pipeline stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware wiring and limit settings for USB-6451 acquisition.

    Inputs:
        device: NI MAX device name, for example ``"Dev1"``.
        ao_channel: AO channel name without device prefix, for example ``"ao0"``.
        ai_channels: Ordered AI channels without device prefix.
        input_mode: AI terminal mode string accepted by USB6451 API.
        ao_min_voltage: AO lower voltage limit in volts (V).
        ao_max_voltage: AO upper voltage limit in volts (V).
        ai_default_min_voltage: Default AI lower limit in volts (V).
        ai_default_max_voltage: Default AI upper limit in volts (V).
        timeout_s: DAQ read timeout in seconds.
    Output:
        Immutable hardware configuration object.
    """

    device: str = "Dev1"
    ao_channel: str = "ao0"
    ai_channels: tuple[str, ...] = ("ai0", "ai7")
    input_mode: str = "differential"
    ao_min_voltage: float = -10.0
    ao_max_voltage: float = 10.0
    ai_default_min_voltage: float = -10.0
    ai_default_max_voltage: float = 10.0
    timeout_s: float = 10.0


@dataclass(frozen=True)
class ExcitationConfig:
    """Sine stimulus settings used for one or more measurements.

    Inputs:
        drive_mode: Excitation mode.
            - ``"auto_from_current_rms"``: compute AO amplitude from
              ``Current_rms`` using Clarke-Hess transconductance ranges.
            - ``"fixed_ao_amplitude"``: use ``amplitude_v`` directly.
        amplitude_v: AO sine peak amplitude in volts (V), used in fixed mode.
        offset_v: AO sine offset in volts (V).
        manual_current_range: Optional fixed range name for auto mode
            (for example ``"20A"``). If omitted, range is selected automatically.
        range_selection_policy: Auto-selection policy. Currently:
            - ``"prefer_no_overrange"``
    Output:
        Immutable excitation settings object.
    """

    drive_mode: str = "auto_from_current_rms"
    amplitude_v: float = 0.2
    offset_v: float = 0.0
    manual_current_range: str | None = None
    range_selection_policy: str = "prefer_no_overrange"


@dataclass(frozen=True)
class TransconductanceRange:
    """One Clarke-Hess 8100 current range definition.

    Inputs:
        name: Range label shown to users, for example ``"20A"``.
        transconductance_siemens: Range transconductance in Siemens (A/V).
        min_current_rms_a: Lower supported current in amperes RMS (A).
        full_scale_current_rms_a: Nominal full-scale current in amperes RMS (A).
        max_current_rms_a: Maximum supported current in amperes RMS (A).
        input_full_scale_vrms: Input voltage in volts RMS (V) at full-scale current.
    Output:
        Immutable current range definition.
    """

    name: str
    transconductance_siemens: float
    min_current_rms_a: float
    full_scale_current_rms_a: float
    max_current_rms_a: float
    input_full_scale_vrms: float


@dataclass(frozen=True)
class DriveAmplitudeResult:
    """Computed AO drive settings derived from target current RMS."""

    range_name: str
    transconductance_siemens: float
    current_rms_a: float
    ao_input_vrms: float
    ao_amplitude_v_peak: float
    is_overrange: bool


@dataclass(frozen=True)
class PreflightCheckResult:
    """Result summary from DAQ synchronized connection preflight check."""

    sample_rate_sps: float
    samples_per_channel: int
    measured_shape: tuple[int, int]
    message: str


@dataclass(frozen=True)
class CaptureConditioningConfig:
    """Acquisition conditioning settings for settling and leakage handling.

    Inputs:
        settle_discard_s: Time in seconds discarded from measurement start to
            remove AO startup transients before analysis.
        extra_periods_for_trim: Additional whole periods to acquire beyond
            requested ``N_periods``. These samples provide margin for selecting a
            periodic window with minimal edge discontinuity.
        alignment_search_periods: Number of periods searched (after settling
            discard) to find best periodic window start index.
    Output:
        Immutable conditioning configuration used by acquisition runner.
    Notes:
        When ``settle_discard_s`` is zero and ``extra_periods_for_trim`` is zero,
        behavior matches previous direct-capture flow.
    """

    settle_discard_s: float = 0.02
    extra_periods_for_trim: int = 1
    alignment_search_periods: int = 1


@dataclass(frozen=True)
class MeasurementCapture:
    """Raw capture from one frequency and one repeat in a sweep."""

    row_number: int
    repeat_index: int
    frequency_hz: float
    sample_rate_sps: float
    n_periods: int
    current_rms: float
    ao_amplitude_v_peak: float
    ao_offset_v: float
    current_range_name: str | None
    transconductance_siemens: float | None
    started_at_utc_iso: str
    duration_s: float
    ai_channels: tuple[str, ...]
    ai_range_v: float
    raw_data: np.ndarray
    acquired_periods: int = 0
    discarded_settle_samples: int = 0
    periodic_window_start_sample: int = 0
    periodic_window_samples: int = 0


@dataclass(frozen=True)
class ImpedancePointResult:
    """One impedance result for one frequency point and one repeat.

    Inputs:
        row_number: Config row index (1-based) used for traceability.
        repeat_index: Repeat number (1-based) within the same frequency point.
        frequency_hz: Frequency in hertz (Hz) for this result.
        z_real_ohm: Real impedance component in ohms (Ohm).
        z_imag_ohm: Imaginary impedance component in ohms (Ohm).
        z_magnitude_ohm: Impedance magnitude in ohms (Ohm).
        z_phase_deg: Impedance phase in degrees (deg).
        extraction_method: Method label used to compute impedance.
            Examples: ``"fft"``, ``"sine_fit"``, ``"demo_placeholder"``.
        snr_current_db: Estimated SNR in dB for current channel (shunt voltage).
        snr_voltage_db: Estimated SNR in dB for voltage channel (DUT voltage).
        notes: Optional free text with additional context.
    Output:
        Immutable impedance result record for storage and statistics.
    """

    row_number: int
    repeat_index: int
    frequency_hz: float
    z_real_ohm: float
    z_imag_ohm: float
    z_magnitude_ohm: float
    z_phase_deg: float
    extraction_method: str
    snr_current_db: float | None = None
    snr_voltage_db: float | None = None
    notes: str | None = None


@dataclass(frozen=True)
class SweepProgress:
    """Progress update payload for UI progress bars and logs."""

    total_steps: int
    completed_steps: int
    row_number: int
    frequency_hz: float
    repeat_index: int


@dataclass(frozen=True)
class SweepRunResult:
    """Full sweep run output before processing/analysis stages."""

    started_at_utc_iso: str
    finished_at_utc_iso: str
    repeats: int
    captures: tuple[MeasurementCapture, ...]
    preflight: PreflightCheckResult | None
