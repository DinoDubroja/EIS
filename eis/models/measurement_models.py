"""Data models for Phase 1 acquisition orchestration."""

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
    """Sine stimulus settings used for one or more measurements."""

    amplitude_v: float
    offset_v: float = 0.0


@dataclass(frozen=True)
class PreflightCheckResult:
    """Result summary from DAQ synchronized connection preflight check."""

    sample_rate_sps: float
    samples_per_channel: int
    measured_shape: tuple[int, int]
    message: str


@dataclass(frozen=True)
class MeasurementCapture:
    """Raw capture from one frequency and one repeat in a sweep."""

    row_number: int
    repeat_index: int
    frequency_hz: float
    sample_rate_sps: float
    n_periods: int
    current_rms: float
    started_at_utc_iso: str
    duration_s: float
    ai_channels: tuple[str, ...]
    ai_range_v: float
    raw_data: np.ndarray


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
