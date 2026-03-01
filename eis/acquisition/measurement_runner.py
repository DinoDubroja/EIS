"""Single-point synchronized acquisition runner.

This module converts one validated sweep row into one DAQ capture operation.
It resolves excitation mode (automatic current-based drive or fixed amplitude),
applies AO limit guards, acquires extra data for conditioning, trims settling
transients, chooses a low-discontinuity periodic window, calls the USB6451
adapter, and returns a typed capture record with timing and metadata needed by
later storage/reporting layers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter

import numpy as np

from eis.acquisition.transconductance import compute_drive_amplitude_from_current
from eis.acquisition.usb6451_adapter import USB6451Adapter
from eis.models.config_models import MeasurementPointConfig
from eis.models.measurement_models import (
    CaptureConditioningConfig,
    ExcitationConfig,
    HardwareConfig,
    MeasurementCapture,
)


def _validate_conditioning(
    *,
    conditioning: CaptureConditioningConfig,
    frequency_hz: float,
) -> None:
    """Validate conditioning inputs for one measurement point."""

    if conditioning.settle_discard_s < 0:
        raise ValueError("conditioning.settle_discard_s must be >= 0.")
    if conditioning.extra_periods_for_trim < 0:
        raise ValueError("conditioning.extra_periods_for_trim must be >= 0.")
    if conditioning.alignment_search_periods < 0:
        raise ValueError("conditioning.alignment_search_periods must be >= 0.")
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be > 0.")


def _choose_periodic_window_start(
    *,
    data: np.ndarray,
    target_samples: int,
    max_start: int,
) -> int:
    """Choose start index minimizing first-vs-last sample discontinuity."""

    if target_samples < 1:
        raise ValueError("target_samples must be >= 1.")
    if data.shape[1] < target_samples:
        raise ValueError("data does not contain enough samples for target window.")
    if max_start <= 0:
        return 0

    best_start = 0
    best_score = float("inf")
    for start in range(max_start + 1):
        end_index = start + target_samples - 1
        edge_delta = data[:, start] - data[:, end_index]
        score = float(np.linalg.norm(edge_delta))
        if score < best_score:
            best_score = score
            best_start = start
    return best_start


def run_measurement_point(
    *,
    adapter: USB6451Adapter,
    point: MeasurementPointConfig,
    hardware: HardwareConfig,
    excitation: ExcitationConfig,
    conditioning: CaptureConditioningConfig | None = None,
    repeat_index: int,
    samples_per_period: int | None = None,
) -> MeasurementCapture:
    """Acquire one synchronized raw capture for one frequency point.

    Inputs:
        adapter: USB6451 adapter instance.
        point: One validated sweep row.
        hardware: Hardware wiring and limits configuration.
        excitation: Sine output settings for this run.
        conditioning: Settling discard and periodic trim strategy.
        repeat_index: Repeat number (1-based) for the same sweep point.
        samples_per_period: Optional forced samples/period for sine generation.
    Output:
        ``MeasurementCapture`` containing timestamp, settings, and raw AI matrix.
    Raises:
        RuntimeError: Returned raw matrix shape is not compatible with channel count.
    """

    if repeat_index < 1:
        raise ValueError("repeat_index must be >= 1.")

    effective_conditioning = conditioning or CaptureConditioningConfig()
    _validate_conditioning(
        conditioning=effective_conditioning,
        frequency_hz=float(point.frequency_hz),
    )

    started_at = datetime.now(timezone.utc)
    tic = perf_counter()

    # USB6451 currently uses shared AI limits across channels; choose the safer
    # channel range (larger absolute value) from config row.
    ai_range_v = float(max(point.ch0_range_v, point.ch1_range_v))
    current_rms_a = float(point.current_rms)

    drive_mode = excitation.drive_mode.strip().lower()
    if drive_mode == "auto_from_current_rms":
        drive = compute_drive_amplitude_from_current(
            current_rms_a=current_rms_a,
            manual_range_name=excitation.manual_current_range,
            selection_policy=excitation.range_selection_policy,
        )
        ao_amplitude_v_peak = float(drive.ao_amplitude_v_peak)
        selected_range_name = drive.range_name
        selected_transconductance = float(drive.transconductance_siemens)
    elif drive_mode == "fixed_ao_amplitude":
        if excitation.amplitude_v <= 0:
            raise ValueError("Excitation amplitude_v must be > 0 in fixed_ao_amplitude mode.")
        ao_amplitude_v_peak = float(excitation.amplitude_v)
        selected_range_name = None
        selected_transconductance = None
    else:
        raise ValueError(
            "Unsupported excitation drive_mode. "
            "Use 'auto_from_current_rms' or 'fixed_ao_amplitude'."
        )

    ao_high = float(excitation.offset_v + ao_amplitude_v_peak)
    ao_low = float(excitation.offset_v - ao_amplitude_v_peak)
    if ao_high > hardware.ao_max_voltage or ao_low < hardware.ao_min_voltage:
        raise ValueError(
            "Computed AO waveform exceeds hardware AO limits: "
            f"[{ao_low:.6g}, {ao_high:.6g}] V is outside "
            f"[{hardware.ao_min_voltage:.6g}, {hardware.ao_max_voltage:.6g}] V."
        )

    settle_periods = int(np.ceil(effective_conditioning.settle_discard_s * point.frequency_hz))
    acquired_periods = int(
        point.n_periods + effective_conditioning.extra_periods_for_trim + settle_periods
    )
    if acquired_periods < point.n_periods:
        raise RuntimeError("Internal error: acquired_periods is smaller than requested periods.")

    raw_data_full = adapter.measure_sine_point(
        hardware=hardware,
        frequency_hz=point.frequency_hz,
        sample_rate_sps=point.sample_rate_sps,
        n_periods=acquired_periods,
        amplitude_v=ao_amplitude_v_peak,
        offset_v=excitation.offset_v,
        ai_range_v=ai_range_v,
        samples_per_period=samples_per_period,
    )
    duration_s = perf_counter() - tic

    if raw_data_full.ndim != 2:
        raise RuntimeError(
            "Expected 2D raw data array with shape (channels, samples), "
            f"got ndim={raw_data_full.ndim}."
        )
    if raw_data_full.shape[0] != len(hardware.ai_channels):
        raise RuntimeError(
            "Raw data channel count mismatch: "
            f"expected {len(hardware.ai_channels)}, got {raw_data_full.shape[0]}."
        )

    total_samples = int(raw_data_full.shape[1])
    target_samples = int(round(total_samples * (point.n_periods / acquired_periods)))
    target_samples = max(1, target_samples)

    discarded_settle_samples = int(round(total_samples * (settle_periods / acquired_periods)))
    discarded_settle_samples = max(0, min(discarded_settle_samples, total_samples - 1))

    post_settle = raw_data_full[:, discarded_settle_samples:]
    if post_settle.shape[1] < target_samples:
        raise RuntimeError(
            "Insufficient samples after settling discard. "
            f"Need {target_samples}, available {post_settle.shape[1]}."
        )

    estimated_samples_per_period = int(round(total_samples / acquired_periods))
    estimated_samples_per_period = max(1, estimated_samples_per_period)
    search_samples = int(
        round(effective_conditioning.alignment_search_periods * estimated_samples_per_period)
    )
    max_start = int(min(max(0, search_samples), post_settle.shape[1] - target_samples))
    periodic_start = _choose_periodic_window_start(
        data=post_settle,
        target_samples=target_samples,
        max_start=max_start,
    )
    raw_data = post_settle[:, periodic_start : periodic_start + target_samples]

    return MeasurementCapture(
        row_number=point.row_number,
        repeat_index=repeat_index,
        frequency_hz=point.frequency_hz,
        sample_rate_sps=point.sample_rate_sps,
        n_periods=point.n_periods,
        current_rms=current_rms_a,
        ao_amplitude_v_peak=ao_amplitude_v_peak,
        ao_offset_v=float(excitation.offset_v),
        current_range_name=selected_range_name,
        transconductance_siemens=selected_transconductance,
        started_at_utc_iso=started_at.isoformat(),
        duration_s=float(duration_s),
        ai_channels=hardware.ai_channels,
        ai_range_v=ai_range_v,
        raw_data=np.asarray(raw_data, dtype=np.float64),
        acquired_periods=acquired_periods,
        discarded_settle_samples=discarded_settle_samples,
        periodic_window_start_sample=discarded_settle_samples + periodic_start,
        periodic_window_samples=int(raw_data.shape[1]),
    )
