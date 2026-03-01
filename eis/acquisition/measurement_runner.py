"""Single-point acquisition runner for Phase 1 synchronized measurements."""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter

import numpy as np

from eis.acquisition.usb6451_adapter import USB6451Adapter
from eis.models.config_models import MeasurementPointConfig
from eis.models.measurement_models import ExcitationConfig, HardwareConfig, MeasurementCapture


def run_measurement_point(
    *,
    adapter: USB6451Adapter,
    point: MeasurementPointConfig,
    hardware: HardwareConfig,
    excitation: ExcitationConfig,
    repeat_index: int,
    samples_per_period: int | None = None,
) -> MeasurementCapture:
    """Acquire one synchronized raw capture for one frequency point.

    Inputs:
        adapter: USB6451 adapter instance.
        point: One validated sweep row.
        hardware: Hardware wiring and limits configuration.
        excitation: Sine output settings for this run.
        repeat_index: Repeat number (1-based) for the same sweep point.
        samples_per_period: Optional forced samples/period for sine generation.
    Output:
        ``MeasurementCapture`` containing timestamp, settings, and raw AI matrix.
    Raises:
        RuntimeError: Returned raw matrix shape is not compatible with channel count.
    """

    if repeat_index < 1:
        raise ValueError("repeat_index must be >= 1.")

    started_at = datetime.now(timezone.utc)
    tic = perf_counter()

    # USB6451 currently uses shared AI limits across channels; choose the safer
    # channel range (larger absolute value) from config row.
    ai_range_v = float(max(point.ch0_range_v, point.ch1_range_v))

    raw_data = adapter.measure_sine_point(
        hardware=hardware,
        frequency_hz=point.frequency_hz,
        sample_rate_sps=point.sample_rate_sps,
        n_periods=point.n_periods,
        amplitude_v=excitation.amplitude_v,
        offset_v=excitation.offset_v,
        ai_range_v=ai_range_v,
        samples_per_period=samples_per_period,
    )
    duration_s = perf_counter() - tic

    if raw_data.ndim != 2:
        raise RuntimeError(
            f"Expected 2D raw data array with shape (channels, samples), got ndim={raw_data.ndim}."
        )
    if raw_data.shape[0] != len(hardware.ai_channels):
        raise RuntimeError(
            "Raw data channel count mismatch: "
            f"expected {len(hardware.ai_channels)}, got {raw_data.shape[0]}."
        )

    return MeasurementCapture(
        row_number=point.row_number,
        repeat_index=repeat_index,
        frequency_hz=point.frequency_hz,
        sample_rate_sps=point.sample_rate_sps,
        n_periods=point.n_periods,
        current_rms=point.current_rms,
        started_at_utc_iso=started_at.isoformat(),
        duration_s=float(duration_s),
        ai_channels=hardware.ai_channels,
        ai_range_v=ai_range_v,
        raw_data=np.asarray(raw_data, dtype=np.float64),
    )
