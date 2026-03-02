"""Sweep orchestration with repeat loops and progress callbacks.

Responsibilities:
- optional DAQ preflight check
- deterministic frequency/repeat execution order
- progress event emission for UI progress bars
- returning a complete in-memory sweep result for downstream storage/analysis
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from eis.acquisition.measurement_runner import run_measurement_point
from eis.acquisition.preflight_check import run_preflight_check
from eis.acquisition.usb6451_adapter import USB6451Adapter
from eis.models.config_models import SweepConfig
from eis.models.measurement_models import (
    CaptureConditioningConfig,
    ExcitationConfig,
    HardwareConfig,
    MeasurementCapture,
    PreflightCheckResult,
    SweepProgress,
    SweepRunResult,
)

ProgressCallback = Callable[[SweepProgress], None]


def execute_sweep(
    *,
    sweep: SweepConfig,
    adapter: USB6451Adapter,
    hardware: HardwareConfig,
    excitation: ExcitationConfig,
    repeats: int = 1,
    run_preflight: bool = True,
    preflight_sample_rate_sps: float | None = None,
    preflight_samples_per_channel: int | None = None,
    preflight_test_current_rms_a: float = 10.0,
    preflight_manual_current_range: str | None = None,
    preflight_shunt_resistance_ohm: float = 0.008,
    preflight_shunt_voltage_tolerance_percent: float = 15.0,
    preflight_current_channel_index: int = 0,
    preflight_settle_discard_s: float = 0.15,
    conditioning: CaptureConditioningConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SweepRunResult:
    """Execute full measurement sweep with repeats in synchronized mode.

    Inputs:
        sweep: Validated frequency sweep configuration.
        adapter: USB6451 adapter instance.
        hardware: Hardware wiring and limits.
        excitation: Sine stimulus settings.
        repeats: How many times each frequency point is repeated.
        run_preflight: Whether to run DAQ preflight check before sweep.
        preflight_sample_rate_sps: Optional sample rate override for preflight.
        preflight_samples_per_channel: Optional preflight sample count per AI
            channel. If omitted, it is sized automatically from preflight
            sample rate and settling discard time.
        preflight_test_current_rms_a: Current target used to build preflight AO DC.
        preflight_manual_current_range: Optional fixed Clarke-Hess range for
            preflight current-to-voltage conversion.
        preflight_shunt_resistance_ohm: Nominal shunt value used for expectation.
        preflight_shunt_voltage_tolerance_percent: Allowed shunt-voltage error
            as percent of expected shunt voltage.
        preflight_current_channel_index: AI channel index used as current channel.
        preflight_settle_discard_s: Time discarded from preflight capture start.
        conditioning: Settling discard and periodic trim strategy per capture.
        progress_callback: Optional callback for progress updates.
    Output:
        ``SweepRunResult`` with preflight summary and all captures.
    Raises:
        ValueError: Invalid repeat count or empty sweep.
    """

    if repeats < 1:
        raise ValueError("repeats must be >= 1.")
    if not sweep.points:
        raise ValueError("Sweep configuration contains no points.")

    started = datetime.now(timezone.utc)
    captures: list[MeasurementCapture] = []
    preflight_result: PreflightCheckResult | None = None

    if run_preflight:
        chosen_preflight_rate = (
            float(preflight_sample_rate_sps)
            if preflight_sample_rate_sps is not None
            else float(sweep.points[0].sample_rate_sps)
        )
        if preflight_settle_discard_s < 0:
            raise ValueError("preflight_settle_discard_s must be >= 0.")
        required_settle_samples = int(round(preflight_settle_discard_s * chosen_preflight_rate))
        auto_analysis_samples = max(64, int(round(0.02 * chosen_preflight_rate)))
        chosen_preflight_samples = (
            int(preflight_samples_per_channel)
            if preflight_samples_per_channel is not None
            else (required_settle_samples + auto_analysis_samples)
        )
        if chosen_preflight_samples <= required_settle_samples:
            raise ValueError(
                "preflight_samples_per_channel is too small for requested preflight_settle_discard_s."
            )
        effective_preflight_range = (
            preflight_manual_current_range
            if preflight_manual_current_range is not None
            else excitation.manual_current_range
        )
        preflight_result = run_preflight_check(
            adapter=adapter,
            hardware=hardware,
            sample_rate_sps=chosen_preflight_rate,
            samples_per_channel=chosen_preflight_samples,
            test_current_rms_a=preflight_test_current_rms_a,
            manual_current_range=effective_preflight_range,
            range_selection_policy=excitation.range_selection_policy,
            shunt_resistance_ohm=preflight_shunt_resistance_ohm,
            shunt_voltage_tolerance_percent=preflight_shunt_voltage_tolerance_percent,
            current_channel_index=preflight_current_channel_index,
            settle_discard_s=preflight_settle_discard_s,
        )

    total_steps = len(sweep.points) * repeats
    completed_steps = 0

    for point in sweep.points:
        for repeat_index in range(1, repeats + 1):
            capture = run_measurement_point(
                adapter=adapter,
                point=point,
                hardware=hardware,
                excitation=excitation,
                conditioning=conditioning,
                repeat_index=repeat_index,
                samples_per_period=None,
            )
            captures.append(capture)
            completed_steps += 1

            if progress_callback is not None:
                progress_callback(
                    SweepProgress(
                        total_steps=total_steps,
                        completed_steps=completed_steps,
                        row_number=point.row_number,
                        frequency_hz=point.frequency_hz,
                        repeat_index=repeat_index,
                    )
                )

    finished = datetime.now(timezone.utc)
    return SweepRunResult(
        started_at_utc_iso=started.isoformat(),
        finished_at_utc_iso=finished.isoformat(),
        repeats=repeats,
        captures=tuple(captures),
        preflight=preflight_result,
    )
