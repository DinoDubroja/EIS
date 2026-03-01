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
    preflight_samples_per_channel: int = 256,
    preflight_ao_test_voltage: float = 0.0,
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
        preflight_samples_per_channel: Preflight sample count per AI channel.
        preflight_ao_test_voltage: AO test level during preflight in volts (V).
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
        preflight_result = run_preflight_check(
            adapter=adapter,
            hardware=hardware,
            sample_rate_sps=chosen_preflight_rate,
            samples_per_channel=preflight_samples_per_channel,
            ao_test_voltage=preflight_ao_test_voltage,
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
