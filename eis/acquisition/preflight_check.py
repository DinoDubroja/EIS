"""Preflight connectivity checks before running measurement sweeps.

This module wraps the synchronized AO+AI validation call so sweep orchestration
can perform a standard hardware readiness check before data acquisition starts.
The preflight result is preserved in run metadata for traceability.
"""

from __future__ import annotations

from eis.acquisition.transconductance import compute_drive_amplitude_from_current
from eis.acquisition.usb6451_adapter import USB6451Adapter
from eis.models.measurement_models import HardwareConfig, PreflightCheckResult


def run_preflight_check(
    *,
    adapter: USB6451Adapter,
    hardware: HardwareConfig,
    sample_rate_sps: float,
    samples_per_channel: int = 256,
    test_current_rms_a: float = 10.0,
    manual_current_range: str | None = None,
    range_selection_policy: str = "prefer_no_overrange",
    shunt_resistance_ohm: float = 0.008,
    shunt_voltage_tolerance_percent: float = 15.0,
    current_channel_index: int = 0,
    settle_discard_s: float = 0.15,
) -> PreflightCheckResult:
    """Run synchronized DAQ preflight using current-target and shunt expectation.

    Inputs:
        adapter: USB6451 adapter instance.
        hardware: Hardware wiring/limits configuration.
        sample_rate_sps: Sample rate used for the check in samples/second (S/s).
        samples_per_channel: Number of AI samples captured per channel.
        test_current_rms_a: Target current for preflight in amperes RMS (A).
        manual_current_range: Optional fixed transconductance range name.
        range_selection_policy: Auto-selection policy when range is not fixed.
        shunt_resistance_ohm: Nominal shunt resistance in ohms (Ohm).
        shunt_voltage_tolerance_percent: Allowed relative error around expected
            shunt voltage, expressed in percent (%).
        current_channel_index: AI index used for shunt voltage channel validation.
        settle_discard_s: Settling interval discarded from start of captured data.
    Output:
        ``PreflightCheckResult`` summary for logs/UI.
    """

    if sample_rate_sps <= 0:
        raise ValueError("sample_rate_sps must be > 0.")
    if samples_per_channel < 1:
        raise ValueError("samples_per_channel must be >= 1.")
    if test_current_rms_a <= 0:
        raise ValueError("test_current_rms_a must be > 0 A.")
    if shunt_resistance_ohm <= 0:
        raise ValueError("shunt_resistance_ohm must be > 0.")
    if shunt_voltage_tolerance_percent <= 0:
        raise ValueError("shunt_voltage_tolerance_percent must be > 0.")
    if current_channel_index < 0:
        raise ValueError("current_channel_index must be >= 0.")

    drive = compute_drive_amplitude_from_current(
        current_rms_a=test_current_rms_a,
        manual_range_name=manual_current_range,
        selection_policy=range_selection_policy,
    )
    ao_test_voltage = float(drive.ao_input_vrms)
    expected_shunt_voltage_v = float(test_current_rms_a * shunt_resistance_ohm)
    shunt_voltage_tolerance_v = float(
        abs(expected_shunt_voltage_v) * (shunt_voltage_tolerance_percent / 100.0)
    )

    base_result = adapter.run_preflight_check(
        hardware=hardware,
        sample_rate_sps=sample_rate_sps,
        samples_per_channel=samples_per_channel,
        ao_test_voltage=ao_test_voltage,
        expected_current_channel_voltage_v=expected_shunt_voltage_v,
        current_channel_tolerance_v=shunt_voltage_tolerance_v,
        current_channel_index=current_channel_index,
        settle_discard_s=settle_discard_s,
    )
    return PreflightCheckResult(
        sample_rate_sps=base_result.sample_rate_sps,
        samples_per_channel=base_result.samples_per_channel,
        measured_shape=base_result.measured_shape,
        message=(
            f"{base_result.message} | range={drive.range_name}, "
            f"test_current={test_current_rms_a:.6g} A, "
            f"expected_shunt={expected_shunt_voltage_v:.6g} V, "
            f"tolerance=+/-{shunt_voltage_tolerance_percent:.6g}% "
            f"(+/-{shunt_voltage_tolerance_v:.6g} V)"
        ),
    )
