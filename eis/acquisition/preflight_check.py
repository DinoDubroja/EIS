"""DAQ preflight check helpers for Phase 1 sweep execution."""

from __future__ import annotations

from eis.acquisition.usb6451_adapter import USB6451Adapter
from eis.models.measurement_models import HardwareConfig, PreflightCheckResult


def run_preflight_check(
    *,
    adapter: USB6451Adapter,
    hardware: HardwareConfig,
    sample_rate_sps: float,
    samples_per_channel: int = 256,
    ao_test_voltage: float = 0.0,
) -> PreflightCheckResult:
    """Run synchronized DAQ connectivity preflight before a measurement sweep.

    Inputs:
        adapter: USB6451 adapter instance.
        hardware: Hardware wiring/limits configuration.
        sample_rate_sps: Sample rate used for the check in samples/second (S/s).
        samples_per_channel: Number of AI samples captured per channel.
        ao_test_voltage: Constant AO level in volts (V) during the check.
    Output:
        ``PreflightCheckResult`` summary for logs/UI.
    """

    return adapter.run_preflight_check(
        hardware=hardware,
        sample_rate_sps=sample_rate_sps,
        samples_per_channel=samples_per_channel,
        ao_test_voltage=ao_test_voltage,
    )
