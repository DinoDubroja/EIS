"""USB-6451 adapter bridging EIS orchestration and low-level DAQ API.

The adapter isolates direct calls to `USB6451` so higher-level sweep logic can
be tested without NI drivers/hardware. It maps EIS-level inputs (point config,
hardware config, preflight settings) to the low-level method signatures.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from eis.models.measurement_models import HardwareConfig, PreflightCheckResult


class Usb6451Like(Protocol):
    """Protocol describing USB-6451 methods required by acquisition APIs.

    Purpose:
        Enable dependency injection for tests and demo backends without NI
        hardware. Any object implementing these methods can be used by
        ``USB6451Adapter`` and higher-level sweep logic.
    """

    def validate_sync_connection(
        self,
        *,
        device: str,
        ao_channel: str,
        ai_channels: tuple[str, ...],
        sample_rate: float,
        samples_per_channel: int,
        ao_test_voltage: float,
        expected_current_channel_voltage_v: float | None,
        current_channel_tolerance_v: float,
        current_channel_index: int,
        settle_discard_s: float,
        ao_min_voltage: float,
        ao_max_voltage: float,
        ai_min_voltage: float,
        ai_max_voltage: float,
        input_mode: str,
        ai_terminal_config,
        timeout: float,
    ):
        """Run synchronized preflight check."""

    def measure_sine_periods(
        self,
        *,
        periods: int,
        frequency: float,
        amplitude: float,
        offset: float,
        sample_rate: float,
        samples_per_period,
        device: str,
        ao_channel: str,
        ai_channels: tuple[str, ...],
        ao_min_voltage: float,
        ao_max_voltage: float,
        ai_min_voltage: float,
        ai_max_voltage: float,
        input_mode: str,
        ai_terminal_config,
        timeout: float,
    ) -> np.ndarray:
        """Run synchronized finite sine measurement."""

    def close(self) -> None:
        """Release underlying DAQ tasks."""


class USB6451Adapter:
    """Adapter mapping EIS acquisition models to low-level USB6451 signatures.

    This class keeps orchestration code independent from direct NI driver calls
    and normalizes low-level return types into EIS dataclasses.
    """

    def __init__(self, controller: Usb6451Like | None = None) -> None:
        """Create adapter with an optional injected USB6451 controller instance."""

        if controller is None:
            from USB6451.USB6451 import USB6451

            self._controller = USB6451()
            self._owns_controller = True
        else:
            self._controller = controller
            self._owns_controller = False

    def run_preflight_check(
        self,
        *,
        hardware: HardwareConfig,
        sample_rate_sps: float,
        samples_per_channel: int = 256,
        ao_test_voltage: float = 1.0,
        expected_current_channel_voltage_v: float | None = None,
        current_channel_tolerance_v: float = 0.01,
        current_channel_index: int = 0,
        settle_discard_s: float = 0.15,
    ) -> PreflightCheckResult:
        """Run USB6451 synchronized connection preflight and normalize result model."""

        result = self._controller.validate_sync_connection(
            device=hardware.device,
            ao_channel=hardware.ao_channel,
            ai_channels=hardware.ai_channels,
            sample_rate=sample_rate_sps,
            samples_per_channel=samples_per_channel,
            ao_test_voltage=ao_test_voltage,
            expected_current_channel_voltage_v=expected_current_channel_voltage_v,
            current_channel_tolerance_v=current_channel_tolerance_v,
            current_channel_index=current_channel_index,
            settle_discard_s=settle_discard_s,
            ao_min_voltage=hardware.ao_min_voltage,
            ao_max_voltage=hardware.ao_max_voltage,
            ai_min_voltage=hardware.ai_default_min_voltage,
            ai_max_voltage=hardware.ai_default_max_voltage,
            input_mode=hardware.input_mode,
            ai_terminal_config=None,
            timeout=hardware.timeout_s,
        )
        return PreflightCheckResult(
            sample_rate_sps=float(result.sample_rate),
            samples_per_channel=int(result.samples_per_channel),
            measured_shape=tuple(int(v) for v in result.measured_shape),
            message=str(result.message),
        )

    def measure_sine_point(
        self,
        *,
        hardware: HardwareConfig,
        frequency_hz: float,
        sample_rate_sps: float,
        n_periods: int,
        amplitude_v: float,
        offset_v: float,
        ai_range_v: float,
        samples_per_period: int | None = None,
    ) -> np.ndarray:
        """Run one synchronized sine measurement point and return raw AI matrix."""

        return self._controller.measure_sine_periods(
            periods=n_periods,
            frequency=frequency_hz,
            amplitude=amplitude_v,
            offset=offset_v,
            sample_rate=sample_rate_sps,
            samples_per_period=samples_per_period,
            device=hardware.device,
            ao_channel=hardware.ao_channel,
            ai_channels=hardware.ai_channels,
            ao_min_voltage=hardware.ao_min_voltage,
            ao_max_voltage=hardware.ao_max_voltage,
            ai_min_voltage=-abs(ai_range_v),
            ai_max_voltage=abs(ai_range_v),
            input_mode=hardware.input_mode,
            ai_terminal_config=None,
            timeout=hardware.timeout_s,
        )

    def close(self) -> None:
        """Close owned USB6451 controller instance."""

        if self._owns_controller:
            self._controller.close()
