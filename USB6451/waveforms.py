"""Helpers for building one-period analog output waveforms.

These functions generate one full waveform period as voltage samples. They also
apply NI USB-6451-oriented limits by default:
- AO voltage range: -10 V to +10 V
- Max regenerative period length: 16,383 samples
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_AO_MIN_VOLTAGE = -10.0
DEFAULT_AO_MAX_VOLTAGE = 10.0
DEFAULT_MAX_PERIOD_SAMPLES = 16_383


def _validate_common(
    *,
    samples_per_period: int,
    min_voltage: float,
    max_voltage: float,
    max_samples_per_period: int,
) -> None:
    """Validate common waveform constraints."""
    if samples_per_period < 1:
        raise ValueError("samples_per_period must be >= 1.")
    if max_samples_per_period < 1:
        raise ValueError("max_samples_per_period must be >= 1.")
    if samples_per_period > max_samples_per_period:
        raise ValueError(
            "samples_per_period exceeds max allowed period length: "
            f"{samples_per_period} > {max_samples_per_period}."
        )
    if min_voltage >= max_voltage:
        raise ValueError("min_voltage must be smaller than max_voltage.")


def _validate_waveform_values(
    *,
    data: np.ndarray,
    min_voltage: float,
    max_voltage: float,
) -> None:
    """Validate finite values and output voltage range."""
    if data.ndim != 1:
        raise ValueError("Waveform must be one-dimensional.")
    if data.size < 1:
        raise ValueError("Waveform must contain at least one sample.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Waveform must contain only finite numbers.")

    high = float(np.max(data))
    low = float(np.min(data))
    if high > max_voltage or low < min_voltage:
        raise ValueError(
            "Output waveform exceeds voltage limits: "
            f"[{low:.3f}, {high:.3f}] V is outside "
            f"[{min_voltage:.3f}, {max_voltage:.3f}] V."
        )


def sine_period(
    *,
    amplitude: float,
    offset: float,
    samples_per_period: int,
    min_voltage: float = DEFAULT_AO_MIN_VOLTAGE,
    max_voltage: float = DEFAULT_AO_MAX_VOLTAGE,
    max_samples_per_period: int = DEFAULT_MAX_PERIOD_SAMPLES,
) -> np.ndarray:
    """Generate one sine period.

    Inputs:
        amplitude: Sine peak amplitude in volts (V). Must be >= 0.
        offset: DC offset in volts (V).
        samples_per_period: Number of samples in one period. Must be >= 8.
        min_voltage: Lower voltage limit in volts (V).
        max_voltage: Upper voltage limit in volts (V).
        max_samples_per_period: Maximum allowed period length.
    Output:
        ``numpy.ndarray`` with shape ``(samples_per_period,)`` and ``float64`` values.
    """
    if amplitude < 0:
        raise ValueError("amplitude must be >= 0.")
    if samples_per_period < 8:
        raise ValueError("samples_per_period must be >= 8 for sine output.")

    _validate_common(
        samples_per_period=samples_per_period,
        min_voltage=min_voltage,
        max_voltage=max_voltage,
        max_samples_per_period=max_samples_per_period,
    )

    phase = np.linspace(0.0, 2.0 * np.pi, samples_per_period, endpoint=False)
    data = offset + amplitude * np.sin(phase)
    _validate_waveform_values(data=data, min_voltage=min_voltage, max_voltage=max_voltage)
    return data


def ramp_period(
    *,
    start: float,
    stop: float,
    samples_per_period: int,
    include_endpoint: bool = False,
    min_voltage: float = DEFAULT_AO_MIN_VOLTAGE,
    max_voltage: float = DEFAULT_AO_MAX_VOLTAGE,
    max_samples_per_period: int = DEFAULT_MAX_PERIOD_SAMPLES,
) -> np.ndarray:
    """Generate one ramp period using linear interpolation.

    Inputs:
        start: First sample value in volts (V).
        stop: Last target value in volts (V). Included only if ``include_endpoint=True``.
        samples_per_period: Number of samples in one period. Must be >= 2.
        include_endpoint: If ``False``, last sample is below ``stop`` for cleaner periodic wrap.
        min_voltage: Lower voltage limit in volts (V).
        max_voltage: Upper voltage limit in volts (V).
        max_samples_per_period: Maximum allowed period length.
    Output:
        ``numpy.ndarray`` with shape ``(samples_per_period,)`` and ``float64`` values.
    """
    if samples_per_period < 2:
        raise ValueError("samples_per_period must be >= 2 for ramp output.")

    _validate_common(
        samples_per_period=samples_per_period,
        min_voltage=min_voltage,
        max_voltage=max_voltage,
        max_samples_per_period=max_samples_per_period,
    )

    data = np.linspace(start, stop, samples_per_period, endpoint=include_endpoint)
    _validate_waveform_values(data=data, min_voltage=min_voltage, max_voltage=max_voltage)
    return data


def staircase_period(
    *,
    levels: Sequence[float],
    samples_per_level: int = 1,
    min_voltage: float = DEFAULT_AO_MIN_VOLTAGE,
    max_voltage: float = DEFAULT_AO_MAX_VOLTAGE,
    max_samples_per_period: int = DEFAULT_MAX_PERIOD_SAMPLES,
) -> np.ndarray:
    """Generate one staircase period from level values.

    Inputs:
        levels: Sequence of staircase levels in volts (V), in output order.
        samples_per_level: Number of repeated samples for each level. Must be >= 1.
        min_voltage: Lower voltage limit in volts (V).
        max_voltage: Upper voltage limit in volts (V).
        max_samples_per_period: Maximum allowed period length.
    Output:
        ``numpy.ndarray`` with shape ``(len(levels) * samples_per_level,)`` in ``float64``.
    """
    if samples_per_level < 1:
        raise ValueError("samples_per_level must be >= 1.")

    level_data = np.asarray(levels, dtype=np.float64)
    if level_data.ndim != 1 or level_data.size < 1:
        raise ValueError("levels must be a one-dimensional sequence with at least one value.")

    samples_per_period = int(level_data.size * samples_per_level)
    _validate_common(
        samples_per_period=samples_per_period,
        min_voltage=min_voltage,
        max_voltage=max_voltage,
        max_samples_per_period=max_samples_per_period,
    )

    data = np.repeat(level_data, samples_per_level)
    _validate_waveform_values(data=data, min_voltage=min_voltage, max_voltage=max_voltage)
    return data


def triangle_period(
    *,
    amplitude: float,
    offset: float,
    samples_per_period: int,
    symmetry: float = 0.5,
    min_voltage: float = DEFAULT_AO_MIN_VOLTAGE,
    max_voltage: float = DEFAULT_AO_MAX_VOLTAGE,
    max_samples_per_period: int = DEFAULT_MAX_PERIOD_SAMPLES,
) -> np.ndarray:
    """Generate one triangle period.

    Inputs:
        amplitude: Triangle peak amplitude in volts (V). Must be >= 0.
        offset: DC offset in volts (V).
        samples_per_period: Number of samples in one period. Must be >= 3.
        symmetry: Fraction of period used for rising edge. Must satisfy ``0 < symmetry < 1``.
        min_voltage: Lower voltage limit in volts (V).
        max_voltage: Upper voltage limit in volts (V).
        max_samples_per_period: Maximum allowed period length.
    Output:
        ``numpy.ndarray`` with shape ``(samples_per_period,)`` and ``float64`` values.
    """
    if amplitude < 0:
        raise ValueError("amplitude must be >= 0.")
    if samples_per_period < 3:
        raise ValueError("samples_per_period must be >= 3 for triangle output.")
    if symmetry <= 0.0 or symmetry >= 1.0:
        raise ValueError("symmetry must satisfy 0 < symmetry < 1.")

    _validate_common(
        samples_per_period=samples_per_period,
        min_voltage=min_voltage,
        max_voltage=max_voltage,
        max_samples_per_period=max_samples_per_period,
    )

    rise_count = int(np.floor(samples_per_period * symmetry))
    rise_count = max(1, min(rise_count, samples_per_period - 1))
    fall_count = samples_per_period - rise_count

    rise = np.linspace(-1.0, 1.0, rise_count, endpoint=False)
    fall = np.linspace(1.0, -1.0, fall_count, endpoint=False)
    data = offset + amplitude * np.concatenate([rise, fall])
    _validate_waveform_values(data=data, min_voltage=min_voltage, max_voltage=max_voltage)
    return data


def square_period(
    *,
    amplitude: float,
    offset: float,
    samples_per_period: int,
    duty: float = 0.5,
    min_voltage: float = DEFAULT_AO_MIN_VOLTAGE,
    max_voltage: float = DEFAULT_AO_MAX_VOLTAGE,
    max_samples_per_period: int = DEFAULT_MAX_PERIOD_SAMPLES,
) -> np.ndarray:
    """Generate one square-wave period.

    Inputs:
        amplitude: Square-wave peak amplitude in volts (V). Must be >= 0.
        offset: DC offset in volts (V).
        samples_per_period: Number of samples in one period. Must be >= 2.
        duty: High-level duty cycle fraction. Must satisfy ``0 < duty < 1``.
        min_voltage: Lower voltage limit in volts (V).
        max_voltage: Upper voltage limit in volts (V).
        max_samples_per_period: Maximum allowed period length.
    Output:
        ``numpy.ndarray`` with shape ``(samples_per_period,)`` and ``float64`` values.
    """
    if amplitude < 0:
        raise ValueError("amplitude must be >= 0.")
    if samples_per_period < 2:
        raise ValueError("samples_per_period must be >= 2 for square output.")
    if duty <= 0.0 or duty >= 1.0:
        raise ValueError("duty must satisfy 0 < duty < 1.")

    _validate_common(
        samples_per_period=samples_per_period,
        min_voltage=min_voltage,
        max_voltage=max_voltage,
        max_samples_per_period=max_samples_per_period,
    )

    high_samples = int(round(samples_per_period * duty))
    high_samples = max(1, min(high_samples, samples_per_period - 1))

    data = np.empty(samples_per_period, dtype=np.float64)
    data[:high_samples] = offset + amplitude
    data[high_samples:] = offset - amplitude
    _validate_waveform_values(data=data, min_voltage=min_voltage, max_voltage=max_voltage)
    return data
