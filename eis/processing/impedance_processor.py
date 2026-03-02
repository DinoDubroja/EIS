"""Impedance extraction pipeline from synchronized raw captures.

This module converts stored acquisition captures into per-repeat impedance rows.
Supported extraction strategies:
- ``fft``: single-bin complex projection at commanded frequency
- ``sine_fit`` with backend choice:
  - ``numpy_lstsq`` (always available)
  - ``scipy_least_squares`` (optional, if SciPy is installed)

Optional pre-filtering is available in frequency domain:
- ``none``
- ``lowpass``
- ``bandpass``

Nominal shunt conversion:
- Channel mapped as current channel is interpreted as shunt voltage.
- Complex current is computed as ``I = V_shunt / R_shunt_nominal``.
- Impedance is computed as ``Z = V_dut / I``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from eis.models.measurement_models import ImpedancePointResult, MeasurementCapture, SweepRunResult


@dataclass(frozen=True)
class ImpedanceProcessingConfig:
    """Settings controlling filtering and extraction for impedance computation.

    Inputs:
        method: Main extraction method:
            - ``"fft"``
            - ``"sine_fit"``
        sine_fit_backend: Backend when method is ``"sine_fit"``:
            - ``"numpy_lstsq"``
            - ``"scipy_least_squares"``
        filter_mode: Optional pre-filter mode applied to each channel:
            - ``"none"``
            - ``"lowpass"``
            - ``"bandpass"``
        lowpass_cutoff_hz: Lowpass cutoff in hertz when lowpass is selected.
            If omitted, default is ``2.5 * frequency_hz``.
        bandpass_low_hz: Bandpass low edge in hertz.
            If omitted, default is ``0.5 * frequency_hz``.
        bandpass_high_hz: Bandpass high edge in hertz.
            If omitted, default is ``1.5 * frequency_hz``.
        shunt_resistance_ohm: Nominal shunt resistance in ohms used for
            current conversion. Default is 8 milliohms.
        current_channel_index: Raw capture channel index for shunt voltage.
        voltage_channel_index: Raw capture channel index for DUT voltage.
        remove_dc_before_extraction: If true, subtract per-channel mean before
            filtering and extraction.
    Output:
        Immutable processing configuration.
    """

    method: str = "fft"
    sine_fit_backend: str = "numpy_lstsq"
    filter_mode: str = "none"
    lowpass_cutoff_hz: float | None = None
    bandpass_low_hz: float | None = None
    bandpass_high_hz: float | None = None
    shunt_resistance_ohm: float = 0.008
    current_channel_index: int = 0
    voltage_channel_index: int = 1
    remove_dc_before_extraction: bool = True


def _normalize_text(value: str) -> str:
    """Normalize case/whitespace for option parsing."""

    return value.strip().lower()


def _validate_processing_config(config: ImpedanceProcessingConfig) -> None:
    """Validate processing config values."""

    method = _normalize_text(config.method)
    if method not in {"fft", "sine_fit"}:
        raise ValueError("config.method must be 'fft' or 'sine_fit'.")

    backend = _normalize_text(config.sine_fit_backend)
    if backend not in {"numpy_lstsq", "scipy_least_squares"}:
        raise ValueError(
            "config.sine_fit_backend must be 'numpy_lstsq' or 'scipy_least_squares'."
        )

    filter_mode = _normalize_text(config.filter_mode)
    if filter_mode not in {"none", "lowpass", "bandpass"}:
        raise ValueError("config.filter_mode must be 'none', 'lowpass', or 'bandpass'.")

    if config.shunt_resistance_ohm <= 0:
        raise ValueError("config.shunt_resistance_ohm must be > 0.")
    if config.current_channel_index < 0:
        raise ValueError("config.current_channel_index must be >= 0.")
    if config.voltage_channel_index < 0:
        raise ValueError("config.voltage_channel_index must be >= 0.")


def _apply_frequency_domain_filter(
    *,
    signal: np.ndarray,
    sample_rate_sps: float,
    frequency_hz: float,
    config: ImpedanceProcessingConfig,
) -> np.ndarray:
    """Apply optional frequency-domain lowpass or bandpass filter."""

    mode = _normalize_text(config.filter_mode)
    if mode == "none":
        return signal

    if sample_rate_sps <= 0:
        raise ValueError("sample_rate_sps must be > 0.")
    nyquist = 0.5 * sample_rate_sps
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate_sps)
    spectrum = np.fft.rfft(signal)

    if mode == "lowpass":
        cutoff_hz = config.lowpass_cutoff_hz
        if cutoff_hz is None:
            cutoff_hz = 2.5 * frequency_hz
        if cutoff_hz <= 0 or cutoff_hz >= nyquist:
            raise ValueError(
                "Lowpass cutoff must be > 0 and below Nyquist frequency."
            )
        mask = freqs <= cutoff_hz
    else:
        low_hz = config.bandpass_low_hz
        high_hz = config.bandpass_high_hz
        if low_hz is None:
            low_hz = 0.5 * frequency_hz
        if high_hz is None:
            high_hz = 1.5 * frequency_hz
        if low_hz <= 0 or high_hz <= 0 or low_hz >= high_hz or high_hz >= nyquist:
            raise ValueError(
                "Bandpass limits must satisfy 0 < low < high < Nyquist."
            )
        mask = (freqs >= low_hz) & (freqs <= high_hz)

    filtered_spectrum = spectrum * mask
    return np.fft.irfft(filtered_spectrum, n=signal.size).astype(np.float64, copy=False)


def _extract_phasor_fft(
    *,
    signal: np.ndarray,
    sample_rate_sps: float,
    frequency_hz: float,
) -> complex:
    """Extract sine phasor using single-frequency complex projection."""

    sample_indices = np.arange(signal.size, dtype=np.float64)
    angle = -2.0 * np.pi * frequency_hz * sample_indices / sample_rate_sps
    exp_term = np.exp(1j * angle)
    return complex((2.0 / signal.size) * np.sum(signal * exp_term))


def _extract_phasor_sine_fit_numpy(
    *,
    signal: np.ndarray,
    sample_rate_sps: float,
    frequency_hz: float,
) -> complex:
    """Extract sine phasor with linear least-squares using numpy."""

    t = np.arange(signal.size, dtype=np.float64) / sample_rate_sps
    omega = 2.0 * np.pi * frequency_hz
    design = np.column_stack(
        (
            np.sin(omega * t),
            np.cos(omega * t),
            np.ones_like(t),
        )
    )
    coeffs, _, _, _ = np.linalg.lstsq(design, signal, rcond=None)
    sine_coeff = float(coeffs[0])
    cosine_coeff = float(coeffs[1])
    return complex(sine_coeff, cosine_coeff)


def _extract_phasor_sine_fit_scipy(
    *,
    signal: np.ndarray,
    sample_rate_sps: float,
    frequency_hz: float,
) -> complex:
    """Extract sine phasor with non-linear least-squares using SciPy."""

    try:
        from scipy import optimize  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - only runs when SciPy missing
        raise ImportError(
            "SciPy is required for sine_fit backend 'scipy_least_squares'."
        ) from exc

    t = np.arange(signal.size, dtype=np.float64) / sample_rate_sps
    omega = 2.0 * np.pi * frequency_hz

    initial_linear = _extract_phasor_sine_fit_numpy(
        signal=signal,
        sample_rate_sps=sample_rate_sps,
        frequency_hz=frequency_hz,
    )
    amplitude_0 = float(abs(initial_linear))
    phase_0 = float(np.angle(initial_linear))
    offset_0 = float(np.mean(signal))

    def residuals(params: np.ndarray) -> np.ndarray:
        amplitude, phase, offset = params
        model = amplitude * np.sin(omega * t + phase) + offset
        return model - signal

    result = optimize.least_squares(
        residuals,
        x0=np.array([amplitude_0, phase_0, offset_0], dtype=np.float64),
    )
    amplitude = float(result.x[0])
    phase = float(result.x[1])

    if amplitude < 0:
        amplitude = -amplitude
        phase += math.pi

    return complex(amplitude * math.cos(phase), amplitude * math.sin(phase))


def _extract_phasor(
    *,
    signal: np.ndarray,
    sample_rate_sps: float,
    frequency_hz: float,
    config: ImpedanceProcessingConfig,
) -> complex:
    """Dispatch extraction method and return channel phasor."""

    method = _normalize_text(config.method)
    if method == "fft":
        return _extract_phasor_fft(
            signal=signal,
            sample_rate_sps=sample_rate_sps,
            frequency_hz=frequency_hz,
        )

    backend = _normalize_text(config.sine_fit_backend)
    if backend == "numpy_lstsq":
        return _extract_phasor_sine_fit_numpy(
            signal=signal,
            sample_rate_sps=sample_rate_sps,
            frequency_hz=frequency_hz,
        )

    return _extract_phasor_sine_fit_scipy(
        signal=signal,
        sample_rate_sps=sample_rate_sps,
        frequency_hz=frequency_hz,
    )


def _estimate_snr_db(
    *,
    signal: np.ndarray,
    sample_rate_sps: float,
    frequency_hz: float,
) -> float:
    """Estimate per-channel SNR in dB using fundamental sine-fit residuals.

    Signal model:
        ``x(t) = A*sin(wt) + B*cos(wt) + C + residual``
    SNR definition:
        ``20*log10(rms(fundamental)/rms(residual))``
    """

    if signal.size < 4:
        raise ValueError("Need at least 4 samples to estimate SNR.")
    t = np.arange(signal.size, dtype=np.float64) / sample_rate_sps
    omega = 2.0 * np.pi * frequency_hz
    design = np.column_stack(
        (
            np.sin(omega * t),
            np.cos(omega * t),
            np.ones_like(t),
        )
    )
    coeffs, _, _, _ = np.linalg.lstsq(design, signal, rcond=None)
    fit = design @ coeffs
    fundamental = (design[:, 0] * coeffs[0]) + (design[:, 1] * coeffs[1])
    residual = signal - fit

    signal_rms = float(np.sqrt(np.mean(np.square(fundamental))))
    noise_rms = float(np.sqrt(np.mean(np.square(residual))))

    if signal_rms <= 1e-18:
        return float("-inf")
    if noise_rms <= 1e-18:
        return float("inf")
    return float(20.0 * np.log10(signal_rms / noise_rms))


def _processing_method_label(config: ImpedanceProcessingConfig) -> str:
    """Build compact extraction method label for result rows."""

    method = _normalize_text(config.method)
    if method == "fft":
        base = "fft"
    else:
        base = f"sine_fit:{_normalize_text(config.sine_fit_backend)}"
    filter_mode = _normalize_text(config.filter_mode)
    if filter_mode != "none":
        base = f"{base}+{filter_mode}"
    return base


def compute_impedance_for_capture(
    *,
    capture: MeasurementCapture,
    config: ImpedanceProcessingConfig | None = None,
) -> ImpedancePointResult:
    """Compute one impedance result row from one capture."""

    effective = config or ImpedanceProcessingConfig()
    _validate_processing_config(effective)

    if capture.raw_data.ndim != 2:
        raise ValueError("capture.raw_data must be a 2D array.")
    if capture.raw_data.shape[1] < 2:
        raise ValueError("capture.raw_data must contain at least 2 samples.")
    if capture.sample_rate_sps <= 0:
        raise ValueError("capture.sample_rate_sps must be > 0.")
    if capture.frequency_hz <= 0:
        raise ValueError("capture.frequency_hz must be > 0.")

    channel_count = capture.raw_data.shape[0]
    if effective.current_channel_index >= channel_count:
        raise ValueError("current_channel_index is outside raw_data channel range.")
    if effective.voltage_channel_index >= channel_count:
        raise ValueError("voltage_channel_index is outside raw_data channel range.")

    current_signal = np.asarray(
        capture.raw_data[effective.current_channel_index],
        dtype=np.float64,
    )
    voltage_signal = np.asarray(
        capture.raw_data[effective.voltage_channel_index],
        dtype=np.float64,
    )

    snr_current_db = _estimate_snr_db(
        signal=current_signal,
        sample_rate_sps=float(capture.sample_rate_sps),
        frequency_hz=float(capture.frequency_hz),
    )
    snr_voltage_db = _estimate_snr_db(
        signal=voltage_signal,
        sample_rate_sps=float(capture.sample_rate_sps),
        frequency_hz=float(capture.frequency_hz),
    )

    if effective.remove_dc_before_extraction:
        current_signal = current_signal - float(np.mean(current_signal))
        voltage_signal = voltage_signal - float(np.mean(voltage_signal))

    current_signal = _apply_frequency_domain_filter(
        signal=current_signal,
        sample_rate_sps=float(capture.sample_rate_sps),
        frequency_hz=float(capture.frequency_hz),
        config=effective,
    )
    voltage_signal = _apply_frequency_domain_filter(
        signal=voltage_signal,
        sample_rate_sps=float(capture.sample_rate_sps),
        frequency_hz=float(capture.frequency_hz),
        config=effective,
    )

    v_shunt_phasor = _extract_phasor(
        signal=current_signal,
        sample_rate_sps=float(capture.sample_rate_sps),
        frequency_hz=float(capture.frequency_hz),
        config=effective,
    )
    v_dut_phasor = _extract_phasor(
        signal=voltage_signal,
        sample_rate_sps=float(capture.sample_rate_sps),
        frequency_hz=float(capture.frequency_hz),
        config=effective,
    )

    i_phasor = v_shunt_phasor / float(effective.shunt_resistance_ohm)
    if abs(i_phasor) <= 1e-15:
        raise ValueError("Computed current phasor is too close to zero for impedance division.")

    z_phasor = v_dut_phasor / i_phasor

    return ImpedancePointResult(
        row_number=int(capture.row_number),
        repeat_index=int(capture.repeat_index),
        frequency_hz=float(capture.frequency_hz),
        z_real_ohm=float(np.real(z_phasor)),
        z_imag_ohm=float(np.imag(z_phasor)),
        z_magnitude_ohm=float(abs(z_phasor)),
        z_phase_deg=float(np.degrees(np.angle(z_phasor))),
        extraction_method=_processing_method_label(effective),
        snr_current_db=snr_current_db,
        snr_voltage_db=snr_voltage_db,
        notes=f"Nominal shunt resistance used: {effective.shunt_resistance_ohm:.9g} ohm",
    )


def compute_impedance_for_run(
    *,
    run_result: SweepRunResult,
    config: ImpedanceProcessingConfig | None = None,
) -> tuple[ImpedancePointResult, ...]:
    """Compute impedance rows for all captures in one run result."""

    rows = [
        compute_impedance_for_capture(capture=capture, config=config)
        for capture in run_result.captures
    ]
    return tuple(rows)
