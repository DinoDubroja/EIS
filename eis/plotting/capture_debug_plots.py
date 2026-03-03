"""Capture-level debug plots for one selected frequency/repeat from last run.

These helpers operate on in-memory ``SweepRunResult`` captures so notebook
users can inspect one selected measurement point without reloading from disk.
Each plot supports selectable signal components:
- ``raw``: directly acquired channel waveform
- ``filtered``: waveform after the same DC/filter conditioning used for
  impedance extraction
- ``fitted``: fitted fundamental sine reconstructed from conditioned signal
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eis.models.measurement_models import MeasurementCapture, SweepRunResult
from eis.processing import ImpedanceProcessingConfig, prepare_signal_for_processing


_CURRENT_COLOR = "#8B0000"  # dark red
_VOLTAGE_COLOR = "#00008B"  # dark blue
_FALLBACK_COLOR = "#4D4D4D"
_VALID_COMPONENTS = ("raw", "filtered", "fitted")
_COMPONENT_STYLE = {
    "raw": {"linestyle": "-", "linewidth": 0.9, "alpha": 0.45},
    "filtered": {"linestyle": "-", "linewidth": 1.2, "alpha": 0.9},
    "fitted": {"linestyle": "--", "linewidth": 1.4, "alpha": 0.98},
}


@dataclass(frozen=True)
class CaptureDebugComponentSummary:
    """Sine-fit summary values for one channel and one selected component.

    Fields:
        channel_name: Channel label (for example ``ai0``).
        component_name: One of ``raw``, ``filtered``, ``fitted``.
        amplitude_v_peak: Fitted fundamental amplitude in volts peak (Vpk).
        phase_deg: Fitted phase in degrees.
        offset_v: Fitted DC offset in volts.
        residual_rms_v: Residual RMS in volts after fit.
        snr_db: Fundamental-to-residual ratio in dB.
    """

    channel_name: str
    component_name: str
    amplitude_v_peak: float
    phase_deg: float
    offset_v: float
    residual_rms_v: float
    snr_db: float


@dataclass(frozen=True)
class CaptureDebugPlotResult:
    """Returned context from one time-domain or FFT debug plotting call.

    Fields:
        row_number: Config row number selected for plotting.
        repeat_index: Repeat index selected for plotting.
        frequency_hz: Capture frequency in hertz.
        components: Components that were plotted.
        channel_summaries: Per-channel/per-component fit summaries.
    """

    row_number: int
    repeat_index: int
    frequency_hz: float
    components: tuple[str, ...]
    channel_summaries: tuple[CaptureDebugComponentSummary, ...]


@dataclass(frozen=True)
class _FitSummary:
    """Internal fit container used while building debug plots."""

    amplitude_v_peak: float
    phase_deg: float
    offset_v: float
    residual_rms_v: float
    snr_db: float


def _normalize_components(
    components: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    """Validate and normalize selected component list preserving input order."""

    if components is None:
        return _VALID_COMPONENTS

    normalized: list[str] = []
    seen: set[str] = set()
    for item in components:
        token = str(item).strip().lower()
        if token not in _VALID_COMPONENTS:
            raise ValueError(
                "components must contain only: raw, filtered, fitted."
            )
        if token not in seen:
            normalized.append(token)
            seen.add(token)
    if not normalized:
        raise ValueError("components must include at least one selection.")
    return tuple(normalized)


def _channel_color(channel_name: str) -> str:
    """Resolve line color from channel naming convention."""

    token = channel_name.strip().lower()
    if any(key in token for key in ("current", "ai0", "ch1")):
        return _CURRENT_COLOR
    if any(key in token for key in ("voltage", "ai7", "ch2")):
        return _VOLTAGE_COLOR
    return _FALLBACK_COLOR


def _fit_sine_linear(
    *,
    time_s: np.ndarray,
    signal: np.ndarray,
    frequency_hz: float,
) -> tuple[np.ndarray, _FitSummary]:
    """Fit one signal to ``A*sin(wt) + B*cos(wt) + C`` and return model+stats."""

    omega = 2.0 * np.pi * frequency_hz
    design = np.column_stack(
        (
            np.sin(omega * time_s),
            np.cos(omega * time_s),
            np.ones_like(time_s),
        )
    )
    coeffs, _, _, _ = np.linalg.lstsq(design, signal, rcond=None)

    sine_coeff = float(coeffs[0])
    cosine_coeff = float(coeffs[1])
    offset_v = float(coeffs[2])

    fitted = design @ coeffs
    residual = signal - fitted
    fundamental = (design[:, 0] * sine_coeff) + (design[:, 1] * cosine_coeff)

    amplitude_v_peak = float(math.hypot(sine_coeff, cosine_coeff))
    phase_deg = float(np.degrees(np.arctan2(cosine_coeff, sine_coeff)))
    residual_rms_v = float(np.sqrt(np.mean(np.square(residual))))
    fundamental_rms = float(np.sqrt(np.mean(np.square(fundamental))))

    if fundamental_rms <= 1e-18:
        snr_db = float("-inf")
    elif residual_rms_v <= 1e-18:
        snr_db = float("inf")
    else:
        snr_db = float(20.0 * np.log10(fundamental_rms / residual_rms_v))

    return fitted.astype(np.float64, copy=False), _FitSummary(
        amplitude_v_peak=amplitude_v_peak,
        phase_deg=phase_deg,
        offset_v=offset_v,
        residual_rms_v=residual_rms_v,
        snr_db=snr_db,
    )


def _format_snr(snr_db: float) -> str:
    """Format SNR value for labels and console output."""

    if np.isinf(snr_db):
        return "inf" if snr_db > 0 else "-inf"
    return f"{snr_db:.2f}"


def _select_capture(
    *,
    run_result: SweepRunResult,
    frequency_hz: float,
    repeat_index: int,
    frequency_tolerance_hz: float,
) -> MeasurementCapture:
    """Select one capture by frequency and repeat index from run result."""

    if repeat_index < 1:
        raise ValueError("repeat_index must be >= 1.")
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be > 0.")
    if frequency_tolerance_hz < 0:
        raise ValueError("frequency_tolerance_hz must be >= 0.")

    matches = [
        capture
        for capture in run_result.captures
        if capture.repeat_index == repeat_index
        and abs(float(capture.frequency_hz) - float(frequency_hz)) <= frequency_tolerance_hz
    ]
    if not matches:
        raise ValueError(
            "No capture found for selected frequency/repeat. "
            f"frequency_hz={frequency_hz:.12g}, repeat_index={repeat_index}."
        )
    if len(matches) > 1:
        row_numbers = ", ".join(str(item.row_number) for item in matches)
        raise ValueError(
            "Frequency/repeat selection is ambiguous. "
            f"Matched multiple rows: {row_numbers}. "
            "Use a tighter frequency_tolerance_hz or unique sweep frequencies."
        )
    return matches[0]


def _resolve_channel_indices(
    *,
    capture: MeasurementCapture,
    channel_indices: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...]:
    """Validate and normalize selected channel indices."""

    count = len(capture.ai_channels)
    if channel_indices is None:
        return tuple(range(count))
    selected = tuple(int(value) for value in channel_indices)
    if not selected:
        raise ValueError("channel_indices must include at least one channel.")
    for index in selected:
        if index < 0 or index >= count:
            raise ValueError(
                f"channel index {index} is outside available range 0..{count - 1}."
            )
    return selected


def _build_component_signals(
    *,
    raw_signal: np.ndarray,
    sample_rate_sps: float,
    frequency_hz: float,
    processing_config: ImpedanceProcessingConfig,
) -> tuple[dict[str, np.ndarray], dict[str, _FitSummary]]:
    """Build raw/filtered/fitted signals and fit summaries for one channel."""

    conditioned = prepare_signal_for_processing(
        signal=raw_signal,
        sample_rate_sps=sample_rate_sps,
        frequency_hz=frequency_hz,
        config=processing_config,
    )
    time_s = np.arange(raw_signal.size, dtype=np.float64) / sample_rate_sps

    raw_fitted, raw_summary = _fit_sine_linear(
        time_s=time_s,
        signal=raw_signal,
        frequency_hz=frequency_hz,
    )
    filtered_fitted, filtered_summary = _fit_sine_linear(
        time_s=time_s,
        signal=conditioned,
        frequency_hz=frequency_hz,
    )
    _, fitted_summary = _fit_sine_linear(
        time_s=time_s,
        signal=filtered_fitted,
        frequency_hz=frequency_hz,
    )
    signal_map = {
        "raw": np.asarray(raw_signal, dtype=np.float64),
        "filtered": np.asarray(conditioned, dtype=np.float64),
        "fitted": np.asarray(filtered_fitted, dtype=np.float64),
    }
    summary_map = {
        "raw": raw_summary,
        "filtered": filtered_summary,
        "fitted": fitted_summary,
    }
    return signal_map, summary_map


def _amplitude_spectrum(
    *,
    signal: np.ndarray,
    sample_rate_sps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-sided amplitude spectrum for a real-valued signal."""

    sample_count = int(signal.size)
    if sample_count < 2:
        raise ValueError("Need at least two samples for FFT plotting.")

    freqs = np.fft.rfftfreq(sample_count, d=1.0 / sample_rate_sps)
    spectrum = np.fft.rfft(signal)
    amplitudes = np.abs(spectrum) * (2.0 / sample_count)
    amplitudes[0] = amplitudes[0] * 0.5
    if sample_count % 2 == 0 and amplitudes.size > 1:
        amplitudes[-1] = amplitudes[-1] * 0.5
    return freqs, amplitudes.astype(np.float64, copy=False)


def _append_component_summaries(
    *,
    summaries: list[CaptureDebugComponentSummary],
    channel_name: str,
    components: tuple[str, ...],
    summary_map: dict[str, _FitSummary],
) -> None:
    """Append selected component fit summaries to output list."""

    for component_name in components:
        fit = summary_map[component_name]
        summaries.append(
            CaptureDebugComponentSummary(
                channel_name=channel_name,
                component_name=component_name,
                amplitude_v_peak=fit.amplitude_v_peak,
                phase_deg=fit.phase_deg,
                offset_v=fit.offset_v,
                residual_rms_v=fit.residual_rms_v,
                snr_db=fit.snr_db,
            )
        )


def _maybe_print_snr_table(
    *,
    capture: MeasurementCapture,
    summaries: list[CaptureDebugComponentSummary],
) -> None:
    """Print concise SNR table for selected channel/component traces."""

    print(
        "SNR summary | "
        f"row={capture.row_number}, repeat={capture.repeat_index}, "
        f"f={capture.frequency_hz:.6g} Hz"
    )
    for item in summaries:
        print(
            "  "
            f"{item.channel_name:<8} {item.component_name:<8} "
            f"SNR={_format_snr(item.snr_db)} dB"
        )


def plot_capture_time_domain_components(
    *,
    run_result: SweepRunResult,
    frequency_hz: float,
    repeat_index: int = 1,
    components: tuple[str, ...] | list[str] | None = None,
    processing_config: ImpedanceProcessingConfig | None = None,
    channel_indices: tuple[int, ...] | list[int] | None = None,
    max_samples_for_plot: int | None = 4000,
    frequency_tolerance_hz: float = 1e-6,
    axes: tuple[plt.Axes, ...] | None = None,
    title: str | None = None,
    print_snr_table: bool = True,
    save_path: str | Path | None = None,
    save_vector_path: str | Path | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, ...], CaptureDebugPlotResult]:
    """Plot selected time-domain components for one capture frequency/repeat.

    Inputs:
        run_result: In-memory sweep result containing captures.
        frequency_hz: Selected frequency in hertz.
        repeat_index: Selected repeat index (1-based).
        components: Components to plot: ``raw``, ``filtered``, ``fitted``.
        processing_config: Conditioning/filter config for ``filtered``/``fitted``.
        channel_indices: Optional subset of channel indices to plot.
        max_samples_for_plot: Optional decimation cap for display only.
        frequency_tolerance_hz: Absolute frequency tolerance for capture match.
        axes: Optional axis tuple, one axis per selected channel.
        title: Optional figure title.
        print_snr_table: If true, print component SNR summary to console.
        save_path: Optional raster image output path.
        save_vector_path: Optional vector image output path (for example ``.svg``).
    Output:
        ``(fig, axes, result)`` where result includes selected capture identity
        and fit/SNR summary for each channel/component.
    """

    selected_components = _normalize_components(components)
    capture = _select_capture(
        run_result=run_result,
        frequency_hz=frequency_hz,
        repeat_index=repeat_index,
        frequency_tolerance_hz=frequency_tolerance_hz,
    )
    selected_channels = _resolve_channel_indices(
        capture=capture,
        channel_indices=channel_indices,
    )
    cfg = processing_config or ImpedanceProcessingConfig()

    time_s = np.arange(capture.raw_data.shape[1], dtype=np.float64) / float(capture.sample_rate_sps)
    stride = 1
    if max_samples_for_plot is not None and max_samples_for_plot > 0 and time_s.size > max_samples_for_plot:
        stride = int(math.ceil(time_s.size / float(max_samples_for_plot)))

    if axes is None:
        fig, axes_obj = plt.subplots(
            len(selected_channels),
            1,
            figsize=(8.2, 2.9 * len(selected_channels) + 0.8),
            sharex=True,
        )
        axes_tuple = (axes_obj,) if len(selected_channels) == 1 else tuple(axes_obj)
    else:
        axes_tuple = axes
        if len(axes_tuple) != len(selected_channels):
            raise ValueError("Length of axes must match selected channel count.")
        fig = axes_tuple[0].figure

    summary_rows: list[CaptureDebugComponentSummary] = []
    for axis, channel_index in zip(axes_tuple, selected_channels):
        channel_name = capture.ai_channels[channel_index]
        base_color = _channel_color(channel_name)
        raw_signal = np.asarray(capture.raw_data[channel_index], dtype=np.float64)
        signal_map, summary_map = _build_component_signals(
            raw_signal=raw_signal,
            sample_rate_sps=float(capture.sample_rate_sps),
            frequency_hz=float(capture.frequency_hz),
            processing_config=cfg,
        )

        for component_name in selected_components:
            style = _COMPONENT_STYLE[component_name]
            fit_summary = summary_map[component_name]
            axis.plot(
                time_s[::stride],
                signal_map[component_name][::stride],
                color=base_color,
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
                label=f"{component_name} | SNR={_format_snr(fit_summary.snr_db)} dB",
            )

        axis.set_ylabel(f"{channel_name} (V)")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=8)

        _append_component_summaries(
            summaries=summary_rows,
            channel_name=channel_name,
            components=selected_components,
            summary_map=summary_map,
        )

    axes_tuple[-1].set_xlabel("Time (s)")
    fig.suptitle(
        title
        or (
            "Time Domain Debug | "
            f"row={capture.row_number}, repeat={capture.repeat_index}, "
            f"f={capture.frequency_hz:.6g} Hz"
        )
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        raster_output = Path(save_path)
        raster_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(raster_output, dpi=140)
    if save_vector_path is not None:
        vector_output = Path(save_vector_path)
        vector_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(vector_output)

    if print_snr_table:
        _maybe_print_snr_table(capture=capture, summaries=summary_rows)

    return (
        fig,
        axes_tuple,
        CaptureDebugPlotResult(
            row_number=capture.row_number,
            repeat_index=capture.repeat_index,
            frequency_hz=float(capture.frequency_hz),
            components=selected_components,
            channel_summaries=tuple(summary_rows),
        ),
    )


def plot_capture_fft_components(
    *,
    run_result: SweepRunResult,
    frequency_hz: float,
    repeat_index: int = 1,
    components: tuple[str, ...] | list[str] | None = None,
    processing_config: ImpedanceProcessingConfig | None = None,
    channel_indices: tuple[int, ...] | list[int] | None = None,
    frequency_tolerance_hz: float = 1e-6,
    max_frequency_hz: float | None = None,
    axes: tuple[plt.Axes, ...] | None = None,
    title: str | None = None,
    print_snr_table: bool = True,
    save_path: str | Path | None = None,
    save_vector_path: str | Path | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, ...], CaptureDebugPlotResult]:
    """Plot FFT magnitude of selected components for one capture.

    Inputs:
        run_result: In-memory sweep result containing captures.
        frequency_hz: Selected frequency in hertz.
        repeat_index: Selected repeat index (1-based).
        components: Components to plot: ``raw``, ``filtered``, ``fitted``.
        processing_config: Conditioning/filter config for ``filtered``/``fitted``.
        channel_indices: Optional subset of channel indices to plot.
        frequency_tolerance_hz: Absolute frequency tolerance for capture match.
        max_frequency_hz: Optional x-axis upper bound for FFT in hertz.
        axes: Optional axis tuple, one axis per selected channel.
        title: Optional figure title.
        print_snr_table: If true, print component SNR summary to console.
        save_path: Optional raster image output path.
        save_vector_path: Optional vector image output path (for example ``.svg``).
    Output:
        ``(fig, axes, result)`` where result includes selected capture identity
        and fit/SNR summary for each channel/component.
    """

    selected_components = _normalize_components(components)
    capture = _select_capture(
        run_result=run_result,
        frequency_hz=frequency_hz,
        repeat_index=repeat_index,
        frequency_tolerance_hz=frequency_tolerance_hz,
    )
    selected_channels = _resolve_channel_indices(
        capture=capture,
        channel_indices=channel_indices,
    )
    cfg = processing_config or ImpedanceProcessingConfig()

    if axes is None:
        fig, axes_obj = plt.subplots(
            len(selected_channels),
            1,
            figsize=(8.2, 2.9 * len(selected_channels) + 0.8),
            sharex=True,
        )
        axes_tuple = (axes_obj,) if len(selected_channels) == 1 else tuple(axes_obj)
    else:
        axes_tuple = axes
        if len(axes_tuple) != len(selected_channels):
            raise ValueError("Length of axes must match selected channel count.")
        fig = axes_tuple[0].figure

    summary_rows: list[CaptureDebugComponentSummary] = []
    for axis, channel_index in zip(axes_tuple, selected_channels):
        channel_name = capture.ai_channels[channel_index]
        base_color = _channel_color(channel_name)
        raw_signal = np.asarray(capture.raw_data[channel_index], dtype=np.float64)
        signal_map, summary_map = _build_component_signals(
            raw_signal=raw_signal,
            sample_rate_sps=float(capture.sample_rate_sps),
            frequency_hz=float(capture.frequency_hz),
            processing_config=cfg,
        )

        for component_name in selected_components:
            style = _COMPONENT_STYLE[component_name]
            fit_summary = summary_map[component_name]
            freqs, amps = _amplitude_spectrum(
                signal=signal_map[component_name],
                sample_rate_sps=float(capture.sample_rate_sps),
            )
            axis.plot(
                freqs,
                amps,
                color=base_color,
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
                label=f"{component_name} | SNR={_format_snr(fit_summary.snr_db)} dB",
            )

        axis.axvline(
            float(capture.frequency_hz),
            color=base_color,
            linestyle=":",
            linewidth=1.0,
            alpha=0.8,
        )
        axis.set_ylabel(f"{channel_name} (V)")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=8)

        _append_component_summaries(
            summaries=summary_rows,
            channel_name=channel_name,
            components=selected_components,
            summary_map=summary_map,
        )

    nyquist_hz = 0.5 * float(capture.sample_rate_sps)
    if max_frequency_hz is None:
        upper = min(nyquist_hz, max(float(capture.frequency_hz) * 8.0, float(capture.frequency_hz) + 20.0))
    else:
        if max_frequency_hz <= 0:
            raise ValueError("max_frequency_hz must be > 0 when provided.")
        upper = min(float(max_frequency_hz), nyquist_hz)
    for axis in axes_tuple:
        axis.set_xlim(0.0, upper)

    axes_tuple[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle(
        title
        or (
            "Frequency Domain Debug (FFT) | "
            f"row={capture.row_number}, repeat={capture.repeat_index}, "
            f"f={capture.frequency_hz:.6g} Hz"
        )
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        raster_output = Path(save_path)
        raster_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(raster_output, dpi=140)
    if save_vector_path is not None:
        vector_output = Path(save_vector_path)
        vector_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(vector_output)

    if print_snr_table:
        _maybe_print_snr_table(capture=capture, summaries=summary_rows)

    return (
        fig,
        axes_tuple,
        CaptureDebugPlotResult(
            row_number=capture.row_number,
            repeat_index=capture.repeat_index,
            frequency_hz=float(capture.frequency_hz),
            components=selected_components,
            channel_summaries=tuple(summary_rows),
        ),
    )
