"""Time-domain raw-vs-fitted plotting for persisted RAW capture CSV files.

Why this module exists:
- Measurement folders already store raw captures per row/repeat in ``RAW/``.
- Engineers need quick visual confirmation that extraction fits are valid,
  especially when noise or setup issues are present.
- This API reads saved raw files directly so notebook analysis can be rerun
  without re-acquiring data.

What this module provides:
- A robust CSV loader for RAW artifacts produced by this project.
- A linear sine-fit routine at commanded frequency.
- Overlay plots of raw channel waveforms and fitted fundamentals.
- Per-channel fit summaries (amplitude, phase, offset, residual RMS, SNR).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


_POINT_FOLDER_FREQ_PATTERN = re.compile(r"_f(?P<freq_token>[0-9_]+)Hz$", re.IGNORECASE)


@dataclass(frozen=True)
class ChannelFitSummary:
    """Sine-fit summary for one raw channel."""

    channel_name: str
    amplitude_v_peak: float
    phase_deg: float
    offset_v: float
    residual_rms_v: float
    snr_db: float


@dataclass(frozen=True)
class RawFitPlotResult:
    """Returned metadata summary for one raw-vs-fitted plot call."""

    raw_csv_path: Path
    frequency_hz: float
    channel_summaries: tuple[ChannelFitSummary, ...]


def infer_frequency_from_raw_path(raw_csv_path: str | Path) -> float:
    """Infer frequency (Hz) from project RAW folder naming convention.

    Expected parent folder format example:
        ``row_0002_f12_54Hz``
    which corresponds to ``12.54 Hz``.
    """

    path = Path(raw_csv_path)
    parent_name = path.parent.name
    match = _POINT_FOLDER_FREQ_PATTERN.search(parent_name)
    if not match:
        raise ValueError(
            "Could not infer frequency from RAW file path. "
            "Provide frequency_hz explicitly."
        )
    freq_token = match.group("freq_token")
    return float(freq_token.replace("_", "."))


def _load_raw_capture_columns(raw_csv_path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load RAW csv into time array and channel-value arrays."""

    path = Path(raw_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"RAW csv file does not exist: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("RAW csv has no header row.")

        if "time_s" not in reader.fieldnames:
            raise ValueError("RAW csv must contain 'time_s' column.")

        channel_names = [
            name for name in reader.fieldnames if name not in {"sample_index", "time_s"}
        ]
        if not channel_names:
            raise ValueError("RAW csv must include at least one channel column.")

        times: list[float] = []
        values: dict[str, list[float]] = {name: [] for name in channel_names}
        for row in reader:
            times.append(float(row["time_s"]))
            for name in channel_names:
                values[name].append(float(row[name]))

    time_s = np.asarray(times, dtype=np.float64)
    channel_arrays = {
        name: np.asarray(series, dtype=np.float64) for name, series in values.items()
    }
    return time_s, channel_arrays


def _fit_sine_linear(
    *,
    time_s: np.ndarray,
    signal: np.ndarray,
    frequency_hz: float,
) -> tuple[np.ndarray, ChannelFitSummary]:
    """Fit one channel to ``A*sin(wt) + B*cos(wt) + C`` and return model+summary."""

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

    summary = ChannelFitSummary(
        channel_name="",
        amplitude_v_peak=amplitude_v_peak,
        phase_deg=phase_deg,
        offset_v=offset_v,
        residual_rms_v=residual_rms_v,
        snr_db=snr_db,
    )
    return fitted.astype(np.float64, copy=False), summary


def plot_raw_vs_fitted_from_csv(
    *,
    raw_csv_path: str | Path,
    frequency_hz: float | None = None,
    channel_names: tuple[str, ...] | None = None,
    max_samples_for_plot: int | None = 4000,
    axes: tuple[plt.Axes, ...] | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, ...], RawFitPlotResult]:
    """Plot raw channel waveforms overlaid with fitted fundamentals.

    Inputs:
        raw_csv_path: Path to one saved RAW capture csv file.
        frequency_hz: Commanded sine frequency in hertz. If omitted, frequency
            is inferred from parent folder name (for project RAW naming style).
        channel_names: Optional ordered subset of channel columns to plot.
            Channel names must match csv header column names (for example
            ``("ai0_v", "ai7_v")``).
        max_samples_for_plot: Optional plot decimation cap. Fit is computed on
            full data and only plotting density is reduced.
        axes: Optional axis tuple with one axis per plotted channel.
        title: Optional figure title.
        save_path: Optional image output path.
    Output:
        ``(fig, axes, result)`` where ``result`` includes per-channel fit
        summaries useful for notebook tables or threshold checks.
    """

    path = Path(raw_csv_path)
    freq_hz = float(frequency_hz) if frequency_hz is not None else infer_frequency_from_raw_path(path)
    if freq_hz <= 0:
        raise ValueError("frequency_hz must be > 0.")

    time_s, all_channels = _load_raw_capture_columns(path)
    if channel_names is None:
        selected_names = tuple(all_channels.keys())
    else:
        selected_names = tuple(channel_names)
        missing = [name for name in selected_names if name not in all_channels]
        if missing:
            raise ValueError(f"Requested channels not found in RAW csv: {missing}")

    if not selected_names:
        raise ValueError("At least one channel must be selected for plotting.")

    if axes is None:
        fig, axes_obj = plt.subplots(
            len(selected_names),
            1,
            figsize=(8.0, 2.8 * len(selected_names) + 0.6),
            sharex=True,
        )
        if len(selected_names) == 1:
            axes_tuple = (axes_obj,)
        else:
            axes_tuple = tuple(axes_obj)
    else:
        axes_tuple = axes
        if len(axes_tuple) != len(selected_names):
            raise ValueError("Length of axes must match number of selected channels.")
        fig = axes_tuple[0].figure

    summaries: list[ChannelFitSummary] = []
    plot_stride = 1
    if max_samples_for_plot is not None and max_samples_for_plot > 0 and time_s.size > max_samples_for_plot:
        plot_stride = int(math.ceil(time_s.size / float(max_samples_for_plot)))

    for axis, channel_name in zip(axes_tuple, selected_names):
        signal = all_channels[channel_name]
        fitted, summary_base = _fit_sine_linear(
            time_s=time_s,
            signal=signal,
            frequency_hz=freq_hz,
        )
        summary = ChannelFitSummary(
            channel_name=channel_name,
            amplitude_v_peak=summary_base.amplitude_v_peak,
            phase_deg=summary_base.phase_deg,
            offset_v=summary_base.offset_v,
            residual_rms_v=summary_base.residual_rms_v,
            snr_db=summary_base.snr_db,
        )
        summaries.append(summary)

        axis.plot(
            time_s[::plot_stride],
            signal[::plot_stride],
            color="#4d4d4d",
            linewidth=0.9,
            alpha=0.85,
            label="raw",
        )
        axis.plot(
            time_s[::plot_stride],
            fitted[::plot_stride],
            color="#1f77b4",
            linewidth=1.3,
            alpha=0.95,
            label="fitted fundamental",
        )
        axis.set_ylabel(f"{channel_name} (V)")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=8)
        axis.text(
            0.01,
            0.97,
            (
                f"SNR={summary.snr_db:.2f} dB | "
                f"A={summary.amplitude_v_peak:.4g} Vpk | "
                f"phase={summary.phase_deg:.2f} deg"
            ),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55},
        )

    axes_tuple[-1].set_xlabel("Time (s)")
    fig.suptitle(title or f"Raw vs Fitted | f={freq_hz:.6g} Hz")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=140)

    return (
        fig,
        axes_tuple,
        RawFitPlotResult(
            raw_csv_path=path,
            frequency_hz=freq_hz,
            channel_summaries=tuple(summaries),
        ),
    )
