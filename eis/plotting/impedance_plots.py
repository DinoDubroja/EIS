"""Impedance plotting helpers for single-run and multi-run notebook workflows.

This module intentionally keeps plotting entry points small and explicit so
technicians and engineers can read function signatures and understand:
- which measurement runs are selected from disk
- how repeats are aggregated before plotting
- where output images are written for report/notebook usage

Current views:
- Nyquist and inverse Nyquist overlays
- Bode magnitude/phase overlays
- SNR vs frequency overlays with threshold-region shading and pass/fail checks
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eis.plotting.run_selection import RunFolderRecord, RunSelection, select_run_folders
from eis.storage.run_artifacts import load_impedance_rows_from_run


@dataclass(frozen=True)
class SNRThresholdCheckResult:
    """Threshold check result for one plotted run in SNR-vs-frequency view.

    Fields:
        run: Run metadata inferred from the run folder name.
        checked_points: Number of SNR points checked against threshold.
        threshold_db: Threshold used for this check.
        good_region: Rule used to classify pass/fail:
            - ``"below_threshold"``
            - ``"above_threshold"``
        passed: True if all checked points are in the configured good region.
        min_snr_db: Minimum SNR value in checked points.
        max_snr_db: Maximum SNR value in checked points.
    """

    run: RunFolderRecord
    checked_points: int
    threshold_db: float
    good_region: str
    passed: bool
    min_snr_db: float
    max_snr_db: float


def _run_label(run: RunFolderRecord) -> str:
    """Build readable legend label for one run."""

    return f"{run.serial_number} | {run.started_at_local:%Y-%m-%d %H:%M}"


def _row_to_complex_impedance(row: dict[str, object]) -> complex:
    """Convert one impedance row dictionary to complex impedance."""

    return complex(float(row["z_real_ohm"]), float(row["z_imag_ohm"]))


def _aggregate_mean_by_frequency(rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate repeat rows into one mean complex impedance per frequency."""

    grouped: dict[float, list[complex]] = {}
    for row in rows:
        frequency_hz = float(row["frequency_hz"])
        grouped.setdefault(frequency_hz, []).append(_row_to_complex_impedance(row))

    frequencies = np.asarray(sorted(grouped.keys()), dtype=np.float64)
    z_values = np.asarray(
        [np.mean(np.asarray(grouped[freq], dtype=np.complex128)) for freq in frequencies],
        dtype=np.complex128,
    )
    return frequencies, z_values


def _extract_impedance_series(
    rows: list[dict[str, object]],
    *,
    aggregate_repeats: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract frequency and complex impedance arrays from row dictionaries."""

    if aggregate_repeats:
        return _aggregate_mean_by_frequency(rows)

    sorted_rows = sorted(rows, key=lambda item: (float(item["frequency_hz"]), int(item["repeat_index"])))
    frequencies = np.asarray([float(item["frequency_hz"]) for item in sorted_rows], dtype=np.float64)
    z_values = np.asarray(
        [_row_to_complex_impedance(item) for item in sorted_rows],
        dtype=np.complex128,
    )
    return frequencies, z_values


def _extract_snr_series(
    rows: list[dict[str, object]],
    *,
    snr_key: str,
    aggregate_repeats: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract frequency and SNR arrays from impedance row dictionaries."""

    cleaned = [
        item for item in rows if item.get(snr_key) is not None and str(item.get(snr_key)) != ""
    ]
    if not cleaned:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    if aggregate_repeats:
        grouped: dict[float, list[float]] = {}
        for row in cleaned:
            frequency_hz = float(row["frequency_hz"])
            grouped.setdefault(frequency_hz, []).append(float(row[snr_key]))
        frequencies = np.asarray(sorted(grouped.keys()), dtype=np.float64)
        values = np.asarray([np.mean(grouped[freq]) for freq in frequencies], dtype=np.float64)
        return frequencies, values

    sorted_rows = sorted(cleaned, key=lambda item: (float(item["frequency_hz"]), int(item["repeat_index"])))
    frequencies = np.asarray([float(item["frequency_hz"]) for item in sorted_rows], dtype=np.float64)
    values = np.asarray([float(item[snr_key]) for item in sorted_rows], dtype=np.float64)
    return frequencies, values


def _normalize_snr_key(snr_source: str) -> str:
    """Map user-facing SNR source aliases to impedance table column names."""

    normalized = snr_source.strip().lower()
    mapping = {
        "current": "snr_current_db",
        "ch1": "snr_current_db",
        "ai0": "snr_current_db",
        "voltage": "snr_voltage_db",
        "ch2": "snr_voltage_db",
        "ai7": "snr_voltage_db",
    }
    if normalized not in mapping:
        raise ValueError(
            "snr_source must be one of: current, voltage, ch1, ch2, ai0, ai7."
        )
    return mapping[normalized]


def _normalize_good_region(good_region: str) -> str:
    """Validate and normalize threshold pass/fail orientation."""

    normalized = good_region.strip().lower()
    if normalized not in {"below_threshold", "above_threshold"}:
        raise ValueError("good_region must be 'below_threshold' or 'above_threshold'.")
    return normalized


def plot_impedance_nyquist(
    *,
    base_output_dir: str | Path,
    selection: RunSelection | None = None,
    aggregate_repeats: bool = True,
    ax=None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes, tuple[RunFolderRecord, ...]]:
    """Plot Nyquist overlay for selected run folders.

    Inputs:
        base_output_dir: Root output folder containing measurement run folders.
        selection: Run selection/filter configuration.
        aggregate_repeats: If true, each run contributes one point per frequency
            (mean over repeats). If false, each repeat row is plotted.
        ax: Optional matplotlib axis. If omitted, new figure is created.
        title: Optional custom title.
        save_path: Optional output image path. If provided, figure is saved.
    Output:
        Tuple ``(fig, ax, selected_runs)``.
    Raises:
        ValueError: No runs matched or no impedance rows were found.
    """

    selected_runs = select_run_folders(base_output_dir=base_output_dir, selection=selection)
    if not selected_runs:
        raise ValueError("No run folders matched requested selection.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
    else:
        fig = ax.figure

    return _plot_nyquist_core(
        selected_runs=selected_runs,
        aggregate_repeats=aggregate_repeats,
        ax=ax,
        title=(title or "Nyquist Plot"),
        y_mode="x",
        y_label="X (Ohm)",
        save_path=save_path,
    )


def plot_impedance_inverse_nyquist(
    *,
    base_output_dir: str | Path,
    selection: RunSelection | None = None,
    aggregate_repeats: bool = True,
    ax=None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes, tuple[RunFolderRecord, ...]]:
    """Plot inverse Nyquist overlay (R vs -X) for selected run folders."""

    selected_runs = select_run_folders(base_output_dir=base_output_dir, selection=selection)
    if not selected_runs:
        raise ValueError("No run folders matched requested selection.")

    return _plot_nyquist_core(
        selected_runs=selected_runs,
        aggregate_repeats=aggregate_repeats,
        ax=ax,
        title=(title or "Inverse Nyquist Plot"),
        y_mode="minus_x",
        y_label="-X (Ohm)",
        save_path=save_path,
    )


def _plot_nyquist_core(
    *,
    selected_runs: tuple[RunFolderRecord, ...],
    aggregate_repeats: bool,
    ax,
    title: str,
    y_mode: str,
    y_label: str,
    save_path: str | Path | None,
) -> tuple[plt.Figure, plt.Axes, tuple[RunFolderRecord, ...]]:
    """Internal Nyquist plotting implementation shared by normal/inverse views."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
    else:
        fig = ax.figure

    plotted = 0
    for run in selected_runs:
        rows = load_impedance_rows_from_run(run.root)
        if not rows:
            continue
        _, z_values = _extract_impedance_series(rows, aggregate_repeats=aggregate_repeats)
        y_values = np.imag(z_values) if y_mode == "x" else -np.imag(z_values)
        ax.plot(
            np.real(z_values),
            y_values,
            marker="o",
            linewidth=1.2,
            label=_run_label(run),
        )
        plotted += 1

    if plotted == 0:
        raise ValueError("Selected runs contain no impedance rows to plot.")

    ax.set_xlabel("R (Ohm)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=140)

    return fig, ax, selected_runs


def plot_impedance_bode(
    *,
    base_output_dir: str | Path,
    selection: RunSelection | None = None,
    aggregate_repeats: bool = True,
    axes: tuple[plt.Axes, plt.Axes] | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes], tuple[RunFolderRecord, ...]]:
    """Plot Bode magnitude/phase overlay for selected run folders.

    Inputs:
        base_output_dir: Root output folder containing measurement run folders.
        selection: Run selection/filter configuration.
        aggregate_repeats: If true, each run contributes one point per frequency
            (mean over repeats). If false, each repeat row is plotted.
        axes: Optional tuple ``(ax_magnitude, ax_phase)``.
        title: Optional figure-level title.
        save_path: Optional output image path. If provided, figure is saved.
    Output:
        Tuple ``(fig, (ax_mag, ax_phase), selected_runs)``.
    Raises:
        ValueError: No runs matched or no impedance rows were found.
    """

    selected_runs = select_run_folders(base_output_dir=base_output_dir, selection=selection)
    if not selected_runs:
        raise ValueError("No run folders matched requested selection.")

    if axes is None:
        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    else:
        ax_mag, ax_phase = axes
        fig = ax_mag.figure

    plotted = 0
    for run in selected_runs:
        rows = load_impedance_rows_from_run(run.root)
        if not rows:
            continue
        frequencies, z_values = _extract_impedance_series(rows, aggregate_repeats=aggregate_repeats)

        magnitude = np.abs(z_values)
        phase_deg = np.degrees(np.angle(z_values))
        label = _run_label(run)

        ax_mag.plot(frequencies, magnitude, marker="o", linewidth=1.2, label=label)
        ax_phase.plot(frequencies, phase_deg, marker="o", linewidth=1.2, label=label)
        plotted += 1

    if plotted == 0:
        raise ValueError("Selected runs contain no impedance rows to plot.")

    ax_mag.set_ylabel("|Z| (Ohm)")
    ax_mag.set_xscale("log")
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.legend(loc="best", fontsize=8)

    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.set_xscale("log")
    ax_phase.grid(True, which="both", alpha=0.3)
    ax_phase.legend(loc="best", fontsize=8)

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        fig.tight_layout()

    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=140)

    return fig, (ax_mag, ax_phase), selected_runs


def plot_snr_vs_frequency(
    *,
    base_output_dir: str | Path,
    selection: RunSelection | None = None,
    snr_source: str = "voltage",
    aggregate_repeats: bool = True,
    threshold_db: float | None = None,
    good_region: str = "below_threshold",
    ax=None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[
    plt.Figure,
    plt.Axes,
    tuple[RunFolderRecord, ...],
    tuple[SNRThresholdCheckResult, ...],
]:
    """Plot SNR vs frequency with optional threshold shading and run checks.

    Inputs:
        base_output_dir: Root output folder containing measurement run folders.
        selection: Run selection/filter configuration.
        snr_source: SNR column source:
            - ``"current"``/``"ch1"``/``"ai0"``
            - ``"voltage"``/``"ch2"``/``"ai7"``
        aggregate_repeats: If true, SNR is averaged per frequency over repeats.
        threshold_db: Optional SNR threshold for pass/fail checks.
        good_region: How threshold is interpreted:
            - ``"below_threshold"``: values <= threshold are considered good.
            - ``"above_threshold"``: values >= threshold are considered good.
        ax: Optional matplotlib axis. If omitted, a new figure is created.
        title: Optional custom title.
        save_path: Optional output image path. If provided, figure is saved.
    Output:
        Tuple ``(fig, ax, selected_runs, threshold_results)``.
    Notes:
        Default ``good_region="below_threshold"`` follows current demo
        preference. For conventional SNR acceptance checks, use
        ``good_region="above_threshold"``.
    Raises:
        ValueError: No runs matched or no SNR rows were found for source.
    """

    selected_runs = select_run_folders(base_output_dir=base_output_dir, selection=selection)
    if not selected_runs:
        raise ValueError("No run folders matched requested selection.")

    snr_key = _normalize_snr_key(snr_source)
    threshold_rule = _normalize_good_region(good_region)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
    else:
        fig = ax.figure

    checks: list[SNRThresholdCheckResult] = []
    plotted = 0

    for run in selected_runs:
        rows = load_impedance_rows_from_run(run.root)
        if not rows:
            continue
        frequencies, snr_values = _extract_snr_series(
            rows,
            snr_key=snr_key,
            aggregate_repeats=aggregate_repeats,
        )
        if snr_values.size == 0:
            continue

        ax.plot(
            frequencies,
            snr_values,
            marker="o",
            linewidth=1.2,
            label=_run_label(run),
        )
        plotted += 1

        if threshold_db is not None:
            if threshold_rule == "below_threshold":
                passed = bool(np.all(snr_values <= threshold_db))
            else:
                passed = bool(np.all(snr_values >= threshold_db))
            checks.append(
                SNRThresholdCheckResult(
                    run=run,
                    checked_points=int(snr_values.size),
                    threshold_db=float(threshold_db),
                    good_region=threshold_rule,
                    passed=passed,
                    min_snr_db=float(np.min(snr_values)),
                    max_snr_db=float(np.max(snr_values)),
                )
            )

    if plotted == 0:
        raise ValueError("Selected runs contain no SNR rows to plot for requested snr_source.")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("SNR (dB)")
    ax.set_xscale("log")
    ax.set_title(title or f"SNR vs Frequency ({snr_key})")
    ax.grid(True, which="both", alpha=0.3)

    if threshold_db is not None:
        y_low, y_high = ax.get_ylim()
        y_low = min(y_low, float(threshold_db))
        y_high = max(y_high, float(threshold_db))
        ax.set_ylim(y_low, y_high)

        if threshold_rule == "below_threshold":
            good_low, good_high = y_low, min(float(threshold_db), y_high)
            bad_low, bad_high = max(float(threshold_db), y_low), y_high
        else:
            good_low, good_high = max(float(threshold_db), y_low), y_high
            bad_low, bad_high = y_low, min(float(threshold_db), y_high)

        if good_high > good_low:
            ax.axhspan(good_low, good_high, color="#2ca02c", alpha=0.16, zorder=0)
        if bad_high > bad_low:
            ax.axhspan(bad_low, bad_high, color="#d62728", alpha=0.16, zorder=0)
        ax.axhline(
            float(threshold_db),
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"threshold {threshold_db:.3g} dB",
        )

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=140)

    return fig, ax, selected_runs, tuple(checks)
