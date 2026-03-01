"""Impedance plotting helpers with run selection from measurement folders.

Primary use case:
- In notebooks, quickly plot newest run, last N runs, or all runs.
- Filter candidate runs by serial number and local start time.
- Overlay selected runs on Nyquist/Bode views for comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eis.plotting.run_selection import RunFolderRecord, RunSelection, select_run_folders
from eis.storage.run_artifacts import load_impedance_rows_from_run


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


def plot_impedance_nyquist(
    *,
    base_output_dir: str | Path,
    selection: RunSelection | None = None,
    aggregate_repeats: bool = True,
    ax=None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes, tuple[RunFolderRecord, ...]]:
    """Plot Nyquist overlay for selected run folders.

    Inputs:
        base_output_dir: Root output folder containing measurement run folders.
        selection: Run selection/filter configuration.
        aggregate_repeats: If true, each run contributes one point per frequency
            (mean over repeats). If false, each repeat row is plotted.
        ax: Optional matplotlib axis. If omitted, new figure is created.
        title: Optional custom title.
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

    plotted = 0
    for run in selected_runs:
        rows = load_impedance_rows_from_run(run.root)
        if not rows:
            continue
        _, z_values = _extract_impedance_series(rows, aggregate_repeats=aggregate_repeats)
        ax.plot(
            np.real(z_values),
            -np.imag(z_values),
            marker="o",
            linewidth=1.2,
            label=_run_label(run),
        )
        plotted += 1

    if plotted == 0:
        raise ValueError("Selected runs contain no impedance rows to plot.")

    ax.set_xlabel("Z' (Ohm)")
    ax.set_ylabel("-Z'' (Ohm)")
    ax.set_title(title or "Nyquist Plot")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig, ax, selected_runs


def plot_impedance_bode(
    *,
    base_output_dir: str | Path,
    selection: RunSelection | None = None,
    aggregate_repeats: bool = True,
    axes: tuple[plt.Axes, plt.Axes] | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes], tuple[RunFolderRecord, ...]]:
    """Plot Bode magnitude/phase overlay for selected run folders.

    Inputs:
        base_output_dir: Root output folder containing measurement run folders.
        selection: Run selection/filter configuration.
        aggregate_repeats: If true, each run contributes one point per frequency
            (mean over repeats). If false, each repeat row is plotted.
        axes: Optional tuple ``(ax_magnitude, ax_phase)``.
        title: Optional figure-level title.
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
    return fig, (ax_mag, ax_phase), selected_runs
