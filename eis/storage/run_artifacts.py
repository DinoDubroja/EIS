"""Persistence helpers for repeat-aware RAW and consolidated IMPEDANCE artifacts.

This module is the file-output backbone for measurement reproducibility.
It writes:
- one RAW file per sweep point and repeat (organized in point folders)
- one consolidated ``IMPEDANCE/impedance.csv`` for all sweep rows/repeats
- one consolidated ``IMPEDANCE/summary_mean_std.csv`` with per-frequency stats

The same saved files are indexable through metadata linkage records so reports
and future statistics APIs can rebuild views directly from disk artifacts.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np

from eis.models.measurement_models import (
    ImpedancePointResult,
    MeasurementCapture,
    SweepRunResult,
)
from eis.storage.folder_layout import RunFolderLayout
from eis.storage.naming import build_point_folder_name, build_repeat_file_stem


_SAFE_TOKEN_PATTERN = re.compile(r"[^A-Za-z0-9]+")


@dataclass(frozen=True)
class CaptureArtifactRecord:
    """File-link metadata for one capture (one row and one repeat).

    Fields:
        row_number: Source config row number (1-based).
        repeat_index: Repeat index (1-based).
        frequency_hz: Capture frequency in hertz (Hz).
        raw_csv_relpath: Relative path to saved raw csv artifact.
        impedance_csv_relpath: Relative path to consolidated impedance csv when
            impedance rows were persisted for this capture.
    """

    row_number: int
    repeat_index: int
    frequency_hz: float
    raw_csv_relpath: str
    impedance_csv_relpath: str | None


@dataclass(frozen=True)
class PointSummaryArtifactRecord:
    """Link metadata for one per-frequency summary row in summary table.

    Fields:
        row_number: Source config row number.
        frequency_hz: Frequency represented by this summary.
        repeat_count: Number of repeats included in summary statistics.
        summary_csv_relpath: Relative path to summary csv artifact.
    """

    row_number: int
    frequency_hz: float
    repeat_count: int
    summary_csv_relpath: str


@dataclass(frozen=True)
class PersistedRunArtifacts:
    """Container with all saved artifact-link records for one run.

    Fields:
        capture_artifacts: One link record per capture/repeat raw artifact.
        point_summaries: One link record per frequency summary entry.
    """

    capture_artifacts: tuple[CaptureArtifactRecord, ...]
    point_summaries: tuple[PointSummaryArtifactRecord, ...]


def _as_relpath(path: Path, root: Path) -> str:
    """Convert absolute path into POSIX relative path from run root."""

    return path.relative_to(root).as_posix()


def _safe_token(text: str) -> str:
    """Convert free text into a file-name-safe token."""

    cleaned = _SAFE_TOKEN_PATTERN.sub("_", text.strip()).strip("_")
    return cleaned or "unknown"


def _build_raw_channel_suffix(ai_channels: tuple[str, ...]) -> str:
    """Build channel suffix used in RAW file names.

    Example:
        ``("ai0", "ai7") -> "ch1_ai0_ch2_ai7"``
    """

    parts = []
    for index, channel_name in enumerate(ai_channels, start=1):
        parts.append(f"ch{index}_{_safe_token(channel_name)}")
    return "_".join(parts)


def write_raw_capture_csv(
    *,
    capture: MeasurementCapture,
    output_path: str | Path,
) -> Path:
    """Write one raw capture to CSV with sample index and time axis.

    File columns:
    - ``sample_index``
    - ``time_s``
    - one voltage column per AI channel (for example ``ai0_v``, ``ai7_v``)
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if capture.raw_data.ndim != 2:
        raise ValueError("capture.raw_data must be a 2D array (channels, samples).")
    if capture.raw_data.shape[0] != len(capture.ai_channels):
        raise ValueError(
            "capture.raw_data channel count does not match capture.ai_channels length."
        )
    if capture.sample_rate_sps <= 0:
        raise ValueError("capture.sample_rate_sps must be > 0.")

    sample_count = int(capture.raw_data.shape[1])
    headers = ["sample_index", "time_s"] + [f"{name}_v" for name in capture.ai_channels]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for sample_index in range(sample_count):
            row = [
                sample_index,
                f"{sample_index / float(capture.sample_rate_sps):.12g}",
            ]
            for channel_index in range(len(capture.ai_channels)):
                row.append(f"{float(capture.raw_data[channel_index, sample_index]):.12g}")
            writer.writerow(row)

    return path


def write_impedance_table_csv(
    *,
    results: tuple[ImpedancePointResult, ...] | list[ImpedancePointResult],
    output_path: str | Path,
) -> Path:
    """Write consolidated impedance table with one row per frequency/repeat.

    Inputs:
        results: Per-repeat impedance rows for one run.
        output_path: Target ``impedance.csv`` path.
    Output:
        Path of written consolidated impedance table.
    Raises:
        ValueError: No impedance rows were provided.
    """

    if not results:
        raise ValueError("results must contain at least one impedance row.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "row_number",
        "repeat_index",
        "frequency_hz",
        "z_real_ohm",
        "z_imag_ohm",
        "z_magnitude_ohm",
        "z_phase_deg",
        "extraction_method",
        "snr_current_db",
        "snr_voltage_db",
        "notes",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(results, key=lambda item: (item.row_number, item.repeat_index)):
            writer.writerow(
                {
                    "row_number": result.row_number,
                    "repeat_index": result.repeat_index,
                    "frequency_hz": f"{result.frequency_hz:.12g}",
                    "z_real_ohm": f"{result.z_real_ohm:.12g}",
                    "z_imag_ohm": f"{result.z_imag_ohm:.12g}",
                    "z_magnitude_ohm": f"{result.z_magnitude_ohm:.12g}",
                    "z_phase_deg": f"{result.z_phase_deg:.12g}",
                    "extraction_method": result.extraction_method,
                    "snr_current_db": (
                        f"{result.snr_current_db:.12g}"
                        if result.snr_current_db is not None
                        else ""
                    ),
                    "snr_voltage_db": (
                        f"{result.snr_voltage_db:.12g}"
                        if result.snr_voltage_db is not None
                        else ""
                    ),
                    "notes": result.notes or "",
                }
            )

    return path


def _sample_std(values: np.ndarray) -> float:
    """Return sample standard deviation (ddof=1), or 0.0 for one sample."""

    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _optional_mean_std(values: list[float]) -> tuple[str, str]:
    """Return formatted mean/std strings for optional numeric columns."""

    if not values:
        return "", ""
    arr = np.asarray(values, dtype=np.float64)
    return f"{float(np.mean(arr)):.12g}", f"{_sample_std(arr):.12g}"


def write_impedance_summary_mean_std_csv(
    *,
    results: tuple[ImpedancePointResult, ...] | list[ImpedancePointResult],
    output_path: str | Path,
) -> Path:
    """Write consolidated per-frequency repeat summary with mean/std columns.

    The standard deviation is calculated as sample standard deviation (ddof=1)
    when at least two repeats are available.
    """

    if not results:
        raise ValueError("results must contain at least one impedance row.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "row_number",
        "frequency_hz",
        "repeat_count",
        "extraction_methods",
        "z_real_mean_ohm",
        "z_real_std_ohm",
        "z_imag_mean_ohm",
        "z_imag_std_ohm",
        "z_magnitude_mean_ohm",
        "z_magnitude_std_ohm",
        "z_phase_mean_deg",
        "z_phase_std_deg",
        "snr_current_mean_db",
        "snr_current_std_db",
        "snr_voltage_mean_db",
        "snr_voltage_std_db",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        grouped: dict[tuple[int, float], list[ImpedancePointResult]] = defaultdict(list)
        for item in results:
            grouped[(item.row_number, item.frequency_hz)].append(item)

        for (row_number, frequency_hz), point_results in sorted(
            grouped.items(),
            key=lambda item: (item[0][0], item[0][1]),
        ):
            z_real = np.asarray([item.z_real_ohm for item in point_results], dtype=np.float64)
            z_imag = np.asarray([item.z_imag_ohm for item in point_results], dtype=np.float64)
            z_mag = np.asarray([item.z_magnitude_ohm for item in point_results], dtype=np.float64)
            z_phase = np.asarray([item.z_phase_deg for item in point_results], dtype=np.float64)
            snr_current_values = [
                float(item.snr_current_db)
                for item in point_results
                if item.snr_current_db is not None
            ]
            snr_voltage_values = [
                float(item.snr_voltage_db)
                for item in point_results
                if item.snr_voltage_db is not None
            ]
            snr_current_mean, snr_current_std = _optional_mean_std(snr_current_values)
            snr_voltage_mean, snr_voltage_std = _optional_mean_std(snr_voltage_values)
            methods = sorted({item.extraction_method for item in point_results})

            writer.writerow(
                {
                    "row_number": row_number,
                    "frequency_hz": f"{frequency_hz:.12g}",
                    "repeat_count": len(point_results),
                    "extraction_methods": ";".join(methods),
                    "z_real_mean_ohm": f"{float(np.mean(z_real)):.12g}",
                    "z_real_std_ohm": f"{_sample_std(z_real):.12g}",
                    "z_imag_mean_ohm": f"{float(np.mean(z_imag)):.12g}",
                    "z_imag_std_ohm": f"{_sample_std(z_imag):.12g}",
                    "z_magnitude_mean_ohm": f"{float(np.mean(z_mag)):.12g}",
                    "z_magnitude_std_ohm": f"{_sample_std(z_mag):.12g}",
                    "z_phase_mean_deg": f"{float(np.mean(z_phase)):.12g}",
                    "z_phase_std_deg": f"{_sample_std(z_phase):.12g}",
                    "snr_current_mean_db": snr_current_mean,
                    "snr_current_std_db": snr_current_std,
                    "snr_voltage_mean_db": snr_voltage_mean,
                    "snr_voltage_std_db": snr_voltage_std,
                }
            )

    return path


def persist_run_artifacts(
    *,
    layout: RunFolderLayout,
    run_result: SweepRunResult,
    impedance_results: tuple[ImpedancePointResult, ...] | list[ImpedancePointResult] | None = None,
) -> PersistedRunArtifacts:
    """Persist RAW and IMPEDANCE artifacts for a full run.

    Inputs:
        layout: Folder layout created for this run.
        run_result: In-memory run result returned by acquisition controller.
        impedance_results: Optional impedance results keyed by row+repeat.
    Output:
        ``PersistedRunArtifacts`` with metadata-ready relative file links.
    Raises:
        ValueError: Duplicate or unknown impedance result keys are provided.
    """

    impedance_map: dict[tuple[int, int], ImpedancePointResult] = {}
    if impedance_results is not None:
        for result in impedance_results:
            key = (result.row_number, result.repeat_index)
            if key in impedance_map:
                raise ValueError(
                    "Duplicate impedance result for row/repeat key "
                    f"{key}. Each capture must map to at most one result."
                )
            impedance_map[key] = result

    capture_keys = {(capture.row_number, capture.repeat_index) for capture in run_result.captures}
    unknown_keys = set(impedance_map) - capture_keys
    if unknown_keys:
        raise ValueError(
            "Impedance results contain keys not present in run_result captures: "
            f"{sorted(unknown_keys)}"
        )

    impedance_table_path: Path | None = None
    summary_table_path: Path | None = None
    impedance_table_relpath: str | None = None
    summary_table_relpath: str | None = None
    if impedance_map:
        impedance_table_path = layout.impedance / "impedance.csv"
        summary_table_path = layout.impedance / "summary_mean_std.csv"
        impedance_table_relpath = _as_relpath(impedance_table_path, layout.root)
        summary_table_relpath = _as_relpath(summary_table_path, layout.root)

    capture_records: list[CaptureArtifactRecord] = []
    ordered_impedance_rows: list[ImpedancePointResult] = []
    grouped_repeats: dict[tuple[int, float], int] = defaultdict(int)

    for capture in run_result.captures:
        point_folder_name = build_point_folder_name(capture.row_number, capture.frequency_hz)
        repeat_stem = build_repeat_file_stem(capture.repeat_index)
        channel_suffix = _build_raw_channel_suffix(capture.ai_channels)

        raw_path = layout.raw / point_folder_name / f"{repeat_stem}_raw_{channel_suffix}.csv"
        write_raw_capture_csv(capture=capture, output_path=raw_path)

        key = (capture.row_number, capture.repeat_index)
        impedance_result = impedance_map.get(key)
        if impedance_result is not None:
            ordered_impedance_rows.append(impedance_result)
            grouped_repeats[(capture.row_number, capture.frequency_hz)] += 1

        capture_records.append(
            CaptureArtifactRecord(
                row_number=capture.row_number,
                repeat_index=capture.repeat_index,
                frequency_hz=capture.frequency_hz,
                raw_csv_relpath=_as_relpath(raw_path, layout.root),
                impedance_csv_relpath=(
                    impedance_table_relpath if impedance_result is not None else None
                ),
            )
        )

    point_summaries: list[PointSummaryArtifactRecord] = []
    if ordered_impedance_rows:
        write_impedance_table_csv(results=ordered_impedance_rows, output_path=impedance_table_path)
        write_impedance_summary_mean_std_csv(
            results=ordered_impedance_rows,
            output_path=summary_table_path,
        )
        for (row_number, frequency_hz), repeat_count in sorted(
            grouped_repeats.items(),
            key=lambda item: (item[0][0], item[0][1]),
        ):
            point_summaries.append(
                PointSummaryArtifactRecord(
                    row_number=row_number,
                    frequency_hz=frequency_hz,
                    repeat_count=repeat_count,
                    summary_csv_relpath=str(summary_table_relpath),
                )
            )

    return PersistedRunArtifacts(
        capture_artifacts=tuple(capture_records),
        point_summaries=tuple(point_summaries),
    )


def build_artifact_link_payload(
    artifacts: PersistedRunArtifacts,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert typed artifact records into metadata-bank payload tables.

    Inputs:
        artifacts: Typed run artifact linkage object.
    Output:
        Tuple ``(capture_rows, point_summary_rows)`` ready for metadata bank.
    """

    capture_rows = [
        {
            "row_number": item.row_number,
            "repeat_index": item.repeat_index,
            "frequency_hz": item.frequency_hz,
            "raw_csv_relpath": item.raw_csv_relpath,
            "impedance_csv_relpath": item.impedance_csv_relpath,
        }
        for item in artifacts.capture_artifacts
    ]
    point_summary_rows = [
        {
            "row_number": item.row_number,
            "frequency_hz": item.frequency_hz,
            "repeat_count": item.repeat_count,
            "summary_csv_relpath": item.summary_csv_relpath,
        }
        for item in artifacts.point_summaries
    ]
    return capture_rows, point_summary_rows


def _parse_impedance_csv_row(row: dict[str, str]) -> dict[str, Any]:
    """Parse one impedance CSV row back into typed values."""

    return {
        "row_number": int(row["row_number"]),
        "repeat_index": int(row["repeat_index"]),
        "frequency_hz": float(row["frequency_hz"]),
        "z_real_ohm": float(row["z_real_ohm"]),
        "z_imag_ohm": float(row["z_imag_ohm"]),
        "z_magnitude_ohm": float(row["z_magnitude_ohm"]),
        "z_phase_deg": float(row["z_phase_deg"]),
        "extraction_method": row["extraction_method"],
        "snr_current_db": float(row["snr_current_db"]) if row.get("snr_current_db") else None,
        "snr_voltage_db": float(row["snr_voltage_db"]) if row.get("snr_voltage_db") else None,
        "notes": row["notes"] or None,
    }


def load_impedance_rows_from_run(run_root: str | Path) -> list[dict[str, Any]]:
    """Load impedance rows from one run folder.

    Preferred format:
        ``IMPEDANCE/impedance.csv`` (one table for all rows/repeats).
    Legacy fallback:
        ``IMPEDANCE/row_*/repeat_*_impedance.csv``.
    """

    root = Path(run_root)
    impedance_root = root / "IMPEDANCE"
    if not impedance_root.exists():
        return []

    rows: list[dict[str, Any]] = []

    consolidated_path = impedance_root / "impedance.csv"
    if consolidated_path.exists():
        file_paths = [consolidated_path]
    else:
        file_paths = sorted(impedance_root.glob("row_*_f*Hz/repeat_*_impedance.csv"))

    for file_path in file_paths:
        with file_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parsed = _parse_impedance_csv_row(row)
                parsed["run_folder"] = root.name
                parsed["source_relpath"] = file_path.relative_to(root).as_posix()
                rows.append(parsed)
    return rows


def load_impedance_rows_from_base(base_output_dir: str | Path) -> list[dict[str, Any]]:
    """Load per-repeat impedance rows from all run folders under base directory.

    Inputs:
        base_output_dir: Root output directory containing run folders.
    Output:
        Flat list of parsed impedance rows from all discoverable run folders.
    """

    base = Path(base_output_dir)
    if not base.exists():
        return []

    all_rows: list[dict[str, Any]] = []
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "IMPEDANCE").exists():
            continue
        all_rows.extend(load_impedance_rows_from_run(run_dir))
    return all_rows
