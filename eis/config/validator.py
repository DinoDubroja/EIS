"""Validation logic for EIS Excel measurement configuration.

This module converts raw worksheet rows into strict, typed sweep settings and
collects user-facing validation issues with row/column context. The goal is to
fail early with actionable feedback before any DAQ task starts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from eis.config.excel_loader import load_config_table
from eis.models.config_models import (
    ConfigValidationError,
    ConfigValidationIssue,
    MeasurementPointConfig,
    RawConfigRow,
    RawConfigTable,
    SweepConfig,
)

# USB-6451 AO maximum update rate from module docs/manual.
USB6451_MAX_AO_SAMPLE_RATE_SPS = 250_000.0
MIN_SAMPLES_PER_PERIOD = 8.0

_REQUIRED_CANONICAL_COLUMNS = (
    "frequency",
    "ch0_range",
    "ch1_range",
    "sample_rate",
    "n_periods",
    "current_rms",
)

_COLUMN_ALIASES = {
    "frequency": "frequency",
    "freq": "frequency",
    "ch0_range": "ch0_range",
    "ch0range": "ch0_range",
    "ch1_range": "ch1_range",
    "ch1range": "ch1_range",
    "sample_rate": "sample_rate",
    "samplerate": "sample_rate",
    "n_periods": "n_periods",
    "nperiods": "n_periods",
    "periods": "n_periods",
    "current_rms": "current_rms",
    "currentrms": "current_rms",
}


def load_and_validate_config(xlsx_path: str | Path, sheet_name: str | None = None) -> SweepConfig:
    """Load workbook and return fully validated sweep config in one call.

    Inputs:
        xlsx_path: Path to measurement configuration workbook.
        sheet_name: Optional sheet name override. If omitted, first sheet is used by loader defaults.
    Output:
        ``SweepConfig`` ready for acquisition execution.
    Raises:
        ConfigValidationError: Workbook content failed schema/value checks.
    """

    raw_table = load_config_table(xlsx_path=xlsx_path, sheet_name=sheet_name)
    return validate_config_table(raw_table)


def validate_config_table(raw_table: RawConfigTable) -> SweepConfig:
    """Validate raw config table and return normalized sweep configuration.

    Inputs:
        raw_table: Table loaded from Excel.
    Output:
        ``SweepConfig`` with typed rows for measurement sweep execution.
    Raises:
        ConfigValidationError: One or more rows or columns are invalid.
    """

    canonical_header_map, header_issues = _build_canonical_header_map(raw_table.headers)
    if header_issues:
        raise ConfigValidationError(header_issues)

    issues: list[ConfigValidationIssue] = []
    points: list[MeasurementPointConfig] = []

    for row in raw_table.rows:
        point, row_issues = _validate_row(row, canonical_header_map)
        issues.extend(row_issues)
        if point is not None:
            points.append(point)

    if not points:
        issues.append(
            ConfigValidationIssue(
                message="No valid measurement rows were found in the configuration table.",
            )
        )

    if issues:
        raise ConfigValidationError(issues)

    return SweepConfig(
        source_path=raw_table.source_path,
        sheet_name=raw_table.sheet_name,
        points=tuple(points),
    )


def _build_canonical_header_map(
    headers: Iterable[str],
) -> tuple[dict[str, str], list[ConfigValidationIssue]]:
    """Map table headers to canonical names and collect schema-level issues."""

    issues: list[ConfigValidationIssue] = []
    canonical_to_original: dict[str, str] = {}
    for header in headers:
        normalized = _normalize_header(header)
        canonical = _COLUMN_ALIASES.get(normalized)
        if canonical is None:
            continue
        if canonical not in canonical_to_original:
            canonical_to_original[canonical] = header

    for required in _REQUIRED_CANONICAL_COLUMNS:
        if required not in canonical_to_original:
            issues.append(
                ConfigValidationIssue(
                    message=(
                        f"Missing required column '{required}'. "
                        f"Required columns: {', '.join(_REQUIRED_CANONICAL_COLUMNS)}."
                    ),
                    column=required,
                )
            )

    return canonical_to_original, issues


def _validate_row(
    row: RawConfigRow,
    canonical_header_map: dict[str, str],
) -> tuple[MeasurementPointConfig | None, list[ConfigValidationIssue]]:
    """Validate one row and return typed config or collected row issues."""

    issues: list[ConfigValidationIssue] = []

    frequency = _read_positive_float(
        row=row,
        canonical_name="frequency",
        canonical_header_map=canonical_header_map,
        issues=issues,
    )
    ch0_range = _read_positive_float(
        row=row,
        canonical_name="ch0_range",
        canonical_header_map=canonical_header_map,
        issues=issues,
    )
    ch1_range = _read_positive_float(
        row=row,
        canonical_name="ch1_range",
        canonical_header_map=canonical_header_map,
        issues=issues,
    )
    sample_rate = _read_positive_float(
        row=row,
        canonical_name="sample_rate",
        canonical_header_map=canonical_header_map,
        issues=issues,
    )
    current_rms = _read_positive_float(
        row=row,
        canonical_name="current_rms",
        canonical_header_map=canonical_header_map,
        issues=issues,
    )
    n_periods = _read_positive_int(
        row=row,
        canonical_name="n_periods",
        canonical_header_map=canonical_header_map,
        issues=issues,
    )

    if sample_rate is not None and sample_rate > USB6451_MAX_AO_SAMPLE_RATE_SPS:
        header = canonical_header_map["sample_rate"]
        issues.append(
            ConfigValidationIssue(
                row_number=row.row_number,
                column=header,
                message=(
                    f"Sample rate {sample_rate:g} S/s exceeds USB-6451 AO limit "
                    f"{USB6451_MAX_AO_SAMPLE_RATE_SPS:g} S/s."
                ),
            )
        )

    if ch0_range is not None and ch0_range > 10.0:
        header = canonical_header_map["ch0_range"]
        issues.append(
            ConfigValidationIssue(
                row_number=row.row_number,
                column=header,
                message="Range value must be <= 10 V for USB-6451 analog input channels.",
            )
        )

    if ch1_range is not None and ch1_range > 10.0:
        header = canonical_header_map["ch1_range"]
        issues.append(
            ConfigValidationIssue(
                row_number=row.row_number,
                column=header,
                message="Range value must be <= 10 V for USB-6451 analog input channels.",
            )
        )

    if frequency is not None and sample_rate is not None:
        samples_per_period = sample_rate / frequency
        if samples_per_period < MIN_SAMPLES_PER_PERIOD:
            header = canonical_header_map["frequency"]
            issues.append(
                ConfigValidationIssue(
                    row_number=row.row_number,
                    column=header,
                    message=(
                        "Frequency is too high for selected sample rate. "
                        f"Need at least {MIN_SAMPLES_PER_PERIOD:g} samples/period, "
                        f"current value is {samples_per_period:.3f}."
                    ),
                )
            )

    if issues:
        return None, issues

    assert frequency is not None
    assert ch0_range is not None
    assert ch1_range is not None
    assert sample_rate is not None
    assert n_periods is not None
    assert current_rms is not None

    return (
        MeasurementPointConfig(
            row_number=row.row_number,
            frequency_hz=frequency,
            ch0_range_v=ch0_range,
            ch1_range_v=ch1_range,
            sample_rate_sps=sample_rate,
            n_periods=n_periods,
            current_rms=current_rms,
        ),
        issues,
    )


def _normalize_header(header: str) -> str:
    """Normalize header labels to matching key format."""

    return header.strip().lower().replace(" ", "").replace("-", "_")


def _read_positive_float(
    *,
    row: RawConfigRow,
    canonical_name: str,
    canonical_header_map: dict[str, str],
    issues: list[ConfigValidationIssue],
) -> float | None:
    """Read and validate positive float field from a row."""

    header = canonical_header_map[canonical_name]
    raw_value = row.values.get(header)

    if raw_value is None or str(raw_value).strip() == "":
        issues.append(
            ConfigValidationIssue(
                row_number=row.row_number,
                column=header,
                message="Value is required.",
            )
        )
        return None

    try:
        number = float(raw_value)
    except (TypeError, ValueError):
        issues.append(
            ConfigValidationIssue(
                row_number=row.row_number,
                column=header,
                message=f"Expected numeric value, got '{raw_value}'.",
            )
        )
        return None

    if number <= 0:
        issues.append(
            ConfigValidationIssue(
                row_number=row.row_number,
                column=header,
                message="Value must be > 0.",
            )
        )
        return None

    return number


def _read_positive_int(
    *,
    row: RawConfigRow,
    canonical_name: str,
    canonical_header_map: dict[str, str],
    issues: list[ConfigValidationIssue],
) -> int | None:
    """Read and validate positive integer field from a row."""

    number = _read_positive_float(
        row=row,
        canonical_name=canonical_name,
        canonical_header_map=canonical_header_map,
        issues=issues,
    )
    if number is None:
        return None

    rounded = int(round(number))
    header = canonical_header_map[canonical_name]
    if abs(number - rounded) > 1e-9:
        issues.append(
            ConfigValidationIssue(
                row_number=row.row_number,
                column=header,
                message=f"Expected integer value, got {number:g}.",
            )
        )
        return None

    if rounded < 1:
        issues.append(
            ConfigValidationIssue(
                row_number=row.row_number,
                column=header,
                message="Value must be >= 1.",
            )
        )
        return None

    return rounded
