"""Data models used by EIS configuration loading and validation.

These models are written for readability in notebooks and for clear error messages
in lab workflows used by technicians and electrical engineers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RawConfigRow:
    """One raw row from the Excel measurement configuration table.

    Inputs:
        row_number: Excel row index (1-based) from the source sheet.
        values: Column name to raw cell value map.
    Output:
        Immutable row object used by the validator.
    """

    row_number: int
    values: Mapping[str, object]


@dataclass(frozen=True)
class RawConfigTable:
    """Raw table extracted from an Excel sheet.

    Inputs:
        source_path: Path to source ``.xlsx`` file.
        sheet_name: Sheet name used for extraction.
        headers: Header labels in the same order as the sheet.
        rows: Data rows below the header.
    Output:
        Immutable table object for validation.
    """

    source_path: Path
    sheet_name: str
    headers: tuple[str, ...]
    rows: tuple[RawConfigRow, ...]


@dataclass(frozen=True)
class MeasurementPointConfig:
    """Validated settings for one impedance measurement frequency point.

    Inputs:
        row_number: Excel row index (1-based) used for traceability.
        frequency_hz: Excitation frequency in hertz (Hz).
        ch0_range_v: Input range magnitude for channel 0 in volts (V).
        ch1_range_v: Input range magnitude for channel 1 in volts (V).
        sample_rate_sps: Shared sample rate in samples/second (S/s).
        n_periods: Number of sine periods to measure.
        current_rms: Requested RMS test current (unit from config file).
    Output:
        Immutable, validated point configuration.
    """

    row_number: int
    frequency_hz: float
    ch0_range_v: float
    ch1_range_v: float
    sample_rate_sps: float
    n_periods: int
    current_rms: float


@dataclass(frozen=True)
class SweepConfig:
    """Validated sweep configuration for a full measurement run.

    Inputs:
        source_path: Source Excel file.
        sheet_name: Parsed sheet name.
        points: Ordered validated frequency points.
    Output:
        Immutable sweep configuration object.
    """

    source_path: Path
    sheet_name: str
    points: tuple[MeasurementPointConfig, ...]

    @property
    def frequencies_hz(self) -> tuple[float, ...]:
        """Return all sweep frequencies in order."""

        return tuple(point.frequency_hz for point in self.points)


@dataclass(frozen=True)
class ConfigValidationIssue:
    """One validation issue collected while checking config data.

    Fields:
        message: Human-readable issue description.
        row_number: Optional source Excel row number for context.
        column: Optional source column header for context.
    """

    message: str
    row_number: int | None = None
    column: str | None = None

    def format_for_user(self) -> str:
        """Format issue text in a concise user-facing style."""

        parts: list[str] = []
        if self.row_number is not None:
            parts.append(f"row {self.row_number}")
        if self.column:
            parts.append(f"column '{self.column}'")

        prefix = ", ".join(parts)
        if not prefix:
            return self.message
        return f"{prefix}: {self.message}"


class ConfigValidationError(ValueError):
    """Validation error carrying one or more explicit config issues.

    This exception preserves structured issue entries while also exposing one
    aggregated text message suitable for CLI/notebook display.
    """

    def __init__(self, issues: Sequence[ConfigValidationIssue]) -> None:
        if not issues:
            raise ValueError("ConfigValidationError requires at least one issue.")

        self.issues = tuple(issues)
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        formatted = "\n".join(f"- {issue.format_for_user()}" for issue in self.issues)
        return f"Configuration validation failed with {len(self.issues)} issue(s):\n{formatted}"
