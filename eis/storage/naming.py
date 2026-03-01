"""Naming helpers for measurement folders and file-safe identity fields.

Functions in this module enforce the agreed run-folder naming format and
serial-number sanitization to keep output paths robust across environments.
"""

from __future__ import annotations

from datetime import datetime
import re


_SERIAL_ALLOWED_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")
_RUN_FOLDER_PATTERN = re.compile(
    r"^(?P<serial>.+)_(?P<day>\d{1,2})_(?P<month>\d{1,2})_(?P<year>\d{4})_(?P<hour>\d{1,2})_(?P<minute>\d{1,2})$"
)


def sanitize_serial_number(serial_number: str) -> str:
    """Normalize user serial number into filesystem-safe token.

    Inputs:
        serial_number: User-entered impedance serial number.
    Output:
        Sanitized serial token.
    Raises:
        ValueError: Serial number is empty after sanitization.
    """

    cleaned = _SERIAL_ALLOWED_PATTERN.sub("_", serial_number.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError("Serial number must contain at least one letter or digit.")
    return cleaned


def build_run_folder_name(serial_number: str, started_at_local: datetime) -> str:
    """Build default measurement folder name ``SERIAL_D_M_Y_H_M``.

    Inputs:
        serial_number: User serial number string.
        started_at_local: Local timestamp used for folder naming.
    Output:
        Folder name matching project rule from prompt.
    """

    serial = sanitize_serial_number(serial_number)
    return (
        f"{serial}_"
        f"{started_at_local.day}_{started_at_local.month}_{started_at_local.year}_"
        f"{started_at_local.hour}_{started_at_local.minute}"
    )


def parse_run_folder_name(folder_name: str) -> tuple[str, datetime]:
    """Parse run folder name ``SERIAL_D_M_Y_H_M``.

    Inputs:
        folder_name: Run folder name to parse.
    Output:
        Tuple ``(serial_number, started_at_local)``.
    Raises:
        ValueError: Name does not match expected folder format.
    """

    match = _RUN_FOLDER_PATTERN.match(folder_name.strip())
    if match is None:
        raise ValueError(
            "Run folder name must follow SERIAL_D_M_Y_H_M format, "
            f"got: {folder_name!r}"
        )

    serial = match.group("serial")
    started_at = datetime(
        year=int(match.group("year")),
        month=int(match.group("month")),
        day=int(match.group("day")),
        hour=int(match.group("hour")),
        minute=int(match.group("minute")),
    )
    return serial, started_at


def format_frequency_token(frequency_hz: float) -> str:
    """Format frequency into a file-safe token used in folder names.

    Inputs:
        frequency_hz: Frequency in hertz (Hz).
    Output:
        Token where decimal separator is ``_``.
    Example:
        ``53.14 -> "53_14"``
    """

    base = f"{float(frequency_hz):.6f}".rstrip("0").rstrip(".")
    return base.replace("-", "m").replace(".", "_")


def build_point_folder_name(row_number: int, frequency_hz: float) -> str:
    """Build point folder name ``row_NNNN_fTOKENHz``.

    Inputs:
        row_number: Config row number (1-based).
        frequency_hz: Frequency in hertz (Hz).
    Output:
        Folder name for one sweep point.
    """

    if row_number < 1:
        raise ValueError("row_number must be >= 1.")
    token = format_frequency_token(frequency_hz)
    return f"row_{row_number:04d}_f{token}Hz"


def build_repeat_file_stem(repeat_index: int) -> str:
    """Build repeat token ``repeat_NNN`` used in file names.

    Inputs:
        repeat_index: Repeat number (1-based).
    Output:
        Repeat token for files.
    """

    if repeat_index < 1:
        raise ValueError("repeat_index must be >= 1.")
    return f"repeat_{repeat_index:03d}"
