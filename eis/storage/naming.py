"""Naming helpers for measurement folders and files."""

from __future__ import annotations

from datetime import datetime
import re


_SERIAL_ALLOWED_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


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
