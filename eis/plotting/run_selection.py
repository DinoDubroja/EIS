"""Run-folder discovery and filtering for multi-run notebook plotting.

This module lets notebook users choose which measurement runs are included in
plots without manually selecting folder paths. Filtering is based on:
- run folder timestamp inferred from ``SERIAL_D_M_Y_H_M``
- serial number inferred from same folder name
- selection mode: last / last_n / all
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from eis.storage.naming import parse_run_folder_name


@dataclass(frozen=True)
class RunFolderRecord:
    """Resolved metadata for one measurement run folder."""

    root: Path
    serial_number: str
    started_at_local: datetime


@dataclass(frozen=True)
class RunSelection:
    """Selection and filter options for loading run folders.

    Inputs:
        mode: Run pick strategy:
            - ``"last"``: pick newest one run after filtering.
            - ``"last_n"``: pick newest ``last_n`` runs after filtering.
            - ``"all"``: pick all filtered runs.
        last_n: Number of runs when mode is ``"last_n"``.
        serial_numbers: Optional exact serial list to include.
        serial_contains: Optional case-insensitive substring filter.
        started_at_or_after: Optional inclusive start-time lower bound.
        started_at_or_before: Optional inclusive start-time upper bound.
    """

    mode: str = "last"
    last_n: int = 1
    serial_numbers: tuple[str, ...] | None = None
    serial_contains: str | None = None
    started_at_or_after: datetime | None = None
    started_at_or_before: datetime | None = None


def _normalize_mode(mode: str) -> str:
    """Normalize and validate selection mode."""

    normalized = mode.strip().lower()
    if normalized not in {"last", "last_n", "all"}:
        raise ValueError("RunSelection.mode must be 'last', 'last_n', or 'all'.")
    return normalized


def discover_run_folders(base_output_dir: str | Path) -> tuple[RunFolderRecord, ...]:
    """Discover parseable run folders under base output path.

    Only folders with an ``IMPEDANCE`` subfolder are included.
    """

    base = Path(base_output_dir)
    if not base.exists():
        return tuple()

    records: list[RunFolderRecord] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if not (child / "IMPEDANCE").exists():
            continue
        try:
            serial_number, started_at_local = parse_run_folder_name(child.name)
        except ValueError:
            continue
        records.append(
            RunFolderRecord(
                root=child,
                serial_number=serial_number,
                started_at_local=started_at_local,
            )
        )
    records.sort(key=lambda item: (item.started_at_local, item.root.name))
    return tuple(records)


def filter_run_folders(
    runs: tuple[RunFolderRecord, ...] | list[RunFolderRecord],
    selection: RunSelection,
) -> tuple[RunFolderRecord, ...]:
    """Apply serial/time filters and mode-based reduction."""

    mode = _normalize_mode(selection.mode)
    rows = list(runs)

    if selection.serial_numbers is not None:
        allowed = {value.strip().lower() for value in selection.serial_numbers if value.strip()}
        rows = [item for item in rows if item.serial_number.lower() in allowed]

    if selection.serial_contains is not None and selection.serial_contains.strip():
        needle = selection.serial_contains.strip().lower()
        rows = [item for item in rows if needle in item.serial_number.lower()]

    if selection.started_at_or_after is not None:
        rows = [item for item in rows if item.started_at_local >= selection.started_at_or_after]

    if selection.started_at_or_before is not None:
        rows = [item for item in rows if item.started_at_local <= selection.started_at_or_before]

    rows.sort(key=lambda item: (item.started_at_local, item.root.name))

    if mode == "all":
        return tuple(rows)
    if mode == "last":
        return tuple(rows[-1:]) if rows else tuple()

    if selection.last_n < 1:
        raise ValueError("RunSelection.last_n must be >= 1 when mode='last_n'.")
    return tuple(rows[-selection.last_n :]) if rows else tuple()


def select_run_folders(
    *,
    base_output_dir: str | Path,
    selection: RunSelection | None = None,
) -> tuple[RunFolderRecord, ...]:
    """Discover and filter run folders with one convenience call."""

    effective = selection or RunSelection(mode="last")
    discovered = discover_run_folders(base_output_dir)
    return filter_run_folders(discovered, effective)
