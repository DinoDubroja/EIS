"""Run folder layout and anti-overwrite creation helpers.

This module creates the default measurement output tree and enforces the data
safety rule: if a run folder name already exists, acquisition is blocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from eis.storage.naming import build_run_folder_name


@dataclass(frozen=True)
class RunFolderLayout:
    """Resolved absolute folder paths for one created measurement run.

    Fields:
        root: Run root folder (``SERIAL_D_M_Y_H_M``).
        raw: Folder containing per-point/per-repeat RAW csv captures.
        plots: Folder containing generated plot images.
        impedance: Folder containing consolidated impedance tables.
        reports: Folder containing generated reports (HTML/PDF).
    """

    root: Path
    raw: Path
    plots: Path
    impedance: Path
    reports: Path


def create_run_folder_layout(
    *,
    base_output_dir: str | Path,
    serial_number: str,
    started_at_local: datetime,
) -> RunFolderLayout:
    """Create run folder tree and return resolved paths.

    Inputs:
        base_output_dir: Root directory where measurement folders are created.
        serial_number: User-entered impedance serial number.
        started_at_local: Local measurement start time for folder naming.
    Output:
        ``RunFolderLayout`` with created paths.
    Raises:
        FileExistsError: Folder with target run name already exists.
    """

    base = Path(base_output_dir)
    base.mkdir(parents=True, exist_ok=True)

    folder_name = build_run_folder_name(serial_number, started_at_local)
    run_root = base / folder_name
    if run_root.exists():
        raise FileExistsError(
            "Measurement folder already exists, acquisition is blocked to avoid overwriting data: "
            f"{run_root}"
        )

    raw = run_root / "RAW"
    plots = run_root / "PLOTS"
    impedance = run_root / "IMPEDANCE"
    reports = run_root / "REPORTS"

    for path in (run_root, raw, plots, impedance, reports):
        path.mkdir(parents=True, exist_ok=False)

    return RunFolderLayout(
        root=run_root,
        raw=raw,
        plots=plots,
        impedance=impedance,
        reports=reports,
    )
