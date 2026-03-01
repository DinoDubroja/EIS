"""Unit tests for run selection and filtering from folder names."""

from __future__ import annotations

from datetime import datetime
import tempfile
import unittest
from pathlib import Path

from eis.plotting.run_selection import RunSelection, discover_run_folders, select_run_folders
from eis.storage.folder_layout import create_run_folder_layout


class TestPlottingRunSelectionUnit(unittest.TestCase):
    """Checks selection modes and serial/time filter behavior."""

    def _create_run(self, base: Path, serial: str, dt: datetime) -> Path:
        layout = create_run_folder_layout(
            base_output_dir=base,
            serial_number=serial,
            started_at_local=dt,
        )
        return layout.root

    # Checks discovery keeps only parseable run folders with IMPEDANCE structure.
    def test_discover_run_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._create_run(base, "SNA", datetime(2026, 3, 1, 10, 0))
            self._create_run(base, "SNB", datetime(2026, 3, 1, 11, 0))
            (base / "INVALID").mkdir(parents=True, exist_ok=True)

            runs = discover_run_folders(base)
            self.assertEqual(len(runs), 2)
            self.assertEqual(runs[0].serial_number, "SNA")
            self.assertEqual(runs[1].serial_number, "SNB")

    # Checks last/last_n/all mode behavior after filtering.
    def test_select_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._create_run(base, "SN1", datetime(2026, 3, 1, 9, 0))
            self._create_run(base, "SN1", datetime(2026, 3, 1, 10, 0))
            self._create_run(base, "SN2", datetime(2026, 3, 1, 11, 0))

            last = select_run_folders(base_output_dir=base, selection=RunSelection(mode="last"))
            self.assertEqual(len(last), 1)
            self.assertEqual(last[0].serial_number, "SN2")

            last_two = select_run_folders(
                base_output_dir=base,
                selection=RunSelection(mode="last_n", last_n=2),
            )
            self.assertEqual(len(last_two), 2)
            self.assertEqual([item.serial_number for item in last_two], ["SN1", "SN2"])

            all_runs = select_run_folders(base_output_dir=base, selection=RunSelection(mode="all"))
            self.assertEqual(len(all_runs), 3)

    # Checks serial and time filters parsed from run folder names.
    def test_select_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._create_run(base, "ALPHA", datetime(2026, 3, 1, 9, 0))
            self._create_run(base, "BETA", datetime(2026, 3, 1, 10, 0))
            self._create_run(base, "BETA_2", datetime(2026, 3, 1, 11, 0))

            serial_exact = select_run_folders(
                base_output_dir=base,
                selection=RunSelection(mode="all", serial_numbers=("BETA",)),
            )
            self.assertEqual(len(serial_exact), 1)
            self.assertEqual(serial_exact[0].serial_number, "BETA")

            serial_contains = select_run_folders(
                base_output_dir=base,
                selection=RunSelection(mode="all", serial_contains="beta"),
            )
            self.assertEqual(len(serial_contains), 2)

            time_filtered = select_run_folders(
                base_output_dir=base,
                selection=RunSelection(
                    mode="all",
                    started_at_or_after=datetime(2026, 3, 1, 10, 0),
                    started_at_or_before=datetime(2026, 3, 1, 10, 30),
                ),
            )
            self.assertEqual(len(time_filtered), 1)
            self.assertEqual(time_filtered[0].serial_number, "BETA")


if __name__ == "__main__":
    unittest.main()
