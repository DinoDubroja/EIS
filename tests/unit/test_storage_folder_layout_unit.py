"""Unit tests for run folder naming and collision safety."""

from __future__ import annotations

from datetime import datetime
import tempfile
import unittest
from pathlib import Path

from eis.storage.folder_layout import create_run_folder_layout
from eis.storage.naming import (
    build_point_folder_name,
    build_repeat_file_stem,
    build_run_folder_name,
    format_frequency_token,
    sanitize_serial_number,
)


class TestStorageFolderLayoutUnit(unittest.TestCase):
    """Checks naming format and anti-overwrite behavior for run folders."""

    # Checks serial sanitization keeps safe chars and removes unsupported chars.
    def test_sanitize_serial_number(self) -> None:
        self.assertEqual(sanitize_serial_number(" Z100/N34 "), "Z100_N34")
        self.assertEqual(sanitize_serial_number("A__B"), "A_B")
        with self.assertRaises(ValueError):
            sanitize_serial_number("___")

    # Checks folder name format matches SERIAL_D_M_Y_H_M convention.
    def test_build_run_folder_name_format(self) -> None:
        dt = datetime(2026, 3, 1, 14, 45)
        name = build_run_folder_name("Z100N34", dt)
        self.assertEqual(name, "Z100N34_1_3_2026_14_45")

    # Checks per-point/per-repeat naming used for repeat artifact persistence.
    def test_point_and_repeat_naming_helpers(self) -> None:
        self.assertEqual(format_frequency_token(53.14), "53_14")
        self.assertEqual(build_point_folder_name(7, 53.14), "row_0007_f53_14Hz")
        self.assertEqual(build_repeat_file_stem(3), "repeat_003")
        with self.assertRaises(ValueError):
            build_point_folder_name(0, 10.0)
        with self.assertRaises(ValueError):
            build_repeat_file_stem(0)

    # Checks layout creates expected tree and blocks collisions.
    def test_create_run_folder_layout_collision_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dt = datetime(2026, 3, 1, 14, 45)
            layout = create_run_folder_layout(
                base_output_dir=base,
                serial_number="Z100N34",
                started_at_local=dt,
            )
            self.assertTrue(layout.root.exists())
            self.assertTrue(layout.raw.exists())
            self.assertTrue(layout.plots.exists())
            self.assertTrue(layout.impedance.exists())
            self.assertTrue(layout.reports.exists())

            with self.assertRaises(FileExistsError):
                create_run_folder_layout(
                    base_output_dir=base,
                    serial_number="Z100N34",
                    started_at_local=dt,
                )


if __name__ == "__main__":
    unittest.main()
