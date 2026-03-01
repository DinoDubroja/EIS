"""Unit tests for EIS Excel config loader."""

from __future__ import annotations

import unittest
from pathlib import Path

from eis.config.excel_loader import load_config_table


class TestConfigExcelLoaderUnit(unittest.TestCase):
    """Checks raw Excel table loading behavior for Phase 1 config files."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.config_path = cls.repo_root / "config_examples" / "config_phase1_example.xlsx"

    # Checks loader reads expected headers and row count from repository config.xlsx.
    def test_load_config_table_reads_expected_schema(self) -> None:
        table = load_config_table(self.config_path)
        self.assertEqual(table.sheet_name, "Sheet1")
        self.assertEqual(
            table.headers,
            (
                "Frequency",
                "Ch0_range",
                "Ch1_range",
                "Sample_rate",
                "N_periods",
                "Current_rms",
            ),
        )
        self.assertEqual(len(table.rows), 14)

        first = table.rows[0]
        self.assertEqual(first.row_number, 2)
        self.assertAlmostEqual(float(first.values["Frequency"]), 12.54)
        self.assertAlmostEqual(float(first.values["Sample_rate"]), 250000.0)

    # Checks loader rejects path that does not exist.
    def test_load_config_table_rejects_missing_file(self) -> None:
        missing = self.repo_root / "this_file_should_not_exist.xlsx"
        with self.assertRaises(FileNotFoundError):
            load_config_table(missing)


if __name__ == "__main__":
    unittest.main()
