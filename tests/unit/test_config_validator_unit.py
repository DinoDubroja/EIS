"""Unit tests for EIS config validation logic."""

from __future__ import annotations

import unittest
from pathlib import Path

from eis.config.excel_loader import load_config_table
from eis.config.validator import load_and_validate_config, validate_config_table
from eis.models.config_models import (
    ConfigValidationError,
    RawConfigRow,
    RawConfigTable,
)


class TestConfigValidatorUnit(unittest.TestCase):
    """Checks schema and numeric validation for measurement config rows."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.config_path = cls.repo_root / "config_examples" / "config_phase1_example.xlsx"

    # Checks repository config.xlsx validates and returns typed sweep points.
    def test_load_and_validate_config_accepts_repo_config(self) -> None:
        sweep = load_and_validate_config(self.config_path)
        self.assertEqual(sweep.sheet_name, "Sheet1")
        self.assertEqual(len(sweep.points), 14)
        self.assertAlmostEqual(sweep.points[0].frequency_hz, 12.54)
        self.assertEqual(sweep.points[0].n_periods, 20)

    # Checks validator rejects row where sample rate exceeds USB-6451 AO limit.
    def test_validate_config_table_rejects_too_high_sample_rate(self) -> None:
        table = load_config_table(self.config_path)
        first_row = table.rows[0]
        bad_first_values = dict(first_row.values)
        bad_first_values["Sample_rate"] = 300001.0
        bad_rows = (
            RawConfigRow(row_number=first_row.row_number, values=bad_first_values),
            *table.rows[1:],
        )
        bad_table = RawConfigTable(
            source_path=table.source_path,
            sheet_name=table.sheet_name,
            headers=table.headers,
            rows=bad_rows,
        )

        with self.assertRaises(ConfigValidationError) as ctx:
            validate_config_table(bad_table)

        self.assertIn("USB-6451 AO limit", str(ctx.exception))

    # Checks validator rejects non-integer N_periods value.
    def test_validate_config_table_rejects_non_integer_period_count(self) -> None:
        table = load_config_table(self.config_path)
        first_row = table.rows[0]
        bad_first_values = dict(first_row.values)
        bad_first_values["N_periods"] = 20.5
        bad_rows = (
            RawConfigRow(row_number=first_row.row_number, values=bad_first_values),
            *table.rows[1:],
        )
        bad_table = RawConfigTable(
            source_path=table.source_path,
            sheet_name=table.sheet_name,
            headers=table.headers,
            rows=bad_rows,
        )

        with self.assertRaises(ConfigValidationError) as ctx:
            validate_config_table(bad_table)

        self.assertIn("Expected integer value", str(ctx.exception))

    # Checks validator fails when one required column is missing.
    def test_validate_config_table_rejects_missing_required_column(self) -> None:
        headers = ("Frequency", "Ch0_range", "Ch1_range", "Sample_rate", "N_periods")
        rows = (
            RawConfigRow(
                row_number=2,
                values={
                    "Frequency": 100.0,
                    "Ch0_range": 2.5,
                    "Ch1_range": 2.5,
                    "Sample_rate": 250000.0,
                    "N_periods": 20.0,
                },
            ),
        )
        table = RawConfigTable(
            source_path=self.config_path,
            sheet_name="Sheet1",
            headers=headers,
            rows=rows,
        )

        with self.assertRaises(ConfigValidationError) as ctx:
            validate_config_table(table)

        self.assertIn("Missing required column 'current_rms'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
