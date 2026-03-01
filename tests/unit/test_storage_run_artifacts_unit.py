"""Unit tests for repeat-aware RAW/IMPEDANCE artifact persistence.

These tests verify that:
- one file is written per repeat for RAW outputs
- consolidated ``impedance.csv`` and ``summary_mean_std.csv`` are generated
- saved impedance files can be loaded back from one run or many run folders
"""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eis.models.measurement_models import ImpedancePointResult, MeasurementCapture, SweepRunResult
from eis.storage.folder_layout import create_run_folder_layout
from eis.storage.run_artifacts import (
    load_impedance_rows_from_base,
    load_impedance_rows_from_run,
    persist_run_artifacts,
)


class TestStorageRunArtifactsUnit(unittest.TestCase):
    """Checks run artifact writing and loader behavior."""

    def _build_run_result(self) -> SweepRunResult:
        captures: list[MeasurementCapture] = []
        for row_number, frequency_hz in ((2, 10.0), (3, 20.0)):
            for repeat_index in (1, 2):
                captures.append(
                    MeasurementCapture(
                        row_number=row_number,
                        repeat_index=repeat_index,
                        frequency_hz=frequency_hz,
                        sample_rate_sps=1000.0,
                        n_periods=4,
                        current_rms=1.0,
                        ao_amplitude_v_peak=0.2,
                        ao_offset_v=0.0,
                        current_range_name="2A",
                        transconductance_siemens=1.0,
                        started_at_utc_iso=datetime.now(timezone.utc).isoformat(),
                        duration_s=0.05,
                        ai_channels=("ai0", "ai7"),
                        ai_range_v=2.5,
                        raw_data=np.array(
                            [
                                [0.1, 0.2, 0.1, -0.1],
                                [0.3, 0.4, 0.2, -0.2],
                            ],
                            dtype=np.float64,
                        ),
                    )
                )
        return SweepRunResult(
            started_at_utc_iso=datetime.now(timezone.utc).isoformat(),
            finished_at_utc_iso=datetime.now(timezone.utc).isoformat(),
            repeats=2,
            captures=tuple(captures),
            preflight=None,
        )

    def _build_impedance_results(self) -> tuple[ImpedancePointResult, ...]:
        return (
            ImpedancePointResult(2, 1, 10.0, 100.0, -5.0, 100.12492197, -2.862405226, "fft"),
            ImpedancePointResult(2, 2, 10.0, 102.0, -4.0, 102.07840124, -2.245742565, "fft"),
            ImpedancePointResult(3, 1, 20.0, 80.0, -10.0, 80.62257748, -7.125016349, "fft"),
            ImpedancePointResult(3, 2, 20.0, 81.0, -9.0, 81.49846624, -6.340191746, "fft"),
        )

    # Checks RAW + IMPEDANCE files and summary_mean_std generation for repeats.
    def test_persist_run_artifacts_with_impedance(self) -> None:
        run_result = self._build_run_result()
        impedance_results = self._build_impedance_results()

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            layout = create_run_folder_layout(
                base_output_dir=base,
                serial_number="Z100N34",
                started_at_local=datetime(2026, 3, 1, 14, 45),
            )

            persisted = persist_run_artifacts(
                layout=layout,
                run_result=run_result,
                impedance_results=impedance_results,
            )

            self.assertEqual(len(persisted.capture_artifacts), 4)
            self.assertEqual(len(persisted.point_summaries), 2)

            for item in persisted.capture_artifacts:
                self.assertTrue((layout.root / item.raw_csv_relpath).exists())
                self.assertIn("_raw_ch1_ai0_ch2_ai7.csv", item.raw_csv_relpath)
                self.assertIsNotNone(item.impedance_csv_relpath)
                self.assertTrue((layout.root / str(item.impedance_csv_relpath)).exists())
                self.assertEqual(str(item.impedance_csv_relpath), "IMPEDANCE/impedance.csv")

            for summary in persisted.point_summaries:
                self.assertTrue((layout.root / summary.summary_csv_relpath).exists())
                self.assertEqual(summary.summary_csv_relpath, "IMPEDANCE/summary_mean_std.csv")

            summary_table = layout.impedance / "summary_mean_std.csv"
            with summary_table.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            row_2 = next(row for row in rows if int(row["row_number"]) == 2)
            self.assertEqual(int(row_2["repeat_count"]), 2)
            self.assertAlmostEqual(float(row_2["z_real_mean_ohm"]), 101.0, places=9)
            self.assertAlmostEqual(float(row_2["z_real_std_ohm"]), np.sqrt(2.0), places=9)

            impedance_table = layout.impedance / "impedance.csv"
            with impedance_table.open("r", newline="", encoding="utf-8") as handle:
                impedance_rows = list(csv.DictReader(handle))
            self.assertEqual(len(impedance_rows), 4)
            self.assertIn("frequency_hz", impedance_rows[0])

            run_rows = load_impedance_rows_from_run(layout.root)
            self.assertEqual(len(run_rows), 4)
            self.assertEqual(run_rows[0]["run_folder"], layout.root.name)

            base_rows = load_impedance_rows_from_base(base)
            self.assertEqual(len(base_rows), 4)

    # Checks function also works in raw-only mode before impedance processing exists.
    def test_persist_run_artifacts_raw_only(self) -> None:
        run_result = self._build_run_result()

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            layout = create_run_folder_layout(
                base_output_dir=base,
                serial_number="Z100N34",
                started_at_local=datetime(2026, 3, 1, 14, 46),
            )

            persisted = persist_run_artifacts(
                layout=layout,
                run_result=run_result,
                impedance_results=None,
            )

            self.assertEqual(len(persisted.capture_artifacts), 4)
            self.assertEqual(len(persisted.point_summaries), 0)
            for item in persisted.capture_artifacts:
                self.assertTrue((layout.root / item.raw_csv_relpath).exists())
                self.assertIsNone(item.impedance_csv_relpath)

    # Checks mismatch protection for impedance rows not present in run captures.
    def test_persist_run_artifacts_rejects_unknown_impedance_keys(self) -> None:
        run_result = self._build_run_result()
        bad_results = (
            ImpedancePointResult(999, 1, 1.0, 1.0, 0.0, 1.0, 0.0, "fft"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            layout = create_run_folder_layout(
                base_output_dir=base,
                serial_number="Z100N34",
                started_at_local=datetime(2026, 3, 1, 14, 47),
            )
            with self.assertRaises(ValueError):
                persist_run_artifacts(
                    layout=layout,
                    run_result=run_result,
                    impedance_results=bad_results,
                )


if __name__ == "__main__":
    unittest.main()
