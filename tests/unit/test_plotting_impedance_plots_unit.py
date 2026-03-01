"""Unit tests for Nyquist/Bode plotting with run selection filters."""

from __future__ import annotations

from datetime import datetime
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eis.models.measurement_models import ImpedancePointResult
from eis.plotting import RunSelection, plot_impedance_bode, plot_impedance_nyquist
from eis.storage.folder_layout import create_run_folder_layout
from eis.storage.run_artifacts import write_impedance_summary_mean_std_csv, write_impedance_table_csv


class TestPlottingImpedancePlotsUnit(unittest.TestCase):
    """Checks impedance plotting outputs and selection overlays."""

    def _create_run_with_impedance(self, base: Path, serial: str, dt: datetime) -> Path:
        layout = create_run_folder_layout(
            base_output_dir=base,
            serial_number=serial,
            started_at_local=dt,
        )
        rows = (
            ImpedancePointResult(2, 1, 10.0, 5.0, 1.0, 5.099, 11.3099, "fft"),
            ImpedancePointResult(2, 2, 10.0, 5.1, 1.1, 5.217, 12.168, "fft"),
            ImpedancePointResult(3, 1, 20.0, 4.8, 0.8, 4.866, 9.462, "fft"),
            ImpedancePointResult(3, 2, 20.0, 4.9, 0.9, 4.982, 10.408, "fft"),
        )
        write_impedance_table_csv(results=rows, output_path=layout.impedance / "impedance.csv")
        write_impedance_summary_mean_std_csv(
            results=rows,
            output_path=layout.impedance / "summary_mean_std.csv",
        )
        return layout.root

    def tearDown(self) -> None:
        plt.close("all")

    # Checks Nyquist plot overlays only selected runs.
    def test_plot_impedance_nyquist_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._create_run_with_impedance(base, "SN_A", datetime(2026, 3, 1, 9, 0))
            self._create_run_with_impedance(base, "SN_B", datetime(2026, 3, 1, 10, 0))
            self._create_run_with_impedance(base, "SN_C", datetime(2026, 3, 1, 11, 0))

            fig, ax, selected = plot_impedance_nyquist(
                base_output_dir=base,
                selection=RunSelection(mode="last_n", last_n=2),
            )
            self.assertIsNotNone(fig)
            self.assertEqual(len(selected), 2)
            self.assertEqual(len(ax.lines), 2)

    # Checks Bode plot supports serial and time filters inferred from folder names.
    def test_plot_impedance_bode_serial_and_time_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._create_run_with_impedance(base, "ALPHA", datetime(2026, 3, 1, 9, 0))
            self._create_run_with_impedance(base, "BETA", datetime(2026, 3, 1, 10, 0))
            self._create_run_with_impedance(base, "BETA_X", datetime(2026, 3, 1, 11, 0))

            fig, axes, selected = plot_impedance_bode(
                base_output_dir=base,
                selection=RunSelection(
                    mode="all",
                    serial_contains="beta",
                    started_at_or_before=datetime(2026, 3, 1, 10, 30),
                ),
            )
            self.assertIsNotNone(fig)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].serial_number, "BETA")
            self.assertEqual(len(axes[0].lines), 1)
            self.assertEqual(len(axes[1].lines), 1)


if __name__ == "__main__":
    unittest.main()
