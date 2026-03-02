"""Unit tests for raw-vs-fitted plotting helpers."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eis.plotting.raw_fit_plots import infer_frequency_from_raw_path, plot_raw_vs_fitted_from_csv


class TestPlottingRawFitPlotsUnit(unittest.TestCase):
    """Checks raw csv loading, fit overlays, and frequency inference behavior."""

    def tearDown(self) -> None:
        plt.close("all")

    def _write_demo_raw_csv(self, output_path: Path, *, frequency_hz: float) -> Path:
        sample_rate_sps = 4000.0
        sample_count = 1600
        time_s = np.arange(sample_count, dtype=np.float64) / sample_rate_sps
        omega = 2.0 * np.pi * frequency_hz

        rng = np.random.default_rng(20260302)
        ai0 = 0.03 * np.sin(omega * time_s + 0.10) + rng.normal(0.0, 5e-4, sample_count)
        ai7 = 7.0 * np.sin(omega * time_s + 0.55) + rng.normal(0.0, 2e-2, sample_count)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_index", "time_s", "ai0_v", "ai7_v"])
            for sample_index in range(sample_count):
                writer.writerow(
                    [
                        sample_index,
                        f"{time_s[sample_index]:.12g}",
                        f"{ai0[sample_index]:.12g}",
                        f"{ai7[sample_index]:.12g}",
                    ]
                )
        return output_path

    # Checks plotting from explicit frequency writes image and summaries.
    def test_plot_raw_vs_fitted_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            raw_path = self._write_demo_raw_csv(
                base / "RAW" / "row_0002_f50Hz" / "repeat_001_raw_ch1_ai0_ch2_ai7.csv",
                frequency_hz=50.0,
            )
            save_path = base / "PLOTS" / "raw_fit.png"

            fig, axes, result = plot_raw_vs_fitted_from_csv(
                raw_csv_path=raw_path,
                frequency_hz=50.0,
                save_path=save_path,
            )

            self.assertIsNotNone(fig)
            self.assertEqual(len(axes), 2)
            self.assertEqual(len(result.channel_summaries), 2)
            self.assertTrue(save_path.exists())
            self.assertTrue(save_path.stat().st_size > 0)
            self.assertGreater(result.channel_summaries[0].snr_db, 10.0)

    # Checks frequency can be inferred from RAW folder naming convention.
    def test_infer_frequency_from_raw_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "RAW" / "row_0003_f12_54Hz" / "repeat_001_raw_ch1_ai0_ch2_ai7.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("sample_index,time_s,ai0_v\n0,0,0\n", encoding="utf-8")
            frequency_hz = infer_frequency_from_raw_path(path)
            self.assertAlmostEqual(frequency_hz, 12.54, places=6)

    # Checks missing frequency input raises when path does not match naming rule.
    def test_plot_raw_vs_fitted_missing_frequency_and_unparseable_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = self._write_demo_raw_csv(
                Path(tmp) / "RAW" / "custom_folder" / "capture.csv",
                frequency_hz=100.0,
            )
            with self.assertRaises(ValueError):
                plot_raw_vs_fitted_from_csv(raw_csv_path=raw_path, frequency_hz=None)


if __name__ == "__main__":
    unittest.main()
