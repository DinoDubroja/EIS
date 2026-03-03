"""Unit tests for capture-level debug plotting helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eis.models.measurement_models import MeasurementCapture, SweepRunResult
from eis.plotting.capture_debug_plots import (
    plot_capture_fft_components,
    plot_capture_time_domain_components,
)
from eis.processing import ImpedanceProcessingConfig


class TestPlottingCaptureDebugPlotsUnit(unittest.TestCase):
    """Checks time-domain and FFT debug plotting for selected components."""

    def tearDown(self) -> None:
        plt.close("all")

    def _build_run_result(self) -> SweepRunResult:
        """Create one synthetic run result with one capture and two channels."""

        sample_rate_sps = 20_000.0
        frequency_hz = 100.0
        n_periods = 20
        sample_count = int(round(n_periods * sample_rate_sps / frequency_hz))
        time_s = np.arange(sample_count, dtype=np.float64) / sample_rate_sps
        omega = 2.0 * np.pi * frequency_hz

        rng = np.random.default_rng(20260303)
        ai0 = 0.08 * np.sin(omega * time_s + 0.10) + rng.normal(0.0, 9e-4, sample_count)
        ai7 = 2.5 * np.sin(omega * time_s + 0.65) + rng.normal(0.0, 1.5e-2, sample_count)

        capture = MeasurementCapture(
            row_number=3,
            repeat_index=1,
            frequency_hz=frequency_hz,
            sample_rate_sps=sample_rate_sps,
            n_periods=n_periods,
            current_rms=10.0,
            ao_amplitude_v_peak=0.2,
            ao_offset_v=0.0,
            current_range_name="20A",
            transconductance_siemens=50.0,
            started_at_utc_iso=datetime.now(timezone.utc).isoformat(),
            duration_s=0.25,
            ai_channels=("ai0", "ai7"),
            ai_range_v=10.0,
            raw_data=np.vstack([ai0, ai7]),
            acquired_periods=n_periods + 1,
            discarded_settle_samples=0,
            periodic_window_start_sample=0,
            periodic_window_samples=sample_count,
        )
        return SweepRunResult(
            started_at_utc_iso=datetime.now(timezone.utc).isoformat(),
            finished_at_utc_iso=datetime.now(timezone.utc).isoformat(),
            repeats=1,
            captures=(capture,),
            preflight=None,
        )

    # Checks time-domain API supports component combinations and file outputs.
    def test_plot_capture_time_domain_components(self) -> None:
        run_result = self._build_run_result()
        processing = ImpedanceProcessingConfig(
            method="fft",
            filter_mode="lowpass",
            lowpass_cutoff_hz=1_500.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            save_png = base / "time_debug.png"
            save_svg = base / "time_debug.svg"
            fig, axes, result = plot_capture_time_domain_components(
                run_result=run_result,
                frequency_hz=100.0,
                repeat_index=1,
                components=("raw", "filtered", "fitted"),
                processing_config=processing,
                print_snr_table=False,
                save_path=save_png,
                save_vector_path=save_svg,
            )

            self.assertIsNotNone(fig)
            self.assertEqual(len(axes), 2)
            self.assertEqual(result.row_number, 3)
            self.assertEqual(result.repeat_index, 1)
            self.assertEqual(result.components, ("raw", "filtered", "fitted"))
            self.assertEqual(len(result.channel_summaries), 6)
            self.assertTrue(save_png.exists())
            self.assertTrue(save_svg.exists())

    # Checks FFT API supports independent component selection and returns summaries.
    def test_plot_capture_fft_components(self) -> None:
        run_result = self._build_run_result()
        with tempfile.TemporaryDirectory() as tmp:
            save_png = Path(tmp) / "fft_debug.png"
            fig, axes, result = plot_capture_fft_components(
                run_result=run_result,
                frequency_hz=100.0,
                repeat_index=1,
                components=("raw", "fitted"),
                print_snr_table=False,
                max_frequency_hz=800.0,
                save_path=save_png,
            )
            self.assertIsNotNone(fig)
            self.assertEqual(len(axes), 2)
            self.assertEqual(result.components, ("raw", "fitted"))
            self.assertEqual(len(result.channel_summaries), 4)
            self.assertTrue(save_png.exists())

    # Checks invalid component name is rejected with clear error.
    def test_plot_capture_components_invalid_component(self) -> None:
        run_result = self._build_run_result()
        with self.assertRaises(ValueError):
            plot_capture_time_domain_components(
                run_result=run_result,
                frequency_hz=100.0,
                repeat_index=1,
                components=("raw", "invalid"),
                print_snr_table=False,
            )


if __name__ == "__main__":
    unittest.main()
