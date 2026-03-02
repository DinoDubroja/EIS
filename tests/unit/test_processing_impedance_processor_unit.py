"""Unit tests for FFT/sine-fit impedance processing from raw captures."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest

import numpy as np

from eis.models.measurement_models import MeasurementCapture, SweepRunResult
from eis.processing.impedance_processor import (
    ImpedanceProcessingConfig,
    compute_impedance_for_capture,
    compute_impedance_for_run,
)


class TestProcessingImpedanceProcessorUnit(unittest.TestCase):
    """Checks impedance extraction accuracy and configuration behavior."""

    @staticmethod
    def _build_capture(
        *,
        z_target: complex = complex(5.0, 2.0),
        frequency_hz: float = 100.0,
        sample_rate_sps: float = 20_000.0,
        n_periods: int = 30,
        r_shunt_ohm: float = 0.008,
        noise_scale: float = 0.0,
    ) -> MeasurementCapture:
        sample_count = int(round(n_periods * sample_rate_sps / frequency_hz))
        t = np.arange(sample_count, dtype=np.float64) / sample_rate_sps
        omega = 2.0 * np.pi * frequency_hz

        i_peak = 1.8
        i_phase = 0.22
        i_signal = i_peak * np.sin(omega * t + i_phase)
        v_shunt = r_shunt_ohm * i_signal

        v_complex = (i_peak * np.exp(1j * i_phase)) * z_target
        v_peak = float(abs(v_complex))
        v_phase = float(np.angle(v_complex))
        v_dut = v_peak * np.sin(omega * t + v_phase)

        if noise_scale > 0:
            v_shunt = v_shunt + noise_scale * np.sin(3.0 * omega * t + 0.11) + 1e-4
            v_dut = v_dut + (10.0 * noise_scale) * np.sin(3.0 * omega * t - 0.3) - 2e-3

        raw = np.vstack([v_shunt, v_dut]).astype(np.float64)
        return MeasurementCapture(
            row_number=2,
            repeat_index=1,
            frequency_hz=frequency_hz,
            sample_rate_sps=sample_rate_sps,
            n_periods=n_periods,
            current_rms=1.0,
            ao_amplitude_v_peak=0.2,
            ao_offset_v=0.0,
            current_range_name="2A",
            transconductance_siemens=1.0,
            started_at_utc_iso=datetime.now(timezone.utc).isoformat(),
            duration_s=0.03,
            ai_channels=("ai0", "ai7"),
            ai_range_v=2.5,
            raw_data=raw,
            acquired_periods=n_periods,
            discarded_settle_samples=0,
            periodic_window_start_sample=0,
            periodic_window_samples=sample_count,
        )

    # Checks FFT path recovers expected complex impedance from ideal data.
    def test_compute_impedance_fft_matches_target(self) -> None:
        z_target = complex(5.0, 2.0)
        capture = self._build_capture(z_target=z_target)
        result = compute_impedance_for_capture(
            capture=capture,
            config=ImpedanceProcessingConfig(method="fft", shunt_resistance_ohm=0.008),
        )
        self.assertAlmostEqual(result.z_real_ohm, z_target.real, places=3)
        self.assertAlmostEqual(result.z_imag_ohm, z_target.imag, places=3)
        self.assertIsNotNone(result.snr_current_db)
        self.assertIsNotNone(result.snr_voltage_db)
        self.assertTrue(float(result.snr_current_db) > 40.0)
        self.assertTrue(float(result.snr_voltage_db) > 40.0)

    # Checks sine-fit numpy backend also recovers impedance on ideal data.
    def test_compute_impedance_sine_fit_numpy_matches_target(self) -> None:
        z_target = complex(3.2, -1.4)
        capture = self._build_capture(z_target=z_target)
        result = compute_impedance_for_capture(
            capture=capture,
            config=ImpedanceProcessingConfig(
                method="sine_fit",
                sine_fit_backend="numpy_lstsq",
                shunt_resistance_ohm=0.008,
            ),
        )
        self.assertAlmostEqual(result.z_real_ohm, z_target.real, places=3)
        self.assertAlmostEqual(result.z_imag_ohm, z_target.imag, places=3)

    # Checks optional pre-filter helps robustness when harmonic noise is present.
    def test_compute_impedance_lowpass_filter_with_noise(self) -> None:
        z_target = complex(6.0, 1.5)
        capture = self._build_capture(z_target=z_target, noise_scale=0.003)
        result = compute_impedance_for_capture(
            capture=capture,
            config=ImpedanceProcessingConfig(
                method="fft",
                filter_mode="lowpass",
                lowpass_cutoff_hz=400.0,
                shunt_resistance_ohm=0.008,
            ),
        )
        self.assertAlmostEqual(result.z_real_ohm, z_target.real, places=2)
        self.assertAlmostEqual(result.z_imag_ohm, z_target.imag, places=2)

    # Checks SNR values decrease when synthetic noise level is increased.
    def test_compute_impedance_snr_reflects_noise_level(self) -> None:
        capture_low_noise = self._build_capture(z_target=complex(5.0, 1.0), noise_scale=0.0002)
        capture_high_noise = self._build_capture(z_target=complex(5.0, 1.0), noise_scale=0.005)

        result_low = compute_impedance_for_capture(
            capture=capture_low_noise,
            config=ImpedanceProcessingConfig(method="fft"),
        )
        result_high = compute_impedance_for_capture(
            capture=capture_high_noise,
            config=ImpedanceProcessingConfig(method="fft"),
        )

        self.assertTrue(float(result_low.snr_current_db) > float(result_high.snr_current_db))
        self.assertTrue(float(result_low.snr_voltage_db) > float(result_high.snr_voltage_db))

    # Checks run-level helper returns one result row per capture.
    def test_compute_impedance_for_run_returns_all_rows(self) -> None:
        capture_a = self._build_capture(z_target=complex(5.0, 2.0))
        capture_b = self._build_capture(z_target=complex(4.0, -1.0))
        run_result = SweepRunResult(
            started_at_utc_iso=datetime.now(timezone.utc).isoformat(),
            finished_at_utc_iso=datetime.now(timezone.utc).isoformat(),
            repeats=1,
            captures=(capture_a, capture_b),
            preflight=None,
        )
        rows = compute_impedance_for_run(
            run_result=run_result,
            config=ImpedanceProcessingConfig(method="fft"),
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].row_number, capture_a.row_number)
        self.assertEqual(rows[1].row_number, capture_b.row_number)


if __name__ == "__main__":
    unittest.main()
