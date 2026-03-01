"""Unit tests for Clarke-Hess 8100 transconductance conversion."""

from __future__ import annotations

import math
import unittest

from eis.acquisition.transconductance import compute_drive_amplitude_from_current


class TestAcquisitionTransconductanceUnit(unittest.TestCase):
    """Checks range selection and AO amplitude conversion from current RMS."""

    # Checks nominal conversion on 20A range for 10 Arms target.
    def test_compute_drive_amplitude_nominal(self) -> None:
        result = compute_drive_amplitude_from_current(current_rms_a=10.0)
        self.assertEqual(result.range_name, "20A")
        self.assertAlmostEqual(result.transconductance_siemens, 10.0)
        self.assertFalse(result.is_overrange)
        self.assertAlmostEqual(result.ao_input_vrms, 1.0)
        self.assertAlmostEqual(result.ao_amplitude_v_peak, math.sqrt(2.0))

    # Checks current above 20A full-scale picks 100A range to avoid overrange.
    def test_compute_drive_amplitude_prefers_no_overrange(self) -> None:
        result = compute_drive_amplitude_from_current(current_rms_a=30.0)
        self.assertEqual(result.range_name, "100A")
        self.assertFalse(result.is_overrange)
        self.assertAlmostEqual(result.ao_input_vrms, 0.3)

    # Checks manual range selection is supported when current is in range limits.
    def test_compute_drive_amplitude_manual_range(self) -> None:
        result = compute_drive_amplitude_from_current(
            current_rms_a=3.0,
            manual_range_name="2A",
        )
        self.assertEqual(result.range_name, "2A")
        self.assertTrue(result.is_overrange)
        self.assertAlmostEqual(result.ao_input_vrms, 3.0)

    # Checks invalid current below 2mA range minimum is rejected.
    def test_compute_drive_amplitude_rejects_out_of_supported_range(self) -> None:
        with self.assertRaises(ValueError):
            compute_drive_amplitude_from_current(current_rms_a=1e-4)


if __name__ == "__main__":
    unittest.main()
