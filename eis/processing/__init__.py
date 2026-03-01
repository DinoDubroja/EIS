"""Signal processing module namespace.

This package is reserved for core signal processing steps:
- optional filtering and leakage handling
- FFT/sine-fit extraction of amplitude and phase
- impedance computation and (later) uncertainty estimation hooks
"""

from eis.processing.impedance_processor import (
    ImpedanceProcessingConfig,
    compute_impedance_for_capture,
    compute_impedance_for_run,
)

__all__ = [
    "ImpedanceProcessingConfig",
    "compute_impedance_for_capture",
    "compute_impedance_for_run",
]
