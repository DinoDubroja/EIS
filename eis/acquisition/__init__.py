"""Acquisition module exports for synchronized measurement execution.

Scope:
- DAQ preflight connectivity checks
- single-point synchronized acquisition
- full sweep execution with repeats and progress callbacks
- transconductance-based drive conversion integration

These exports are intentionally high-level so notebooks can run sweeps without
direct interaction with low-level USB6451 task plumbing.
"""

from eis.acquisition.measurement_runner import run_measurement_point
from eis.acquisition.preflight_check import run_preflight_check
from eis.acquisition.sweep_controller import execute_sweep
from eis.acquisition.transconductance import compute_drive_amplitude_from_current
from eis.acquisition.usb6451_adapter import USB6451Adapter

__all__ = [
    "USB6451Adapter",
    "compute_drive_amplitude_from_current",
    "execute_sweep",
    "run_measurement_point",
    "run_preflight_check",
]
