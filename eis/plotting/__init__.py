"""Plotting module namespace.

This package is reserved for Phase 1/2 plotting APIs:
- raw-vs-fit time-domain plots
- Nyquist and Bode impedance views
- reusable style templates for report and notebook consistency
"""

from eis.plotting.impedance_plots import (
    plot_impedance_bode,
    plot_impedance_inverse_nyquist,
    plot_impedance_nyquist,
)
from eis.plotting.run_selection import RunFolderRecord, RunSelection, select_run_folders

__all__ = [
    "RunFolderRecord",
    "RunSelection",
    "plot_impedance_bode",
    "plot_impedance_inverse_nyquist",
    "plot_impedance_nyquist",
    "select_run_folders",
]
