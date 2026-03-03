"""Plotting API namespace for notebook and report image generation.

The plotting package exposes reusable functions that can consume either:
- freshly computed in-memory data
- persisted measurement artifacts on disk

Current scope:
- run-selection overlays (Nyquist / inverse Nyquist / Bode / SNR)
- raw-vs-fitted time-domain overlays from RAW csv captures
- capture-level time/FFT debug overlays from in-memory run captures
"""

from eis.plotting.capture_debug_plots import (
    CaptureDebugComponentSummary,
    CaptureDebugPlotResult,
    plot_capture_fft_components,
    plot_capture_time_domain_components,
)
from eis.plotting.impedance_plots import (
    SNRThresholdCheckResult,
    plot_impedance_bode,
    plot_impedance_inverse_nyquist,
    plot_impedance_nyquist,
    plot_snr_vs_frequency,
)
from eis.plotting.raw_fit_plots import (
    ChannelFitSummary,
    RawFitPlotResult,
    infer_frequency_from_raw_path,
    plot_raw_vs_fitted_from_csv,
)
from eis.plotting.run_selection import RunFolderRecord, RunSelection, select_run_folders

__all__ = [
    "CaptureDebugComponentSummary",
    "CaptureDebugPlotResult",
    "ChannelFitSummary",
    "RawFitPlotResult",
    "RunFolderRecord",
    "RunSelection",
    "SNRThresholdCheckResult",
    "infer_frequency_from_raw_path",
    "plot_capture_fft_components",
    "plot_capture_time_domain_components",
    "plot_impedance_bode",
    "plot_impedance_inverse_nyquist",
    "plot_impedance_nyquist",
    "plot_raw_vs_fitted_from_csv",
    "plot_snr_vs_frequency",
    "select_run_folders",
]
