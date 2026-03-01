"""Top-level EIS backend API exports.

This package-level module exposes the core entry points used by notebooks and
automation scripts:
- configuration load/validation
- synchronized acquisition orchestration
- drive conversion from current RMS to AO amplitude
- impedance processing (FFT / sine-fit)
- storage and metadata/report generation helpers

The intent is to provide a small, stable import surface for users while
keeping implementation details in submodules.
"""

from eis.acquisition import (
    USB6451Adapter,
    compute_drive_amplitude_from_current,
    execute_sweep,
    run_measurement_point,
    run_preflight_check,
)
from eis.config.validator import load_and_validate_config
from eis.models.measurement_models import (
    CaptureConditioningConfig,
    ExcitationConfig,
    HardwareConfig,
    ImpedancePointResult,
)
from eis.processing import (
    ImpedanceProcessingConfig,
    compute_impedance_for_capture,
    compute_impedance_for_run,
)
from eis.storage import (
    build_artifact_link_payload,
    build_metadata_bank,
    create_run_folder_layout,
    load_impedance_rows_from_base,
    load_impedance_rows_from_run,
    persist_run_artifacts,
    regenerate_reports_from_bank,
    write_impedance_table_csv,
    write_impedance_summary_mean_std_csv,
    write_raw_capture_csv,
    write_description_file,
    write_metadata_bank_csv,
    write_metadata_bank_txt,
    write_metadata_report_html,
    write_metadata_report_pdf,
)

__all__ = [
    "ExcitationConfig",
    "HardwareConfig",
    "ImpedancePointResult",
    "CaptureConditioningConfig",
    "ImpedanceProcessingConfig",
    "USB6451Adapter",
    "build_artifact_link_payload",
    "build_metadata_bank",
    "create_run_folder_layout",
    "compute_drive_amplitude_from_current",
    "compute_impedance_for_capture",
    "compute_impedance_for_run",
    "execute_sweep",
    "load_and_validate_config",
    "load_impedance_rows_from_base",
    "load_impedance_rows_from_run",
    "persist_run_artifacts",
    "regenerate_reports_from_bank",
    "run_measurement_point",
    "run_preflight_check",
    "write_impedance_table_csv",
    "write_impedance_summary_mean_std_csv",
    "write_raw_capture_csv",
    "write_description_file",
    "write_metadata_bank_csv",
    "write_metadata_bank_txt",
    "write_metadata_report_html",
    "write_metadata_report_pdf",
]
