"""Storage module exports for folder layout and metadata/report persistence.

This layer is responsible for:
- creating collision-safe run folder trees
- writing machine-readable metadata bank files
- rendering metadata reports from the bank
- regenerating reports when view files are lost
"""

from eis.storage.folder_layout import RunFolderLayout, create_run_folder_layout
from eis.storage.metadata_writer import (
    build_metadata_bank,
    regenerate_reports_from_bank,
    write_description_file,
    write_metadata_bank_csv,
    write_metadata_bank_txt,
    write_metadata_report_html,
    write_metadata_report_pdf,
)
from eis.storage.naming import build_run_folder_name, sanitize_serial_number

__all__ = [
    "RunFolderLayout",
    "build_metadata_bank",
    "build_run_folder_name",
    "create_run_folder_layout",
    "regenerate_reports_from_bank",
    "sanitize_serial_number",
    "write_description_file",
    "write_metadata_bank_csv",
    "write_metadata_bank_txt",
    "write_metadata_report_html",
    "write_metadata_report_pdf",
]
