"""Configuration module exports.

This package provides:
- low-dependency Excel table loading (`.xlsx`)
- strict validation with actionable row/column errors
- typed sweep configuration output for acquisition execution
"""

from eis.config.excel_loader import load_config_table
from eis.config.validator import load_and_validate_config, validate_config_table

__all__ = [
    "load_config_table",
    "load_and_validate_config",
    "validate_config_table",
]
