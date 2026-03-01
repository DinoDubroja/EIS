# EIS Architecture (Phase 1)

## Goal
Phase 1 builds a modular backend for synchronized USB-6451 impedance measurements, with clear boundaries between:
- hardware acquisition
- signal processing
- data storage
- plotting
- notebook UX

## Module boundaries
- `USB6451/`: low-level NI USB-6451 DAQ API.
- `eis/config/`: Excel loading + config validation.
- `eis/acquisition/`: sweep orchestration over USB6451 APIs.
- `eis/processing/`: FFT/fitting/filtering/impedance extraction.
- `eis/storage/`: folder naming, file writing, metadata.
- `eis/plotting/`: plot templates and plotting API.

## Why this split
- Supports testability without hardware for most logic.
- Keeps DAQ-specific code isolated.
- Makes future features (uncertainty, reports, GUI/web) additive.

## Current implementation status
- Completed in this chunk:
  - `eis/config/excel_loader.py`
  - `eis/config/validator.py`
  - config data models in `eis/models/config_models.py`
- Next chunk:
  - acquisition orchestrator and DAQ preflight integration.
