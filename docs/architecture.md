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
- `eis/acquisition/` now includes:
  - `usb6451_adapter.py`: thin adapter around USB6451 low-level API
  - `preflight_check.py`: DAQ connectivity check wrapper
  - `measurement_runner.py`: one-frequency acquisition execution
  - `sweep_controller.py`: sweep loop with repeats + progress callback
  - `transconductance.py`: Clarke-Hess 8100 range selection + `I_rms -> AO amplitude` conversion
- `eis/models/measurement_models.py`: acquisition result/progress models
- `eis/storage/` now includes:
  - `naming.py`: naming helpers (`SERIAL_D_M_Y_H_M`)
  - `folder_layout.py`: collision-safe run folder creation
  - `run_artifacts.py`: repeat-aware RAW/IMPEDANCE persistence
    - one raw file per frequency+repeat
    - one impedance file per frequency+repeat
    - one `summary_mean_std.csv` per frequency for repeat statistics
    - folder loaders to read saved impedance rows from one run or all runs under a base path
  - `metadata_writer.py`: metadata bank + report generation
    - preferred default view: `REPORTS/metadata_report.html`
    - optional PDF view generated from same bank
    - `description.txt` generated only when user provided non-empty description
- Next chunk:
  - processing pipeline (filtering, FFT, fitting, impedance extraction) to replace demo placeholder impedance values in artifact files.
