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
  - includes `CaptureConditioningConfig` for settling/leakage control
- `eis/storage/` now includes:
  - `naming.py`: naming helpers (`SERIAL_D_M_Y_H_M`)
  - `folder_layout.py`: collision-safe run folder creation
  - `run_artifacts.py`: repeat-aware RAW/IMPEDANCE persistence
    - one raw file per frequency+repeat
    - one consolidated `IMPEDANCE/impedance.csv` with all frequencies + repeats
    - one consolidated `IMPEDANCE/summary_mean_std.csv` with per-frequency repeat statistics
    - includes per-row and per-frequency SNR fields for current and voltage channels
    - folder loaders to read saved impedance rows from one run or all runs under a base path
  - `metadata_writer.py`: metadata bank + report generation
    - preferred default view: `REPORTS/metadata_report.html`
    - optional PDF view generated from same bank
    - `description.txt` generated only when user provided non-empty description
- `eis/processing/` now includes:
  - `impedance_processor.py`: processing pipeline for:
    - optional filtering (`none`/`lowpass`/`bandpass`)
    - extraction method selection (`fft` or `sine_fit`)
    - sine-fit backend selection (`numpy_lstsq` or `scipy_least_squares`)
    - nominal shunt conversion (`R_shunt = 0.008 ohm`) and complex impedance output
    - SNR estimation per channel and per frequency
- `eis/plotting/` now includes:
  - `run_selection.py`: run discovery + filters from folder names
    - modes: `last`, `last_n`, `all`
    - filters: serial exact/contains, start-time range
  - `impedance_plots.py`: Nyquist/Bode overlays over selected runs
    from persisted `IMPEDANCE/impedance.csv` artifacts
- Next chunk:
  - raw-vs-fit plotting and higher-level uncertainty analysis/report layering.
