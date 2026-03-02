# Project Rules And Decisions

Last updated: 2026-03-02

## 1) Scope and precedence
- This file consolidates:
  - Rules from the initial prompt document: `Eis Impedance Calibration Api Prompt V2.docx`.
  - Additional rules and decisions agreed in this chat.
- When two rules conflict, newer chat decisions override older prompt text.

## 2) Workflow and collaboration rules
- Work in chunks/phases.
- Before each implementation pass, re-read the initial `.docx` prompt.
- If requirements are unclear or preference-based, ask follow-up questions.
- Before implementing any change, provide plan summary and ask permission to proceed.
- If restructuring is needed, explain the proposed restructure and ask permission first.
- Do not perform destructive changes without explicit permission.
- Prioritize modularity, expandability, robustness, and documentation clarity over speed.
- Suggest better implementation when a requested approach is technically weak, with reasoning.
- Propose next steps whenever a chunk is completed.
- Do not invent new schemas/requirements not stated in prompt or later decisions.
- Explain advanced syntax/conventions and computation-time optimizations in practical terms when needed.

## 3) Repo and git rules
- Keep work on `dev` branch.
- Handle regular commits as part of normal work; do not push.
- Help with merge-to-main only when requested.
- Maintain repo hygiene:
  - documentation
  - folder structure
  - naming conventions
  - README updates
- Keep measurement outputs out of git tracking (`measurements/` remains local-only).

## 4) Documentation and naming rules
- Write docs/comments/docstrings for technicians and electrical engineers, not only software developers.
- Use descriptive module headers (not only one-line headers when more context is needed).
- Keep naming descriptive and consistent across modules.
- Continue periodic strict standardization passes for docstrings/comment quality.

## 5) Architecture and module placement rules
- Primary hardware target is **NI USB-6451**.
- Phase 1 implementation belongs to the global `eis/` package (not packed under `USB6451/`).
- `USB6451/` remains the low-level DAQ layer and may be extended as needed.
- Use NI reference materials and provided manuals when implementing low-level DAQ behavior.

## 6) Measurement and acquisition rules
- Acquisition and signal generation must be synchronous.
- Configuration is driven by Excel `.xlsx`, with automatic validation.
- Run a preflight DAQ connection check before measurement.
- Preflight is current-target based:
  - set test current
  - compute AO DC level from transconductance/range
  - validate expected shunt voltage on current channel (single overall pass/fail)
  - tolerance is defined as percent of expected shunt voltage
- Support repeated measurements per frequency.
- Show progress while sweep is running.
- Support non-integer-divisor frequencies relative to sample rate.

## 7) Signal processing rules
- Support selectable extraction methods:
  - FFT-based
  - sine-fit
- Sine-fit implementation should support both backends, with clear/easy switching:
  - `numpy` least-squares
  - `scipy.optimize` backend
- Support optional filtering before extraction:
  - lowpass
  - bandpass
- Include leakage-control option:
  - acquire margin (for example N+1 periods)
  - trim/select periodic window for minimal edge discontinuity
- Implement startup-settling handling:
  - discard fixed initial settling interval
  - analyze only stable segment
- Shunt handling:
  - nominal `R_shunt = 8 mOhm` in Phase 1
  - frequency-dependent uncertainty handling deferred to later uncertainty/report phase

## 8) Storage, naming, and safety rules
- Default run folder naming: `SERIAL_D_M_Y_H_M`.
- User specifies a global output root; each measurement run creates a new run folder.
- If target run folder name already exists (same serial and minute collision), do not start measurement and show clear warning.
- File/folder names must be descriptive.
- Raw storage can remain organized per point/repeat for cleanliness.
- Raw file names should include both channel identifiers (`ch1` and `ch2`) where applicable.
- Impedance results must be consolidated:
  - one `IMPEDANCE/impedance.csv` containing all frequencies/rows/repeats
  - include frequency column(s) in consolidated table

## 9) Metadata and report rules
- Metadata output should support machine-readable "data bank" regeneration workflow.
- Preferred report presentation format is HTML in `REPORTS/`.
- `description.txt` must be generated only when description is provided (no empty description file).
- Metadata/report artifacts should enable report regeneration if report files are deleted.
- Phase 1 keeps uncertainty architecture expandable; full uncertainty/report logic continues later.
- Metadata bank should preserve enough information to regenerate report outputs.

## 10) Plotting rules
- Plot APIs must work for:
  - newly acquired in-memory data
  - data loaded from persisted run folders
- Selection/filtering in notebook workflows must support:
  - `last`
  - `last_n`
  - `all`
  - serial-based filters
  - time-based filters
- Save plots under `PLOTS/` folder for run outputs.
- Nyquist labels:
  - standard Nyquist axes titles: `R`, `X`
  - also provide inverse Nyquist with imaginary axis flipped and title `-X`
- SNR plotting:
  - support SNR vs frequency
  - support threshold checking across all frequencies
  - include transparent region highlighting for threshold sides
- Raw-vs-fitted style rule:
  - current traces: dark red
  - voltage traces: dark blue
- Support vector export for plots where possible (for example `.svg`), in addition to raster output.

## 11) Testing and demo rules
- Maintain tests in `tests/` and runnable demos in `demo_tests/`.
- Document what each test/demo validates.
- Demos should reflect realistic frequency coverage (use many points like config file where applicable).
- Include synthetic-noise demos where requested (for example SNR and raw-vs-fitted behavior).
- Validate and test new behavior as part of each implementation chunk.

## 12) Uncertainty and statistics roadmap rules
- Type A uncertainty workflow will rely on repeated measurements and folder-level statistics APIs.
- Future API should support statistical analysis over user-specified measurement folders.
- Mechanical re-fixturing/unplug-replug contributions are part of later uncertainty workflow and should be supported by architecture.
- Type B uncertainty integration will be added later with report-generation phase.

## 13) Clarified/superseded items
- Initial prompt referenced NI USB-6351, but project target is now explicitly NI USB-6451.
- Initial prompt said report generation is future work; current agreed scope includes metadata HTML report generation in `REPORTS/`.
- Initial prompt allowed `.txt`/`.xlsx` metadata; current workflow uses metadata-bank files for regeneration plus HTML report as preferred human-facing output.

## 14) Explicit prompt checklist (condensed)
- Measurement flow:
  - generate sine/current stimulus
  - measure N periods
  - analyze and save
  - repeat across frequency list
- User-selectable processing path:
  - FFT extraction
  - sine-fitting extraction
  - optional lowpass/bandpass before extraction
- Leakage-control option must exist (N+1/trim or equivalent periodic-window approach).
- User-selectable saved artifacts include:
  - raw DAQ csv
  - impedance csv
  - uncertainty csv (later full implementation)
  - plot images
  - metadata bank/report artifacts
- Default folder structure includes `RAW/`, `PLOTS/`, `IMPEDANCE/`, `REPORTS/`.
- File names must remain descriptive and traceable.
- Data-safety behavior:
  - if destination run folder already exists, block acquisition and warn clearly.
- Notebook Phase 1 goal:
  - load config
  - choose what to save
  - run measurement
  - plot raw-vs-fit and Nyquist/Bode from resulting data
