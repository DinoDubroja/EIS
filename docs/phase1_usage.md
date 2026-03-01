# Phase 1 Usage (Current)

## Validate config file
```python
from eis import load_and_validate_config

sweep = load_and_validate_config("config_examples/config_phase1_example.xlsx")
print(len(sweep.points))
print(sweep.points[0])
```

## What is validated now
- Required columns:
  - `Frequency`
  - `Ch0_range`
  - `Ch1_range`
  - `Sample_rate`
  - `N_periods`
  - `Current_rms`
- Positive numeric values.
- `N_periods` must be integer.
- `Sample_rate` must not exceed USB-6451 AO limit (`250000 S/s`).
- At least 8 samples per period (`Sample_rate / Frequency >= 8`).

## Notes for technicians and EE users
Error messages include row and column references from Excel to speed up troubleshooting.

## Execute acquisition sweep (Chunk 2)
```python
from eis import (
    ExcitationConfig,
    HardwareConfig,
    USB6451Adapter,
    execute_sweep,
    load_and_validate_config,
)

sweep = load_and_validate_config("config_examples/config_phase1_example.xlsx")
adapter = USB6451Adapter()
hardware = HardwareConfig(
    device="Dev1",
    ao_channel="ao0",
    ai_channels=("ai0", "ai7"),
    input_mode="differential",
)
excitation = ExcitationConfig(
    drive_mode="auto_from_current_rms",
    offset_v=0.0,
)

def on_progress(p):
    print(f"{p.completed_steps}/{p.total_steps} f={p.frequency_hz:.2f} Hz")

result = execute_sweep(
    sweep=sweep,
    adapter=adapter,
    hardware=hardware,
    excitation=excitation,
    repeats=1,
    run_preflight=True,
    progress_callback=on_progress,
)
adapter.close()
```

This returns raw captures in memory.

## Automatic current-to-amplitude conversion (Chunk 3)
- Default excitation mode is `auto_from_current_rms`.
- `Current_rms` from config is treated as **A RMS**.
- AO amplitude is computed using Clarke-Hess 8100 transconductance ranges from:
  - `USB6451/Clarke Hess 8100 Datsheet.pdf`
- Range selection policy:
  - Prefer ranges where current is within 0-100% full scale.
  - If not possible, use smallest range that still supports requested current.

You can force a specific range:
```python
excitation = ExcitationConfig(
    drive_mode="auto_from_current_rms",
    manual_current_range="20A",
)
```

Or bypass auto conversion:
```python
excitation = ExcitationConfig(
    drive_mode="fixed_ao_amplitude",
    amplitude_v=0.25,  # Vpeak
)
```

## Metadata Bank + Reports (Chunk 4)
Use metadata bank (`.txt` JSON + `.csv`) as the source of truth for report regeneration.

```python
from datetime import datetime
from eis import (
    build_metadata_bank,
    create_run_folder_layout,
    write_description_file,
    write_metadata_bank_txt,
    write_metadata_bank_csv,
    write_metadata_report_html,
)

layout = create_run_folder_layout(
    base_output_dir="measurements",
    serial_number="Z100N34",
    started_at_local=datetime.now(),
)
metadata_bank = build_metadata_bank(
    sweep=sweep,
    run_result=result,
    hardware=hardware,
    excitation=excitation,
    serial_number="Z100N34",
    user_name="operator",
    description="fixture A, room 3",
)
write_metadata_bank_txt(metadata_bank, layout.root / "metadata_bank.txt")
write_metadata_bank_csv(metadata_bank, layout.root / "metadata_measurements.csv")
write_metadata_report_html(metadata_bank, layout.reports / "metadata_report.html")

# Optional description file (generated only if description has content)
write_description_file(metadata_bank["identity"]["description"], layout.root / "description.txt")
```

Notes:
- `metadata_bank.txt` is machine-oriented data bank (JSON), not a human report.
- HTML report is the preferred default view, stored under `REPORTS/`.
- PDF can still be generated on demand from the same metadata bank.
- If report files are lost, regenerate from `metadata_bank.txt`.

## Persist RAW + IMPEDANCE Artifacts (Chunk 5)
This chunk adds repeat-aware disk persistence for acquired data and computed results.

```python
from datetime import datetime
from eis import (
    ImpedancePointResult,
    build_artifact_link_payload,
    build_metadata_bank,
    create_run_folder_layout,
    persist_run_artifacts,
    write_metadata_bank_txt,
    write_metadata_bank_csv,
    write_metadata_report_html,
)

layout = create_run_folder_layout(
    base_output_dir="measurements",
    serial_number="Z100N34",
    started_at_local=datetime.now(),
)

# Placeholder values shown here. In processing chunk these will come from FFT/fitting.
impedance_results = tuple(
    ImpedancePointResult(
        row_number=c.row_number,
        repeat_index=c.repeat_index,
        frequency_hz=c.frequency_hz,
        z_real_ohm=100.0,
        z_imag_ohm=-5.0,
        z_magnitude_ohm=100.1249,
        z_phase_deg=-2.8624,
        extraction_method="demo_placeholder",
    )
    for c in result.captures
)

persisted = persist_run_artifacts(
    layout=layout,
    run_result=result,
    impedance_results=impedance_results,
)
capture_artifacts, point_summaries = build_artifact_link_payload(persisted)

metadata_bank = build_metadata_bank(
    sweep=sweep,
    run_result=result,
    hardware=hardware,
    excitation=excitation,
    serial_number="Z100N34",
    user_name="operator",
    description="fixture A, room 3",
    capture_artifacts=capture_artifacts,
    point_summaries=point_summaries,
)

write_metadata_bank_txt(metadata_bank, layout.root / "metadata_bank.txt")
write_metadata_bank_csv(metadata_bank, layout.root / "metadata_measurements.csv")
write_metadata_report_html(metadata_bank, layout.reports / "metadata_report.html")
```

Repeat file organization:
- `RAW/row_0002_f10Hz/repeat_001_raw.csv`
- `IMPEDANCE/row_0002_f10Hz/repeat_001_impedance.csv`
- `IMPEDANCE/row_0002_f10Hz/summary_mean_std.csv`

`summary_mean_std.csv` is designed to support later Type A uncertainty workflows based on repeats.

## Load Saved Impedance Rows (Preparation for folder-level statistics)
```python
from eis import load_impedance_rows_from_base, load_impedance_rows_from_run

rows_one_run = load_impedance_rows_from_run("measurements/Z100N34_1_3_2026_14_45")
rows_many_runs = load_impedance_rows_from_base("measurements")
```
