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

This returns raw captures in memory. Saving and plotting layers will be added in next chunks.

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
