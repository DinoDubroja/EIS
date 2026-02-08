# USB6451 Module

High-level helper class for NI USB-6451 control.

## Current scope

- Continuous sine generation on analog output (AO)
- Continuous arbitrary periodic waveform generation from one-period sample list
- Input validation for safe waveform setup
- Config preview without starting output via `get_continuous_sine_output_config(...)`
- Waveform period generators in `USB6451/waveforms.py`

## Main API

- `start_continuous_sine_output(...)`
  - Generates periodic sine output.
- `start_continuous_periodic_output(period_samples=..., sample_rate=..., ...)`
  - Generates periodic output from user-provided one-period voltage samples.
  - Always uses regenerative mode for periodic replay.
  - Rejects `period_samples` longer than `16,383` samples.
  - Returns output frequency `sample_rate / len(period_samples)`.
- `stop_output()`
  - Stops output and releases the AO task.

## Waveform helpers

`USB6451/waveforms.py` provides one-period generators:

- `sine_period(...)`
- `ramp_period(...)`
- `staircase_period(...)`
- `triangle_period(...)`
- `square_period(..., duty=...)`

Default safety checks in helpers:
- AO voltage range: `-10 V` to `+10 V`
- max period length: `16,383` samples

## Demo notebook

- `USB6451/Demos/sinewave_periodic.ipynb`  
  A focused demo for periodic sine generation on AO using the `USB6451` class. It explains non-integer frequency behavior and prints requested vs actual frequency before output starts.
- `USB6451/Demos/waveforms_demo.ipynb`  
  A DAQ-free notebook for waveform data preparation and visualization from `waveforms.py`. It includes staircase-style plots and an interactive explorer for live parameter tuning.

## Tests

Tests are stored per module in:

- `USB6451/tests/test_usb6451_unit.py`
- `USB6451/tests/test_waveforms_unit.py`

These are unit tests (no hardware needed).  
They stub `nidaqmx` so they can run even when NI drivers are not installed.

Run from repository root:

```powershell
python -m unittest discover -s USB6451/tests -p "test_*.py"
```

## Next planned tests

- Hardware tests in a separate file (requires connected DAQ), for example:
  - start output
  - stop output
  - optional AO-to-AI loopback verification
