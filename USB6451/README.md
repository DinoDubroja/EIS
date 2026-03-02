# USB6451 Module

High-level helper class for NI USB-6451 control.

## Current scope

- Continuous sine generation on analog output (AO)
- Continuous non-regenerative sine generation for non-integer divider frequencies
- Continuous arbitrary periodic waveform generation from one-period sample list
- Continuous analog input (AI) start/read/stop API
- Finite analog input (AI) measurement API
- Finite synchronized AO+AI measurement API
- Finite sine-period measurement helper (auto regenerative vs non-regenerative sample path)
- Input validation for safe waveform setup
- Simple AI input mode selection (`default`, `differential`, `rse`, `nrse`, `pseudodifferential`)
- Config preview without starting output via `get_continuous_sine_output_config(...)`
- Waveform period generators in `USB6451/waveforms.py`

## Main API

- `start_continuous_sine_output(...)`
  - Generates periodic sine output.
- `start_continuous_sine_output_non_regen(...)`
  - Starts non-regenerative sine output for exact requested sine frequency.
  - Writes one initial chunk and starts AO task.
- `write_sine_chunk_non_regen(...)`
  - Generates and writes the next sine chunk with phase continuity.
  - Use this in a loop to keep non-regenerative output running.
- `start_continuous_periodic_output(period_samples=..., sample_rate=..., ...)`
  - Generates periodic output from user-provided one-period voltage samples.
  - Always uses regenerative mode for periodic replay.
  - Rejects `period_samples` longer than `16,383` samples.
  - Returns output frequency `sample_rate / len(period_samples)`.
- `start_continuous_input(...)`
  - Starts continuous AI acquisition with internal sample clock.
  - Uses USB-6451 AI limit guard (`<= 1,000,000 S/s`) and channel-count guard (`<= 16`).
  - Supports `input_mode` for AI wiring selection.
- `read_input_chunk(...)`
  - Reads a chunk from active AI task and returns `numpy.ndarray` with shape
    `(channels, samples_per_channel)`.
- `measure_input_finite(samples_per_channel=..., sample_rate=..., ...)`
  - Acquires one finite AI block and returns raw data with shape
    `(channels, samples_per_channel)`.
  - Supports `input_mode` for AI wiring selection.
- `start_continuous_sync_periodic_io(...)`
  - Starts synchronized AO periodic output and AI acquisition with shared sample rate.
  - Uses AI start trigger terminal as AO start trigger source (NI sync pattern).
  - Returns synchronized configuration metadata including actual sample rate.
  - Supports `input_mode` for AI wiring selection.
- `read_sync_input_chunk(...)`
  - Reads one chunk from synchronized AI task and returns shape
    `(channels, samples_per_channel)`.
- `measure_sync_finite(output_samples=..., sample_rate=..., ...)`
  - Runs one finite synchronized AO output + AI input measurement.
  - Returns raw AI data with shape `(channels, len(output_samples))`.
- `measure_sine_periods(periods=..., frequency=..., ...)`
  - Builds a finite AO sine sequence and captures synchronized AI data.
  - If `sample_rate / frequency` is integer-like, uses periodic replay samples.
  - Otherwise uses phase-continuous non-regenerative style samples.
  - Returns raw AI data with shape `(channels, N)`, where `N` depends on chosen path.
- `validate_sync_connection(...)`
  - Runs a short synchronized AO+AI preflight check for USB-6451 connectivity.
  - Uses a constant AO test level and confirms returned AI sample shape.
  - Discards configurable startup-settling time before validation.
  - Applies current-channel shunt-voltage tolerance check (single overall pass/fail).
  - Intended for "check hardware before sweep" workflows.
- `stop_sync_io()`
  - Stops and releases synchronized AO+AI tasks.
- `stop_input()`
  - Stops input and releases the AI task.
- `stop_output()`
  - Stops output and releases the AO task.

## Docstring Convention

USB6451 method docstrings follow a consistent structure for lab readability:
- `Purpose`
- `Inputs`
- `Output`
- `Raises` (when relevant)
- `Notes` (when relevant)

All units are explicit (`Hz`, `V`, `S/s`) to reduce ambiguity in measurement setups.

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
  A focused sine generation demo with both periodic regenerative mode and non-regenerative chunk streaming mode. It explains non-integer divider behavior and shows how to run exact-frequency non-regen output.
- `USB6451/Demos/waveforms_demo.ipynb`  
  A DAQ-free notebook for waveform data preparation and visualization from `waveforms.py`. It includes staircase-style plots and an interactive explorer for live parameter tuning.
- `USB6451/Demos/analog_input_continuous.ipynb`  
  A step-by-step continuous AI acquisition demo using `start_continuous_input`, `read_input_chunk`, and `stop_input`. It also supports optional AO sine generation for a simple loopback measurement before plotting time-voltage results.

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

## DAQ references used

- NI examples:
  - `Reference/EIS_new_OLD/NI DAQ - eaxmples/nidaqmx-python-master/nidaqmx-python-master/examples/analog_in/voltage_acq_int_clk.py`
  - `Reference/EIS_new_OLD/NI DAQ - eaxmples/nidaqmx-python-master/nidaqmx-python-master/examples/analog_in/cont_voltage_acq_int_clk.py`
  - `Reference/EIS_new_OLD/NI DAQ - eaxmples/nidaqmx-python-master/nidaqmx-python-master/examples/analog_out/cont_gen_voltage_wfm_int_clk.py`
  - `Reference/EIS_new_OLD/NI DAQ - eaxmples/nidaqmx-python-master/nidaqmx-python-master/examples/synchronization/multi_function/ai_ao_sync.py`
  - `Reference/EIS_new_OLD/NI DAQ - eaxmples/nidaqmx-python-master/nidaqmx-python-master/examples/playrec.py`
- USB-6451 manual:
  - `USB6451/USB6451 manual.pdf`

## Next planned tests

- Hardware tests in a separate file (requires connected DAQ), for example:
  - finite AI capture on connected channel
  - finite synchronized AO+AI loopback verification
  - continuous long-run stability checks
