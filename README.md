# EIS
Repository for EIS measurement scripts and libraries

## Modules

- `eis`: Phase 1 backend package (new modular API surface)
  - Config loader: `eis/config/excel_loader.py`
  - Config validator: `eis/config/validator.py`
  - Data models: `eis/models/config_models.py`
  - Architecture notes: `docs/architecture.md`
  - Usage notes: `docs/phase1_usage.md`
- `USB6451`: NI USB-6451 high-level helper class
  - Docs: `USB6451/README.md`
  - Unit tests: `USB6451/tests/test_usb6451_unit.py`, `USB6451/tests/test_waveforms_unit.py`
  - Demo notebook: `USB6451/Demos/sinewave_periodic.ipynb`
  - Waveforms demo notebook: `USB6451/Demos/waveforms_demo.ipynb`
  - Continuous AI demo notebook: `USB6451/Demos/analog_input_continuous.ipynb`
  - Waveform helpers: `USB6451/waveforms.py`
  - Includes continuous/finite AO+AI APIs, synchronized AO+AI APIs, non-regen sine streaming, and AI input-mode selection

## Run USB6451 unit tests

```powershell
python -m unittest discover -s USB6451/tests -p "test_*.py"
```

## Run EIS config unit tests

```powershell
python -m unittest discover -s tests/unit -p "test_*.py"
```

## TODO

* synchronous AO + 2*AI notebook
* higher-level functions
* EIS.py
