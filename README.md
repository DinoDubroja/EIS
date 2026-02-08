# EIS
Repository for EIS measurement scripts and libraries

## Modules

- `USB6451`: NI USB-6451 high-level helper class
  - Docs: `USB6451/README.md`
  - Unit tests: `USB6451/tests/test_usb6451_unit.py`, `USB6451/tests/test_waveforms_unit.py`
  - Demo notebook: `USB6451/Demos/sinewave_periodic.ipynb`
  - Waveforms demo notebook: `USB6451/Demos/waveforms_demo.ipynb`
  - Waveform helpers: `USB6451/waveforms.py`

## Run USB6451 unit tests

```powershell
python -m unittest discover -s USB6451/tests -p "test_*.py"
```

## TODO

* Create EIS.py class
* test repo
