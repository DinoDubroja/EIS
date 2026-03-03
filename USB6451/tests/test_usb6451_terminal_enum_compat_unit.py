"""Compatibility test for NI-DAQmx terminal enum naming variants.

This test simulates a nidaqmx package where terminal enums are exposed as
``DIFF`` / ``PSEUDO_DIFF`` without legacy aliases like ``DIFFERENTIAL``.
It ensures USB6451 input-mode mapping remains compatible across versions.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_usb6451_with_diff_only_enums():
    """Load USB6451 module with a fake nidaqmx exposing DIFF-only enum names."""

    saved_modules = {
        name: sys.modules.get(name) for name in ("nidaqmx", "nidaqmx.constants")
    }
    try:
        fake_nidaqmx = types.ModuleType("nidaqmx")
        fake_constants = types.ModuleType("nidaqmx.constants")

        class DaqError(Exception):
            """Stub DAQ error type."""

        class Task:
            """Minimal stub task type for import compatibility."""

            def __init__(self) -> None:
                pass

        class AcquisitionType:
            CONTINUOUS = "CONTINUOUS"
            FINITE = "FINITE"

        class RegenerationMode:
            DONT_ALLOW_REGENERATION = "DONT_ALLOW_REGENERATION"

        class TerminalConfiguration:
            DEFAULT = "DEFAULT"
            DIFF = "DIFF"
            RSE = "RSE"
            NRSE = "NRSE"
            PSEUDO_DIFF = "PSEUDO_DIFF"

        fake_nidaqmx.DaqError = DaqError
        fake_nidaqmx.Task = Task
        fake_constants.AcquisitionType = AcquisitionType
        fake_constants.RegenerationMode = RegenerationMode
        fake_constants.TerminalConfiguration = TerminalConfiguration

        sys.modules["nidaqmx"] = fake_nidaqmx
        sys.modules["nidaqmx.constants"] = fake_constants

        repo_root = Path(__file__).resolve().parents[2]
        module_path = repo_root / "USB6451" / "USB6451.py"
        module_name = "usb6451_module_diff_only_enum_test"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to import module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in saved_modules.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


class TestUSB6451TerminalEnumCompatUnit(unittest.TestCase):
    """Checks input-mode mapping when nidaqmx exposes DIFF-only enum names."""

    # Verifies differential/pseudodifferential map correctly without legacy enum names.
    def test_resolve_terminal_config_with_diff_only_enum_names(self) -> None:
        mod = _load_usb6451_with_diff_only_enums()

        resolved_diff = mod.USB6451._resolve_terminal_config(
            input_mode="differential",
            terminal_config=None,
        )
        self.assertEqual(resolved_diff, mod.TerminalConfiguration.DIFF)

        resolved_pseudo = mod.USB6451._resolve_terminal_config(
            input_mode="pseudodifferential",
            terminal_config=None,
        )
        self.assertEqual(resolved_pseudo, mod.TerminalConfiguration.PSEUDO_DIFF)


if __name__ == "__main__":
    unittest.main()
