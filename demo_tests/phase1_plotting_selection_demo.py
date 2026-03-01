"""Demo script: plotting selection by last/last_n/all, serial, and time filters.

This demo creates synthetic run folders under `measurements/` and generates
Nyquist/Bode overlays using run-folder name filters.

Run:
    python demo_tests/phase1_plotting_selection_demo.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eis import (
    ImpedancePointResult,
    RunSelection,
    create_run_folder_layout,
    plot_impedance_bode,
    plot_impedance_nyquist,
    write_impedance_summary_mean_std_csv,
    write_impedance_table_csv,
)


def _build_rows(scale: float) -> tuple[ImpedancePointResult, ...]:
    """Build deterministic impedance rows for one synthetic run."""

    rows: list[ImpedancePointResult] = []
    for repeat_index in (1, 2):
        rows.append(
            ImpedancePointResult(
                row_number=2,
                repeat_index=repeat_index,
                frequency_hz=10.0,
                z_real_ohm=scale * (5.0 + 0.05 * repeat_index),
                z_imag_ohm=scale * (1.0 + 0.03 * repeat_index),
                z_magnitude_ohm=scale * 5.2,
                z_phase_deg=11.5,
                extraction_method="fft",
            )
        )
        rows.append(
            ImpedancePointResult(
                row_number=3,
                repeat_index=repeat_index,
                frequency_hz=20.0,
                z_real_ohm=scale * (4.7 + 0.04 * repeat_index),
                z_imag_ohm=scale * (0.8 + 0.02 * repeat_index),
                z_magnitude_ohm=scale * 4.8,
                z_phase_deg=9.7,
                extraction_method="fft",
            )
        )
    return tuple(rows)


def main() -> None:
    """Create synthetic runs, then produce selection-based Nyquist/Bode plots."""

    base = REPO_ROOT / "measurements"
    started = datetime.now().replace(second=0, microsecond=0)
    serial_prefix = f"PLOTDEMO_{datetime.now().strftime('%H%M%S')}"

    created_roots: list[Path] = []
    for index, serial_suffix in enumerate(("A", "A", "B", "B"), start=1):
        serial_number = f"{serial_prefix}_{serial_suffix}"
        dt = started + timedelta(minutes=index)
        layout = create_run_folder_layout(
            base_output_dir=base,
            serial_number=serial_number,
            started_at_local=dt,
        )
        rows = _build_rows(scale=1.0 + 0.05 * index)
        write_impedance_table_csv(results=rows, output_path=layout.impedance / "impedance.csv")
        write_impedance_summary_mean_std_csv(
            results=rows,
            output_path=layout.impedance / "summary_mean_std.csv",
        )
        created_roots.append(layout.root)

    selection_last = RunSelection(mode="last", serial_contains=serial_prefix)
    fig_last, _, runs_last = plot_impedance_nyquist(
        base_output_dir=base,
        selection=selection_last,
    )
    last_png = created_roots[-1] / "REPORTS" / "demo_nyquist_last.png"
    fig_last.savefig(last_png, dpi=140)

    selection_last_n = RunSelection(mode="last_n", last_n=3, serial_contains=serial_prefix)
    fig_last_n, _, runs_last_n = plot_impedance_nyquist(
        base_output_dir=base,
        selection=selection_last_n,
    )
    last_n_png = created_roots[-1] / "REPORTS" / "demo_nyquist_last_n.png"
    fig_last_n.savefig(last_n_png, dpi=140)

    selection_all_filtered = RunSelection(
        mode="all",
        serial_contains=f"{serial_prefix}_B",
        started_at_or_after=started + timedelta(minutes=3),
    )
    fig_bode, _, runs_all_filtered = plot_impedance_bode(
        base_output_dir=base,
        selection=selection_all_filtered,
    )
    bode_png = created_roots[-1] / "REPORTS" / "demo_bode_filtered.png"
    fig_bode.savefig(bode_png, dpi=140)

    print("Plot selection demo completed")
    print(f"Created synthetic runs: {len(created_roots)}")
    print("Selection last run folders:")
    for item in runs_last:
        print(f"  - {item.root.name}")
    print("Selection last_n run folders:")
    for item in runs_last_n:
        print(f"  - {item.root.name}")
    print("Selection filtered(all) run folders:")
    for item in runs_all_filtered:
        print(f"  - {item.root.name}")
    print(f"Saved nyquist last: {last_png}")
    print(f"Saved nyquist last_n: {last_n_png}")
    print(f"Saved bode filtered: {bode_png}")


if __name__ == "__main__":
    main()
