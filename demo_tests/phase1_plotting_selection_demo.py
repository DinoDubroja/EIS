"""Demo script: run-selection plotting (Nyquist/Bode/SNR) + noisy raw-vs-fit.

This demo is notebook-oriented and intentionally covers two use cases:
1) Choose run sets by folder-derived metadata and generate overlays:
   - Nyquist
   - Bode
   - SNR vs frequency with threshold shading/checks
2) Generate a noisy raw capture CSV and plot raw-vs-fitted sine overlays.

Outputs are saved to ``PLOTS/`` under the newest synthetic run folder.

Run:
    python demo_tests/phase1_plotting_selection_demo.py
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eis import (
    ImpedancePointResult,
    RunSelection,
    create_run_folder_layout,
    plot_impedance_bode,
    plot_impedance_nyquist,
    plot_raw_vs_fitted_from_csv,
    plot_snr_vs_frequency,
    write_impedance_summary_mean_std_csv,
    write_impedance_table_csv,
)


def _build_rows(scale: float, snr_base_db: float) -> tuple[ImpedancePointResult, ...]:
    """Build deterministic impedance rows for one synthetic run.

    ``snr_base_db`` is intentionally varied per run so threshold checks in the
    SNR plot include both passing and failing examples.
    """

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
                snr_current_db=snr_base_db + 0.8 * repeat_index,
                snr_voltage_db=snr_base_db + 1.5 * repeat_index,
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
                snr_current_db=snr_base_db + 0.4 * repeat_index,
                snr_voltage_db=snr_base_db + 1.2 * repeat_index,
            )
        )
    return tuple(rows)


def _write_noisy_raw_capture_csv(
    *,
    output_path: Path,
    frequency_hz: float,
    sample_rate_sps: float = 5000.0,
    n_periods: int = 30,
) -> Path:
    """Create one noisy 2-channel raw capture CSV for raw-vs-fit demo plotting."""

    sample_count = max(600, int(round(n_periods * sample_rate_sps / frequency_hz)))
    time_s = np.arange(sample_count, dtype=np.float64) / sample_rate_sps
    omega = 2.0 * np.pi * frequency_hz

    # Two channels with different amplitudes/phases and synthetic Gaussian noise.
    rng = np.random.default_rng(20260302 + int(round(frequency_hz * 100)))
    ch1 = 0.02 * np.sin(omega * time_s + 0.20) + 0.0008 * np.sin(3 * omega * time_s)
    ch2 = 8.50 * np.sin(omega * time_s + 0.95) + 0.080 * np.sin(3 * omega * time_s + 0.3)
    ch1 = ch1 + rng.normal(loc=0.0, scale=8.0e-4, size=sample_count)
    ch2 = ch2 + rng.normal(loc=0.0, scale=0.030, size=sample_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "time_s", "ai0_v", "ai7_v"])
        for sample_index in range(sample_count):
            writer.writerow(
                [
                    sample_index,
                    f"{time_s[sample_index]:.12g}",
                    f"{ch1[sample_index]:.12g}",
                    f"{ch2[sample_index]:.12g}",
                ]
            )
    return output_path


def main() -> None:
    """Create synthetic runs, then generate all plotting-demo outputs."""

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
        rows = _build_rows(scale=1.0 + 0.05 * index, snr_base_db=8.0 + 4.2 * index)
        write_impedance_table_csv(results=rows, output_path=layout.impedance / "impedance.csv")
        write_impedance_summary_mean_std_csv(
            results=rows,
            output_path=layout.impedance / "summary_mean_std.csv",
        )
        created_roots.append(layout.root)

    selection_last = RunSelection(mode="last", serial_contains=serial_prefix)
    last_png = created_roots[-1] / "PLOTS" / "demo_nyquist_last.png"
    _, _, runs_last = plot_impedance_nyquist(
        base_output_dir=base,
        selection=selection_last,
        save_path=last_png,
    )

    selection_last_n = RunSelection(mode="last_n", last_n=3, serial_contains=serial_prefix)
    last_n_png = created_roots[-1] / "PLOTS" / "demo_nyquist_last_n.png"
    _, _, runs_last_n = plot_impedance_nyquist(
        base_output_dir=base,
        selection=selection_last_n,
        save_path=last_n_png,
    )

    selection_all_filtered = RunSelection(
        mode="all",
        serial_contains=f"{serial_prefix}_B",
        started_at_or_after=started + timedelta(minutes=3),
    )
    bode_png = created_roots[-1] / "PLOTS" / "demo_bode_filtered.png"
    _, _, runs_all_filtered = plot_impedance_bode(
        base_output_dir=base,
        selection=selection_all_filtered,
        save_path=bode_png,
    )

    snr_png = created_roots[-1] / "PLOTS" / "demo_snr_filtered.png"
    _, _, runs_snr, snr_checks = plot_snr_vs_frequency(
        base_output_dir=base,
        selection=RunSelection(mode="last_n", last_n=3, serial_contains=serial_prefix),
        snr_source="voltage",
        threshold_db=20.0,
        good_region="below_threshold",
        save_path=snr_png,
    )

    raw_csv = _write_noisy_raw_capture_csv(
        output_path=created_roots[-1]
        / "RAW"
        / "row_0002_f10Hz"
        / "repeat_001_raw_ch1_ai0_ch2_ai7.csv",
        frequency_hz=10.0,
    )
    raw_fit_png = created_roots[-1] / "PLOTS" / "demo_raw_vs_fitted_noise.png"
    _, _, raw_fit_result = plot_raw_vs_fitted_from_csv(
        raw_csv_path=raw_csv,
        frequency_hz=10.0,
        save_path=raw_fit_png,
        title="Noisy raw vs fitted (demo)",
    )

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
    print("Selection SNR (last_n) run folders:")
    for item in runs_snr:
        print(f"  - {item.root.name}")
    print("SNR threshold checks (good region: below threshold):")
    for item in snr_checks:
        print(
            "  - "
            f"{item.run.root.name}: passed={item.passed}, "
            f"checked_points={item.checked_points}, "
            f"min_snr_db={item.min_snr_db:.2f}, max_snr_db={item.max_snr_db:.2f}"
        )
    print("Raw-vs-fitted channel summaries:")
    for channel in raw_fit_result.channel_summaries:
        print(
            "  - "
            f"{channel.channel_name}: SNR={channel.snr_db:.2f} dB, "
            f"A={channel.amplitude_v_peak:.5g} Vpk, phase={channel.phase_deg:.2f} deg"
        )
    print(f"Saved nyquist last: {last_png}")
    print(f"Saved nyquist last_n: {last_n_png}")
    print(f"Saved bode filtered: {bode_png}")
    print(f"Saved SNR plot: {snr_png}")
    print(f"Saved raw-vs-fitted plot: {raw_fit_png}")


if __name__ == "__main__":
    main()
