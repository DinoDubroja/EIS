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
    load_and_validate_config,
    plot_impedance_bode,
    plot_impedance_nyquist,
    plot_raw_vs_fitted_from_csv,
    plot_snr_vs_frequency,
    write_impedance_summary_mean_std_csv,
    write_impedance_table_csv,
)
from eis.storage.naming import build_point_folder_name


def _load_demo_frequencies() -> tuple[float, ...]:
    """Load demo frequencies from config example, with deterministic fallback.

    Primary source:
    - ``config_examples/config_phase1_example.xlsx``
    Fallback:
    - hard-coded list matching the current example config values.
    """

    config_path = REPO_ROOT / "config_examples" / "config_phase1_example.xlsx"
    fallback = (
        12.54,
        19.9,
        31.54,
        49.0,
        79.18,
        125.79,
        199.6,
        314.46,
        497.51,
        793.65,
        1234.57,
        1960.78,
        3030.3,
        4761.91,
    )
    if not config_path.exists():
        return fallback
    try:
        sweep = load_and_validate_config(config_path)
    except Exception:
        return fallback

    values = [float(point.frequency_hz) for point in sweep.points]
    return tuple(values) if values else fallback


def _build_rows(
    *,
    scale: float,
    snr_base_db: float,
    frequencies_hz: tuple[float, ...],
) -> tuple[ImpedancePointResult, ...]:
    """Build deterministic impedance rows for one synthetic run.

    The synthetic model uses a smooth RC-like trend over frequency so Nyquist
    and Bode overlays look realistic while still remaining deterministic.
    """

    rows: list[ImpedancePointResult] = []
    for row_number, frequency_hz in enumerate(frequencies_hz, start=2):
        normalized = np.log10(max(frequency_hz, 1e-6))
        z_real_nominal = 5.3 - 0.34 * normalized
        z_imag_nominal = 1.35 / (1.0 + 0.55 * normalized)
        for repeat_index in (1, 2):
            z_real = scale * (z_real_nominal + 0.015 * repeat_index)
            z_imag = scale * (z_imag_nominal + 0.010 * repeat_index)
            z_value = complex(z_real, z_imag)
            rows.append(
                ImpedancePointResult(
                    row_number=row_number,
                    repeat_index=repeat_index,
                    frequency_hz=frequency_hz,
                    z_real_ohm=float(np.real(z_value)),
                    z_imag_ohm=float(np.imag(z_value)),
                    z_magnitude_ohm=float(abs(z_value)),
                    z_phase_deg=float(np.degrees(np.angle(z_value))),
                    extraction_method="fft",
                    snr_current_db=snr_base_db + 2.0 * normalized + 0.30 * repeat_index,
                    snr_voltage_db=snr_base_db + 2.8 * normalized + 0.45 * repeat_index,
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
    frequencies_hz = _load_demo_frequencies()

    created_roots: list[Path] = []
    for index, serial_suffix in enumerate(("A", "A", "B", "B"), start=1):
        serial_number = f"{serial_prefix}_{serial_suffix}"
        dt = started + timedelta(minutes=index)
        layout = create_run_folder_layout(
            base_output_dir=base,
            serial_number=serial_number,
            started_at_local=dt,
        )
        rows = _build_rows(
            scale=1.0 + 0.05 * index,
            snr_base_db=8.0 + 4.2 * index,
            frequencies_hz=frequencies_hz,
        )
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

    raw_demo_frequency_hz = float(frequencies_hz[0])
    raw_csv = _write_noisy_raw_capture_csv(
        output_path=created_roots[-1]
        / "RAW"
        / build_point_folder_name(2, raw_demo_frequency_hz)
        / "repeat_001_raw_ch1_ai0_ch2_ai7.csv",
        frequency_hz=raw_demo_frequency_hz,
    )
    raw_fit_png = created_roots[-1] / "PLOTS" / "demo_raw_vs_fitted_noise.png"
    raw_fit_svg = created_roots[-1] / "PLOTS" / "demo_raw_vs_fitted_noise.svg"
    _, _, raw_fit_result = plot_raw_vs_fitted_from_csv(
        raw_csv_path=raw_csv,
        frequency_hz=raw_demo_frequency_hz,
        save_path=raw_fit_png,
        save_vector_path=raw_fit_svg,
        title="Noisy raw vs fitted (demo)",
    )

    print("Plot selection demo completed")
    print(f"Created synthetic runs: {len(created_roots)}")
    print(f"Frequency points per run: {len(frequencies_hz)}")
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
    print(f"Saved raw-vs-fitted vector plot: {raw_fit_svg}")


if __name__ == "__main__":
    main()
