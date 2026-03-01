"""Metadata persistence and report rendering for Phase 1 measurement runs.

This module treats metadata as a two-layer system:
1. A machine-readable data bank (`metadata_bank.txt` as JSON and CSV capture table).
2. Human-friendly report views generated from that bank (HTML and optional PDF).

Design intent:
- The data bank is the source of truth for report regeneration.
- Report files are disposable views. If deleted, regenerate from the bank.
- Metadata files include enough context (DAQ settings, config summary, capture
  timing, current range, and computed drive values) to rebuild reports later.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from eis.models.config_models import SweepConfig
from eis.models.measurement_models import ExcitationConfig, HardwareConfig, SweepRunResult


def build_metadata_bank(
    *,
    sweep: SweepConfig,
    run_result: SweepRunResult,
    hardware: HardwareConfig,
    excitation: ExcitationConfig,
    serial_number: str,
    user_name: str,
    description: str | None = None,
) -> dict[str, Any]:
    """Build structured metadata dictionary used as report data bank."""

    captures = []
    for capture in run_result.captures:
        captures.append(
            {
                "row_number": capture.row_number,
                "repeat_index": capture.repeat_index,
                "frequency_hz": capture.frequency_hz,
                "sample_rate_sps": capture.sample_rate_sps,
                "n_periods": capture.n_periods,
                "current_rms_a": capture.current_rms,
                "ao_amplitude_v_peak": capture.ao_amplitude_v_peak,
                "ao_offset_v": capture.ao_offset_v,
                "current_range_name": capture.current_range_name,
                "transconductance_siemens": capture.transconductance_siemens,
                "started_at_utc_iso": capture.started_at_utc_iso,
                "duration_s": capture.duration_s,
                "ai_channels": list(capture.ai_channels),
                "ai_range_v": capture.ai_range_v,
                "raw_shape": [int(capture.raw_data.shape[0]), int(capture.raw_data.shape[1])],
            }
        )

    payload: dict[str, Any] = {
        "schema_version": "phase1_metadata_v1",
        "generated_at_utc_iso": datetime.now(timezone.utc).isoformat(),
        "identity": {
            "serial_number": serial_number,
            "user_name": user_name,
            "description": description or "",
        },
        "hardware": {
            "device": hardware.device,
            "ao_channel": hardware.ao_channel,
            "ai_channels": list(hardware.ai_channels),
            "input_mode": hardware.input_mode,
            "ao_min_voltage": hardware.ao_min_voltage,
            "ao_max_voltage": hardware.ao_max_voltage,
            "ai_default_min_voltage": hardware.ai_default_min_voltage,
            "ai_default_max_voltage": hardware.ai_default_max_voltage,
            "timeout_s": hardware.timeout_s,
        },
        "excitation": {
            "drive_mode": excitation.drive_mode,
            "amplitude_v": excitation.amplitude_v,
            "offset_v": excitation.offset_v,
            "manual_current_range": excitation.manual_current_range,
            "range_selection_policy": excitation.range_selection_policy,
        },
        "sweep": {
            "source_path": str(sweep.source_path),
            "sheet_name": sweep.sheet_name,
            "point_count": len(sweep.points),
            "repeats": run_result.repeats,
            "started_at_utc_iso": run_result.started_at_utc_iso,
            "finished_at_utc_iso": run_result.finished_at_utc_iso,
        },
        "preflight": (
            {
                "sample_rate_sps": run_result.preflight.sample_rate_sps,
                "samples_per_channel": run_result.preflight.samples_per_channel,
                "measured_shape": list(run_result.preflight.measured_shape),
                "message": run_result.preflight.message,
            }
            if run_result.preflight is not None
            else None
        ),
        "captures": captures,
    }
    return payload


def write_metadata_bank_txt(metadata_bank: dict[str, Any], output_path: str | Path) -> Path:
    """Write machine-readable metadata bank to ``.txt`` as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata_bank, indent=2), encoding="utf-8")
    return path


def write_metadata_bank_csv(metadata_bank: dict[str, Any], output_path: str | Path) -> Path:
    """Write per-capture metadata table to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "row_number",
        "repeat_index",
        "frequency_hz",
        "sample_rate_sps",
        "n_periods",
        "current_rms_a",
        "ao_amplitude_v_peak",
        "ao_offset_v",
        "current_range_name",
        "transconductance_siemens",
        "started_at_utc_iso",
        "duration_s",
        "ai_channels",
        "ai_range_v",
        "raw_shape",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metadata_bank["captures"]:
            serializable = dict(row)
            serializable["ai_channels"] = ",".join(row["ai_channels"])
            serializable["raw_shape"] = "x".join(str(v) for v in row["raw_shape"])
            writer.writerow(serializable)
    return path


def write_metadata_report_html(metadata_bank: dict[str, Any], output_path: str | Path) -> Path:
    """Render a visual HTML metadata report from metadata bank."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    identity = metadata_bank["identity"]
    hardware = metadata_bank["hardware"]
    sweep = metadata_bank["sweep"]
    preflight = metadata_bank["preflight"]
    captures = metadata_bank["captures"]

    capture_rows = "\n".join(
        (
            "<tr>"
            f"<td>{idx+1}</td>"
            f"<td>{item['frequency_hz']:.6g}</td>"
            f"<td>{item['repeat_index']}</td>"
            f"<td>{item['current_rms_a']:.6g}</td>"
            f"<td>{item['current_range_name'] or '-'}</td>"
            f"<td>{item['transconductance_siemens'] if item['transconductance_siemens'] is not None else '-'}</td>"
            f"<td>{item['ao_amplitude_v_peak']:.6g}</td>"
            f"<td>{item['started_at_utc_iso']}</td>"
            f"<td>{item['duration_s']:.4f}</td>"
            "</tr>"
        )
        for idx, item in enumerate(captures)
    )

    preflight_html = (
        f"<p><b>Preflight:</b> {preflight['message']} | "
        f"rate={preflight['sample_rate_sps']} S/s, "
        f"samples/ch={preflight['samples_per_channel']}, "
        f"shape={preflight['measured_shape']}</p>"
        if preflight is not None
        else "<p><b>Preflight:</b> skipped</p>"
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Metadata Report - {identity["serial_number"]}</title>
  <style>
    :root {{
      --bg: #f2f6f8;
      --card: #ffffff;
      --ink: #12303b;
      --muted: #4f6a74;
      --line: #d5e0e6;
      --accent: #0a7f9c;
    }}
    body {{
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 0% 0%, #d8eef4 0%, transparent 45%),
        radial-gradient(circle at 100% 0%, #e6f3d8 0%, transparent 40%),
        var(--bg);
    }}
    .wrap {{
      max-width: 1200px;
      margin: 24px auto;
      padding: 0 16px 28px;
    }}
    .title {{
      margin: 0 0 10px;
      font-size: 30px;
      letter-spacing: 0.4px;
    }}
    .subtitle {{
      margin: 0 0 20px;
      color: var(--muted);
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(280px, 1fr));
      gap: 14px;
      margin-bottom: 16px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px 16px;
      box-shadow: 0 10px 20px rgba(18, 48, 59, 0.06);
    }}
    h2 {{
      margin: 0 0 10px;
      color: var(--accent);
      font-size: 16px;
    }}
    p {{
      margin: 4px 0;
      font-size: 13px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      background: #fff;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid var(--line);
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #eef7fa;
      color: #104959;
      position: sticky;
      top: 0;
    }}
    tr:nth-child(even) td {{
      background: #fbfdfe;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1 class="title">EIS Metadata Report</h1>
    <p class="subtitle">Serial: <b>{identity["serial_number"]}</b> | User: <b>{identity["user_name"]}</b> | Generated UTC: {metadata_bank["generated_at_utc_iso"]}</p>

    <div class="grid">
      <section class="card">
        <h2>Identity</h2>
        <p><b>Serial number:</b> {identity["serial_number"]}</p>
        <p><b>User:</b> {identity["user_name"]}</p>
        <p><b>Description:</b> {identity["description"]}</p>
      </section>
      <section class="card">
        <h2>Sweep Summary</h2>
        <p><b>Source config:</b> {sweep["source_path"]}</p>
        <p><b>Sheet:</b> {sweep["sheet_name"]}</p>
        <p><b>Points:</b> {sweep["point_count"]}</p>
        <p><b>Repeats:</b> {sweep["repeats"]}</p>
        <p><b>Start UTC:</b> {sweep["started_at_utc_iso"]}</p>
        <p><b>Finish UTC:</b> {sweep["finished_at_utc_iso"]}</p>
      </section>
      <section class="card">
        <h2>DAQ Settings</h2>
        <p><b>Device:</b> {hardware["device"]}</p>
        <p><b>AO channel:</b> {hardware["ao_channel"]}</p>
        <p><b>AI channels:</b> {", ".join(hardware["ai_channels"])}</p>
        <p><b>Input mode:</b> {hardware["input_mode"]}</p>
        <p><b>AO limits:</b> [{hardware["ao_min_voltage"]}, {hardware["ao_max_voltage"]}] V</p>
        <p><b>Default AI limits:</b> [{hardware["ai_default_min_voltage"]}, {hardware["ai_default_max_voltage"]}] V</p>
      </section>
      <section class="card">
        <h2>Preflight</h2>
        {preflight_html}
      </section>
    </div>

    <section class="card">
      <h2>Capture Timeline</h2>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Frequency (Hz)</th>
            <th>Repeat</th>
            <th>Current RMS (A)</th>
            <th>Range</th>
            <th>Transconductance (S)</th>
            <th>AO Amplitude (Vpeak)</th>
            <th>Start UTC</th>
            <th>Duration (s)</th>
          </tr>
        </thead>
        <tbody>
          {capture_rows}
        </tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path


def write_metadata_report_pdf(metadata_bank: dict[str, Any], output_path: str | Path) -> Path:
    """Render PDF metadata report using matplotlib backend."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    identity = metadata_bank["identity"]
    hardware = metadata_bank["hardware"]
    sweep = metadata_bank["sweep"]
    preflight = metadata_bank["preflight"]
    captures = metadata_bank["captures"]

    with PdfPages(path) as pdf:
        # Page 1: summary blocks
        fig = plt.figure(figsize=(11.7, 8.3))
        ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
        ax.axis("off")
        ax.set_title(
            f"EIS Metadata Report - {identity['serial_number']}",
            fontsize=18,
            fontweight="bold",
            loc="left",
            pad=14,
        )

        lines = [
            f"Generated UTC: {metadata_bank['generated_at_utc_iso']}",
            "",
            "Identity",
            f"  Serial number: {identity['serial_number']}",
            f"  User: {identity['user_name']}",
            f"  Description: {identity['description']}",
            "",
            "Sweep",
            f"  Config source: {sweep['source_path']}",
            f"  Sheet: {sweep['sheet_name']}",
            f"  Points: {sweep['point_count']}, Repeats: {sweep['repeats']}",
            f"  Start UTC: {sweep['started_at_utc_iso']}",
            f"  Finish UTC: {sweep['finished_at_utc_iso']}",
            "",
            "DAQ",
            f"  Device: {hardware['device']}",
            f"  AO: {hardware['ao_channel']}",
            f"  AI: {', '.join(hardware['ai_channels'])}",
            f"  Input mode: {hardware['input_mode']}",
            f"  AO limits: [{hardware['ao_min_voltage']}, {hardware['ao_max_voltage']}] V",
            f"  AI limits default: [{hardware['ai_default_min_voltage']}, {hardware['ai_default_max_voltage']}] V",
            "",
            (
                "Preflight: skipped"
                if preflight is None
                else (
                    "Preflight: "
                    f"{preflight['message']} | rate={preflight['sample_rate_sps']} S/s "
                    f"samples/ch={preflight['samples_per_channel']} shape={preflight['measured_shape']}"
                )
            ),
        ]
        ax.text(0.0, 0.97, "\n".join(lines), va="top", fontsize=11, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2+: capture table chunks
        headers = [
            "#",
            "Freq (Hz)",
            "Rep",
            "Irms (A)",
            "Range",
            "Gm (S)",
            "AO Vpk",
            "Start UTC",
            "Dur (s)",
        ]
        rows = []
        for idx, item in enumerate(captures, start=1):
            rows.append(
                [
                    idx,
                    f"{item['frequency_hz']:.6g}",
                    item["repeat_index"],
                    f"{item['current_rms_a']:.6g}",
                    item["current_range_name"] or "-",
                    (
                        f"{item['transconductance_siemens']:.6g}"
                        if item["transconductance_siemens"] is not None
                        else "-"
                    ),
                    f"{item['ao_amplitude_v_peak']:.6g}",
                    item["started_at_utc_iso"],
                    f"{item['duration_s']:.4f}",
                ]
            )

        chunk_size = 26
        for chunk_start in range(0, len(rows), chunk_size):
            fig = plt.figure(figsize=(11.7, 8.3))
            ax = fig.add_axes([0.03, 0.06, 0.94, 0.88])
            ax.axis("off")
            ax.set_title(
                "Capture Timeline",
                fontsize=16,
                fontweight="bold",
                loc="left",
                pad=10,
            )
            chunk = rows[chunk_start : chunk_start + chunk_size]
            table = ax.table(
                cellText=chunk,
                colLabels=headers,
                loc="upper left",
                cellLoc="left",
                colLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8.5)
            table.scale(1.0, 1.25)
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#d9edf4")
                    cell.set_text_props(weight="bold")
            pdf.savefig(fig)
            plt.close(fig)

    return path


def write_description_file(description: str | None, output_path: str | Path) -> Path | None:
    """Write optional user description text file.

    Inputs:
        description: Optional text entered by user.
        output_path: Target file path.
    Output:
        Path of created file, or ``None`` when description is empty.
    Notes:
        If description is missing/blank, no file is created by design.
    """

    if description is None or description.strip() == "":
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(description.strip(), encoding="utf-8")
    return path


def regenerate_reports_from_bank(
    *,
    metadata_bank_txt_path: str | Path,
    html_output_path: str | Path,
    pdf_output_path: str | Path,
) -> tuple[Path, Path]:
    """Regenerate HTML/PDF reports from metadata bank file."""

    bank = json.loads(Path(metadata_bank_txt_path).read_text(encoding="utf-8"))
    html_path = write_metadata_report_html(bank, html_output_path)
    pdf_path = write_metadata_report_pdf(bank, pdf_output_path)
    return html_path, pdf_path
