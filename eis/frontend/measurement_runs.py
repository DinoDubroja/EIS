"""High-level orchestration helpers for notebook measurement workflow.

This module provides short, technician-friendly wrapper functions that combine
multiple lower-level API calls into a few clear operations:
- run only DAQ preflight check
- run sweep, process impedance, persist artifacts, and write metadata

The intent is to keep notebook cells compact and readable while preserving
access to the full lower-level API when fine control is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from eis.acquisition import USB6451Adapter, execute_sweep, run_preflight_check
from eis.models.config_models import SweepConfig
from eis.models.measurement_models import (
    CaptureConditioningConfig,
    ExcitationConfig,
    HardwareConfig,
    ImpedancePointResult,
    PreflightCheckResult,
    SweepRunResult,
)
from eis.processing import ImpedanceProcessingConfig, compute_impedance_for_run
from eis.storage import (
    PersistedRunArtifacts,
    RunFolderLayout,
    build_artifact_link_payload,
    build_metadata_bank,
    create_run_folder_layout,
    persist_run_artifacts,
    write_description_file,
    write_metadata_bank_csv,
    write_metadata_bank_txt,
    write_metadata_report_html,
    write_metadata_report_pdf,
)


@dataclass(frozen=True)
class RunSaveOptions:
    """Flags controlling which metadata/report artifacts are written.

    Fields:
        write_metadata_bank_txt: If true, write machine-readable metadata bank
            JSON to ``metadata_bank.txt``.
        write_metadata_bank_csv: If true, write per-capture metadata table CSV.
        write_metadata_report_html: If true, write HTML metadata report.
        write_metadata_report_pdf: If true, write PDF metadata report.
        write_description_file: If true, write ``description.txt`` when user
            description is non-empty.
    """

    write_metadata_bank_txt: bool = True
    write_metadata_bank_csv: bool = True
    write_metadata_report_html: bool = True
    write_metadata_report_pdf: bool = False
    write_description_file: bool = True


@dataclass(frozen=True)
class RunExecutionBundle:
    """Combined output from one full run/process/save operation.

    Fields:
        layout: Created run folder paths.
        run_result: In-memory acquisition captures for this run.
        impedance_results: Per-capture impedance output rows.
        persisted_artifacts: RAW/IMPEDANCE artifact linkage records.
        metadata_bank: Metadata bank dictionary used for reports/regeneration.
        saved_paths: Paths written by metadata/report save options.
        capture_frequency_map: Lookup map from ``(row_number, repeat_index)``
            to frequency, useful for debug plotting selectors.
    """

    layout: RunFolderLayout
    run_result: SweepRunResult
    impedance_results: tuple[ImpedancePointResult, ...]
    persisted_artifacts: PersistedRunArtifacts
    metadata_bank: dict[str, Any]
    saved_paths: tuple[Path, ...]
    capture_frequency_map: dict[tuple[int, int], float]


def _resolve_preflight_sample_count(
    *,
    sample_rate_sps: float,
    samples_per_channel: int | None,
    settle_discard_s: float,
) -> int:
    """Resolve preflight sample count with settle-window safety margin."""

    if settle_discard_s < 0:
        raise ValueError("settle_discard_s must be >= 0.")
    required_settle_samples = int(round(settle_discard_s * sample_rate_sps))
    auto_analysis_samples = max(64, int(round(0.02 * sample_rate_sps)))
    chosen = (
        int(samples_per_channel)
        if samples_per_channel is not None
        else (required_settle_samples + auto_analysis_samples)
    )
    if chosen <= required_settle_samples:
        raise ValueError(
            "samples_per_channel is too small for requested settle_discard_s."
        )
    return chosen


def run_preflight_only(
    *,
    sweep: SweepConfig,
    hardware: HardwareConfig,
    excitation: ExcitationConfig,
    sample_rate_sps: float | None = None,
    samples_per_channel: int | None = None,
    test_current_rms_a: float = 10.0,
    manual_current_range: str | None = None,
    shunt_resistance_ohm: float = 0.008,
    shunt_voltage_tolerance_percent: float = 15.0,
    current_channel_index: int = 0,
    settle_discard_s: float = 0.15,
    adapter: USB6451Adapter | None = None,
) -> PreflightCheckResult:
    """Run only preflight check using one concise notebook-friendly call.

    Inputs:
        sweep: Validated sweep configuration used for fallback sample rate.
        hardware: Hardware wiring and limits.
        excitation: Excitation settings used for range policy fallback.
        sample_rate_sps: Optional preflight sample rate. If omitted, first sweep
            row sample rate is used.
        samples_per_channel: Optional preflight sample count per channel.
        test_current_rms_a: Current target used to generate preflight AO DC.
        manual_current_range: Optional fixed transconductance range label.
        shunt_resistance_ohm: Nominal shunt resistance for expected voltage.
        shunt_voltage_tolerance_percent: Allowed error around expected shunt
            voltage expressed in percent.
        current_channel_index: AI channel index interpreted as shunt channel.
        settle_discard_s: Initial time window discarded before validation.
        adapter: Optional externally managed USB6451 adapter instance.
    Output:
        ``PreflightCheckResult`` with DAQ validation summary.
    """

    if not sweep.points:
        raise ValueError("Sweep configuration contains no points.")

    chosen_rate = (
        float(sample_rate_sps)
        if sample_rate_sps is not None
        else float(sweep.points[0].sample_rate_sps)
    )
    chosen_samples = _resolve_preflight_sample_count(
        sample_rate_sps=chosen_rate,
        samples_per_channel=samples_per_channel,
        settle_discard_s=settle_discard_s,
    )
    effective_range = (
        manual_current_range
        if manual_current_range is not None
        else excitation.manual_current_range
    )

    local_adapter = adapter or USB6451Adapter()
    owns_adapter = adapter is None
    try:
        return run_preflight_check(
            adapter=local_adapter,
            hardware=hardware,
            sample_rate_sps=chosen_rate,
            samples_per_channel=chosen_samples,
            test_current_rms_a=test_current_rms_a,
            manual_current_range=effective_range,
            range_selection_policy=excitation.range_selection_policy,
            shunt_resistance_ohm=shunt_resistance_ohm,
            shunt_voltage_tolerance_percent=shunt_voltage_tolerance_percent,
            current_channel_index=current_channel_index,
            settle_discard_s=settle_discard_s,
        )
    finally:
        if owns_adapter:
            local_adapter.close()


def run_measure_process_save(
    *,
    sweep: SweepConfig,
    hardware: HardwareConfig,
    excitation: ExcitationConfig,
    processing: ImpedanceProcessingConfig,
    base_output_dir: str | Path,
    serial_number: str,
    user_name: str,
    description: str | None,
    repeats: int = 1,
    run_preflight_during_sweep: bool = True,
    preflight_sample_rate_sps: float | None = None,
    preflight_samples_per_channel: int | None = None,
    preflight_test_current_rms_a: float = 10.0,
    preflight_manual_current_range: str | None = None,
    preflight_shunt_resistance_ohm: float = 0.008,
    preflight_shunt_voltage_tolerance_percent: float = 15.0,
    preflight_current_channel_index: int = 0,
    preflight_settle_discard_s: float = 0.15,
    conditioning: CaptureConditioningConfig | None = None,
    save_options: RunSaveOptions | None = None,
    started_at_local: datetime | None = None,
    adapter: USB6451Adapter | None = None,
) -> RunExecutionBundle:
    """Run sweep, process impedance, persist artifacts, and write metadata.

    Inputs:
        sweep: Validated sweep configuration.
        hardware: Hardware wiring and limits.
        excitation: Excitation settings for AO/current-range conversion.
        processing: Signal-processing configuration for impedance extraction.
        base_output_dir: Root folder where run directory is created.
        serial_number: User serial number for folder naming.
        user_name: Operator/user name for metadata.
        description: Optional free-text description.
        repeats: Repeat count for each sweep row.
        run_preflight_during_sweep: If true, execute preflight before sweep.
        preflight_*: Preflight options forwarded to sweep orchestration.
        conditioning: Settling discard and periodic trim strategy.
        save_options: Metadata/report write options.
        started_at_local: Optional folder timestamp override.
        adapter: Optional externally managed USB6451 adapter instance.
    Output:
        ``RunExecutionBundle`` containing in-memory data and written-path links.
    """

    options = save_options or RunSaveOptions()
    run_start_local = started_at_local or datetime.now()

    local_adapter = adapter or USB6451Adapter()
    owns_adapter = adapter is None
    try:
        run_result = execute_sweep(
            sweep=sweep,
            adapter=local_adapter,
            hardware=hardware,
            excitation=excitation,
            repeats=repeats,
            run_preflight=run_preflight_during_sweep,
            preflight_sample_rate_sps=preflight_sample_rate_sps,
            preflight_samples_per_channel=preflight_samples_per_channel,
            preflight_test_current_rms_a=preflight_test_current_rms_a,
            preflight_manual_current_range=preflight_manual_current_range,
            preflight_shunt_resistance_ohm=preflight_shunt_resistance_ohm,
            preflight_shunt_voltage_tolerance_percent=(
                preflight_shunt_voltage_tolerance_percent
            ),
            preflight_current_channel_index=preflight_current_channel_index,
            preflight_settle_discard_s=preflight_settle_discard_s,
            conditioning=conditioning,
        )
    finally:
        if owns_adapter:
            local_adapter.close()

    impedance_results = compute_impedance_for_run(
        run_result=run_result,
        config=processing,
    )
    layout = create_run_folder_layout(
        base_output_dir=base_output_dir,
        serial_number=serial_number,
        started_at_local=run_start_local,
    )
    persisted = persist_run_artifacts(
        layout=layout,
        run_result=run_result,
        impedance_results=impedance_results,
    )
    capture_artifacts, point_summaries = build_artifact_link_payload(persisted)

    metadata_bank = build_metadata_bank(
        sweep=sweep,
        run_result=run_result,
        hardware=hardware,
        excitation=excitation,
        serial_number=serial_number,
        user_name=user_name,
        description=description,
        capture_artifacts=capture_artifacts,
        point_summaries=point_summaries,
    )

    saved_paths: list[Path] = []
    if options.write_metadata_bank_txt:
        saved_paths.append(
            write_metadata_bank_txt(
                metadata_bank,
                layout.root / "metadata_bank.txt",
            )
        )
    if options.write_metadata_bank_csv:
        saved_paths.append(
            write_metadata_bank_csv(
                metadata_bank,
                layout.root / "metadata_measurements.csv",
            )
        )
    if options.write_metadata_report_html:
        saved_paths.append(
            write_metadata_report_html(
                metadata_bank,
                layout.reports / "metadata_report.html",
            )
        )
    if options.write_metadata_report_pdf:
        saved_paths.append(
            write_metadata_report_pdf(
                metadata_bank,
                layout.reports / "metadata_report.pdf",
            )
        )
    if options.write_description_file:
        description_path = write_description_file(
            description,
            layout.root / "description.txt",
        )
        if description_path is not None:
            saved_paths.append(description_path)

    capture_frequency_map = {
        (capture.row_number, capture.repeat_index): float(capture.frequency_hz)
        for capture in run_result.captures
    }
    return RunExecutionBundle(
        layout=layout,
        run_result=run_result,
        impedance_results=tuple(impedance_results),
        persisted_artifacts=persisted,
        metadata_bank=metadata_bank,
        saved_paths=tuple(saved_paths),
        capture_frequency_map=capture_frequency_map,
    )
