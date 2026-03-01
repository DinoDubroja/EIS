"""Clarke-Hess 8100 transconductance conversion and range selection.

This module stores range definitions from the Clarke-Hess 8100 datasheet and
provides deterministic conversion from target output current (A RMS) to required
AO drive amplitude (V peak). It also resolves current range selection policy,
including manual fixed-range operation.
"""

from __future__ import annotations

from math import sqrt

from eis.models.measurement_models import DriveAmplitudeResult, TransconductanceRange

# Clarke-Hess 8100 table values from "Clarke Hess 8100 Datsheet.pdf":
# - Range and transconductance table
# - 100A range full scale at 1 Vrms input
# - Other ranges full scale at 2 Vrms input and can operate to 200% full scale
DEFAULT_8100_RANGES: tuple[TransconductanceRange, ...] = (
    TransconductanceRange(
        name="2mA",
        transconductance_siemens=1e-3,
        min_current_rms_a=0.2e-3,
        full_scale_current_rms_a=2e-3,
        max_current_rms_a=4e-3,
        input_full_scale_vrms=2.0,
    ),
    TransconductanceRange(
        name="20mA",
        transconductance_siemens=1e-2,
        min_current_rms_a=2e-3,
        full_scale_current_rms_a=20e-3,
        max_current_rms_a=40e-3,
        input_full_scale_vrms=2.0,
    ),
    TransconductanceRange(
        name="0.2A",
        transconductance_siemens=1e-1,
        min_current_rms_a=20e-3,
        full_scale_current_rms_a=0.2,
        max_current_rms_a=0.4,
        input_full_scale_vrms=2.0,
    ),
    TransconductanceRange(
        name="2A",
        transconductance_siemens=1.0,
        min_current_rms_a=0.2,
        full_scale_current_rms_a=2.0,
        max_current_rms_a=4.0,
        input_full_scale_vrms=2.0,
    ),
    TransconductanceRange(
        name="20A",
        transconductance_siemens=10.0,
        min_current_rms_a=2.0,
        full_scale_current_rms_a=20.0,
        max_current_rms_a=40.0,
        input_full_scale_vrms=2.0,
    ),
    TransconductanceRange(
        name="100A",
        transconductance_siemens=100.0,
        min_current_rms_a=20.0,
        full_scale_current_rms_a=100.0,
        max_current_rms_a=100.0,
        input_full_scale_vrms=1.0,
    ),
)


def compute_drive_amplitude_from_current(
    *,
    current_rms_a: float,
    ranges: tuple[TransconductanceRange, ...] = DEFAULT_8100_RANGES,
    manual_range_name: str | None = None,
    selection_policy: str = "prefer_no_overrange",
) -> DriveAmplitudeResult:
    """Compute AO drive amplitude from target current RMS using range selection.

    Inputs:
        current_rms_a: Target output current in amperes RMS (A).
        ranges: Available transconductance ranges.
        manual_range_name: Optional fixed range label, for example ``"20A"``.
        selection_policy: Auto range policy. Current supported value:
            ``"prefer_no_overrange"``.
    Output:
        ``DriveAmplitudeResult`` containing selected range and AO amplitude.
    Raises:
        ValueError: Target current is invalid or cannot be represented.
    """

    if current_rms_a <= 0:
        raise ValueError("Current_rms must be > 0 A.")
    if not ranges:
        raise ValueError("At least one transconductance range must be provided.")

    if manual_range_name is not None:
        selected = _select_manual_range(
            current_rms_a=current_rms_a,
            ranges=ranges,
            manual_range_name=manual_range_name,
        )
    else:
        selected = _select_auto_range(
            current_rms_a=current_rms_a,
            ranges=ranges,
            selection_policy=selection_policy,
        )

    ao_input_vrms = current_rms_a / selected.transconductance_siemens
    ao_amplitude_v_peak = ao_input_vrms * sqrt(2.0)
    is_overrange = current_rms_a > selected.full_scale_current_rms_a

    return DriveAmplitudeResult(
        range_name=selected.name,
        transconductance_siemens=selected.transconductance_siemens,
        current_rms_a=current_rms_a,
        ao_input_vrms=ao_input_vrms,
        ao_amplitude_v_peak=ao_amplitude_v_peak,
        is_overrange=is_overrange,
    )


def _select_manual_range(
    *,
    current_rms_a: float,
    ranges: tuple[TransconductanceRange, ...],
    manual_range_name: str,
) -> TransconductanceRange:
    """Select user-requested range and validate current support."""

    normalized_requested = manual_range_name.strip().lower()
    for range_item in ranges:
        if range_item.name.strip().lower() != normalized_requested:
            continue
        if current_rms_a < range_item.min_current_rms_a or current_rms_a > range_item.max_current_rms_a:
            raise ValueError(
                f"Current_rms={current_rms_a:g} A is outside selected range '{range_item.name}' "
                f"support [{range_item.min_current_rms_a:g}, {range_item.max_current_rms_a:g}] A."
            )
        return range_item

    available = ", ".join(item.name for item in ranges)
    raise ValueError(f"Unknown manual current range '{manual_range_name}'. Available: {available}.")


def _select_auto_range(
    *,
    current_rms_a: float,
    ranges: tuple[TransconductanceRange, ...],
    selection_policy: str,
) -> TransconductanceRange:
    """Select range automatically from target current and policy."""

    if selection_policy != "prefer_no_overrange":
        raise ValueError(
            "Unsupported range_selection_policy. "
            "Supported values: prefer_no_overrange."
        )

    in_supported = [
        item
        for item in ranges
        if item.min_current_rms_a <= current_rms_a <= item.max_current_rms_a
    ]
    if not in_supported:
        min_supported = min(item.min_current_rms_a for item in ranges)
        max_supported = max(item.max_current_rms_a for item in ranges)
        raise ValueError(
            f"Current_rms={current_rms_a:g} A is outside supported Clarke-Hess 8100 range "
            f"[{min_supported:g}, {max_supported:g}] A."
        )

    within_full_scale = [
        item
        for item in in_supported
        if current_rms_a <= item.full_scale_current_rms_a
    ]
    if within_full_scale:
        # Pick smallest suitable full-scale range to maximize drive resolution.
        return sorted(within_full_scale, key=lambda item: item.full_scale_current_rms_a)[0]

    # Fallback: all candidates are overrange; pick the smallest supporting range.
    return sorted(in_supported, key=lambda item: item.max_current_rms_a)[0]
