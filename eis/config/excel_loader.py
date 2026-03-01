"""Excel loader for EIS measurement configuration tables.

This module intentionally avoids external dependencies so the project can be run in
lab environments where only Python standard library is available.
"""

from __future__ import annotations

from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

from eis.models.config_models import RawConfigRow, RawConfigTable

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS = {"main": _MAIN_NS, "rel": _REL_NS}
_CELL_REF_PATTERN = re.compile(r"^([A-Z]+)([0-9]+)$")


def load_config_table(xlsx_path: str | Path, sheet_name: str | None = None) -> RawConfigTable:
    """Load a raw measurement config table from an Excel ``.xlsx`` file.

    Inputs:
        xlsx_path: Path to the ``.xlsx`` configuration file.
        sheet_name: Optional sheet name. If omitted, first workbook sheet is used.
    Output:
        ``RawConfigTable`` containing headers and row dictionaries.
    Raises:
        FileNotFoundError: Config file is not found.
        ValueError: Workbook is malformed or missing tabular config content.
    """

    path = Path(xlsx_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file was not found: {path}")
    if path.suffix.lower() != ".xlsx":
        raise ValueError(f"Expected '.xlsx' file, got: {path}")

    with zipfile.ZipFile(path) as archive:
        shared_strings = _read_shared_strings(archive)
        sheets = _read_sheet_index(archive)

        if not sheets:
            raise ValueError("Workbook does not contain any sheets.")

        selected_sheet_name, selected_sheet_path = _select_sheet(
            sheets=sheets,
            requested_sheet_name=sheet_name,
        )

        rows_by_index = _read_sheet_rows(
            archive=archive,
            sheet_path=selected_sheet_path,
            shared_strings=shared_strings,
        )

    if not rows_by_index:
        raise ValueError(
            f"Sheet '{selected_sheet_name}' in '{path}' is empty. "
            "Expected a header row and measurement data rows."
        )

    header_row_index = min(rows_by_index)
    headers_by_col = _build_headers(rows_by_index[header_row_index])
    if not headers_by_col:
        raise ValueError(
            f"Sheet '{selected_sheet_name}' in '{path}' does not contain usable headers."
        )

    headers = tuple(headers_by_col[col_index] for col_index in sorted(headers_by_col))
    data_rows = _build_data_rows(
        rows_by_index=rows_by_index,
        header_row_index=header_row_index,
        headers_by_col=headers_by_col,
    )

    if not data_rows:
        raise ValueError(
            f"Sheet '{selected_sheet_name}' in '{path}' does not contain measurement rows "
            "below the header."
        )

    return RawConfigTable(
        source_path=path,
        sheet_name=selected_sheet_name,
        headers=headers,
        rows=tuple(data_rows),
    )


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    """Read workbook shared string table."""

    shared_path = "xl/sharedStrings.xml"
    if shared_path not in archive.namelist():
        return []

    root = ET.fromstring(archive.read(shared_path))
    strings: list[str] = []

    for si in root.findall("main:si", _NS):
        text_parts = [node.text or "" for node in si.findall(".//main:t", _NS)]
        strings.append("".join(text_parts))

    return strings


def _read_sheet_index(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Return list of workbook sheets as ``(name, xml_path)``."""

    workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
    rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))

    rel_map: dict[str, str] = {}
    for rel in rels_root.findall("rel:Relationship", _NS):
        rel_id = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if rel_id and target:
            rel_map[rel_id] = target

    sheets: list[tuple[str, str]] = []
    for sheet in workbook_root.findall("main:sheets/main:sheet", _NS):
        name = sheet.attrib.get("name")
        rel_id = sheet.attrib.get(f"{{{_DOC_REL_NS}}}id")
        if not name or not rel_id or rel_id not in rel_map:
            continue

        target = rel_map[rel_id]
        sheet_path = target if target.startswith("xl/") else f"xl/{target}"
        sheets.append((name, sheet_path))

    return sheets


def _select_sheet(
    *,
    sheets: list[tuple[str, str]],
    requested_sheet_name: str | None,
) -> tuple[str, str]:
    """Select target sheet by requested name or first sheet."""

    if requested_sheet_name is None:
        return sheets[0]

    requested_normalized = requested_sheet_name.strip().lower()
    for name, path in sheets:
        if name.strip().lower() == requested_normalized:
            return name, path

    available = ", ".join(name for name, _ in sheets)
    raise ValueError(
        f"Sheet '{requested_sheet_name}' was not found. Available sheets: {available}."
    )


def _read_sheet_rows(
    *,
    archive: zipfile.ZipFile,
    sheet_path: str,
    shared_strings: list[str],
) -> dict[int, dict[int, object]]:
    """Read all non-empty cells from a worksheet."""

    root = ET.fromstring(archive.read(sheet_path))
    rows_by_index: dict[int, dict[int, object]] = {}

    for row_node in root.findall(".//main:sheetData/main:row", _NS):
        row_index_raw = row_node.attrib.get("r")
        if row_index_raw is None:
            continue

        row_index = int(row_index_raw)
        row_values: dict[int, object] = {}

        for cell_node in row_node.findall("main:c", _NS):
            reference = cell_node.attrib.get("r", "")
            col_index = _column_index_from_cell_reference(reference)
            if col_index is None:
                continue

            value = _decode_cell_value(cell_node, shared_strings)
            if value is None:
                continue

            row_values[col_index] = value

        if row_values:
            rows_by_index[row_index] = row_values

    return rows_by_index


def _decode_cell_value(cell_node: ET.Element, shared_strings: list[str]) -> object | None:
    """Decode one Excel cell value using shared string and primitive coercion."""

    cell_type = cell_node.attrib.get("t")

    if cell_type == "inlineStr":
        inline_text = cell_node.find("main:is/main:t", _NS)
        if inline_text is None or inline_text.text is None:
            return None
        return inline_text.text.strip() or None

    value_node = cell_node.find("main:v", _NS)
    if value_node is None or value_node.text is None:
        return None

    raw_text = value_node.text.strip()
    if raw_text == "":
        return None

    if cell_type == "s":
        index = int(raw_text)
        if index < 0 or index >= len(shared_strings):
            return raw_text
        return shared_strings[index]

    if cell_type == "b":
        return raw_text == "1"

    try:
        numeric = float(raw_text)
    except ValueError:
        return raw_text

    return numeric


def _column_index_from_cell_reference(reference: str) -> int | None:
    """Convert Excel cell reference (for example ``B12``) to 1-based column index."""

    match = _CELL_REF_PATTERN.match(reference)
    if match is None:
        return None

    col_label = match.group(1)
    value = 0
    for char in col_label:
        value = value * 26 + (ord(char) - 64)

    return value


def _build_headers(header_row: dict[int, object]) -> dict[int, str]:
    """Build header name map keyed by column index."""

    headers: dict[int, str] = {}
    for col_index, raw_value in header_row.items():
        text = str(raw_value).strip()
        if text:
            headers[col_index] = text

    return headers


def _build_data_rows(
    *,
    rows_by_index: dict[int, dict[int, object]],
    header_row_index: int,
    headers_by_col: dict[int, str],
) -> list[RawConfigRow]:
    """Map worksheet rows to header-based dictionaries."""

    data_rows: list[RawConfigRow] = []
    sorted_columns = sorted(headers_by_col)

    for row_index in sorted(rows_by_index):
        if row_index <= header_row_index:
            continue

        raw_cells = rows_by_index[row_index]
        row_values: dict[str, object] = {}
        has_user_data = False

        for col_index in sorted_columns:
            header = headers_by_col[col_index]
            value = raw_cells.get(col_index)
            row_values[header] = value
            if value is not None and str(value).strip() != "":
                has_user_data = True

        if has_user_data:
            data_rows.append(RawConfigRow(row_number=row_index, values=row_values))

    return data_rows
