from __future__ import annotations

import html
import re
from html.parser import HTMLParser


def normalize_table_html(table_html: str) -> dict:
    parser = _TableParser()
    parser.feed(table_html)
    rows = _pad_rows(parser.rows)
    markdown = _to_markdown(rows)
    latex = _to_latex(rows)
    return {
        "grid": rows,
        "normalized_markdown": markdown,
        "normalized_latex": latex,
        "table_shape": {
            "rows": len(rows),
            "columns": max((len(row) for row in rows), default=0),
        },
    }


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.rows: list[list[str]] = []
        self._pending_rowspans: dict[int, dict] = {}
        self._current_row: list[str] | None = None
        self._col_index = 0
        self._in_cell = False
        self._cell_text: list[str] = []
        self._cell_rowspan = 1
        self._cell_colspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_l = tag.lower()
        if tag_l == "tr":
            self._current_row = []
            self._col_index = 0
            return
        if tag_l in {"td", "th"}:
            if self._current_row is None:
                self._current_row = []
            attr_map = {key.lower(): value for key, value in attrs}
            self._fill_rowspans_until_free()
            self._in_cell = True
            self._cell_text = []
            self._cell_rowspan = _positive_int(attr_map.get("rowspan"), 1)
            self._cell_colspan = _positive_int(attr_map.get("colspan"), 1)

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        if tag_l in {"td", "th"} and self._in_cell:
            text = _clean_cell_text("".join(self._cell_text))
            start_col = self._col_index
            for offset in range(self._cell_colspan):
                self._current_row.append(text)
                if self._cell_rowspan > 1:
                    self._pending_rowspans[start_col + offset] = {
                        "text": text,
                        "rows_left": self._cell_rowspan - 1,
                    }
            self._col_index += self._cell_colspan
            self._in_cell = False
            self._cell_text = []
            return
        if tag_l == "tr" and self._current_row is not None:
            self._fill_trailing_rowspans()
            self.rows.append(self._current_row)
            self._current_row = None
            self._col_index = 0

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_text.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._in_cell:
            self._cell_text.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._in_cell:
            self._cell_text.append(f"&#{name};")

    def _fill_rowspans_until_free(self) -> None:
        while self._col_index in self._pending_rowspans:
            pending = self._pending_rowspans[self._col_index]
            self._current_row.append(pending["text"])
            pending["rows_left"] -= 1
            if pending["rows_left"] <= 0:
                del self._pending_rowspans[self._col_index]
            self._col_index += 1

    def _fill_trailing_rowspans(self) -> None:
        while self._col_index in self._pending_rowspans:
            self._fill_rowspans_until_free()


def _pad_rows(rows: list[list[str]]) -> list[list[str]]:
    width = max((len(row) for row in rows), default=0)
    return [row + [""] * (width - len(row)) for row in rows]


def _to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    escaped = [[_escape_md_cell(cell) for cell in row] for row in rows]
    header = escaped[0]
    separator = ["---"] * len(header)
    body = escaped[1:]
    md_rows = [header, separator] + body
    return "\n".join("| " + " | ".join(row) + " |" for row in md_rows)


def _escape_md_cell(text: str) -> str:
    return text.replace("|", "\\|")


def _to_latex(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max((len(row) for row in rows), default=0)
    if width <= 0:
        return ""
    lines = [f"\\begin{{tabular}}{{{'l' * width}}}", "\\hline"]
    for idx, row in enumerate(rows):
        cells = [_escape_latex_cell(cell) for cell in row]
        lines.append(" & ".join(cells) + r" \\")
        if idx == 0:
            lines.append("\\hline")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def _escape_latex_cell(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _clean_cell_text(text: str) -> str:
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _positive_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default
