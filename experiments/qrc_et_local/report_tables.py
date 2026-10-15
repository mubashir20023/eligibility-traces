"""Readable W&B HTML-media tables derived from the report's reviewed Markdown."""
import hashlib
import html
import math
import re


def split_tables(parts):
    result, heading, counts = [], "Comparison", {}
    for part in parts:
        if part["kind"] != "markdown":
            result.append(part)
            continue
        lines, pending, index = part["text"].splitlines(), [], 0
        while index < len(lines):
            line = lines[index]
            if line.startswith("#"):
                heading = line.lstrip("# ").strip()
            if (line.startswith("|") and index + 1 < len(lines)
                    and re.fullmatch(r"[| :\-]+", lines[index + 1])):
                if "\n".join(pending).strip():
                    result.append({"kind": "markdown", "text": "\n".join(pending).strip()})
                pending = []
                headers = [c.strip() for c in line.strip("|").split("|")]
                index += 2
                rows = []
                while index < len(lines) and lines[index].startswith("|"):
                    row = [c.strip() for c in lines[index].strip("|").split("|")]
                    if len(row) != len(headers):
                        raise ValueError(f"Table column mismatch: {heading}")
                    rows.append(row)
                    index += 1
                slug = re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_")
                counts[slug] = counts.get(slug, 0) + 1
                result.append({"kind": "table", "title": heading,
                               "key": f"tables/{slug}_{counts[slug]}",
                               "headers": headers, "rows": rows})
                continue
            pending.append(line)
            index += 1
        if "\n".join(pending).strip():
            result.append({"kind": "markdown", "text": "\n".join(pending).strip()})
    return result


def table_html(part):
    def clean(value):
        return html.escape(str(value).replace("**", "").replace("`", ""))
    headers = part["headers"]
    numeric_columns = []
    for index in range(len(headers)):
        values = [row[index] for row in part["rows"] if row[index] not in {"", "N/A"}]
        numeric_columns.append(bool(values) and all(re.fullmatch(r"[+\-]?[\d,.]+", v) for v in values))
    cells = "".join(f'<th scope="col" class="{"numeric" if numeric else "text"}">{clean(h)}</th>'
                    for h, numeric in zip(headers, numeric_columns))
    rows = []
    for row in part["rows"]:
        rendered = []
        for header, value in zip(headers, row):
            numeric = bool(re.fullmatch(r"[+\-]?[\d,.]+", value))
            classes = ["numeric"] if numeric else []
            if header == "ET - QRC" and value.startswith(("+", "-")):
                classes.append("positive" if value.startswith("+") else "negative")
            if value in {"QRC", "Experimental ET", "Experimental QRC-ET"}:
                classes.append("qrc" if value == "QRC" else "et")
            rendered.append(f'<td data-label="{clean(header)}" class="{" ".join(classes)}">{clean(value)}</td>')
        rows.append("<tr>" + "".join(rendered) + "</tr>")
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>""" + clean(part["title"]) + """</title><style>
*{box-sizing:border-box}html,body{margin:0;padding:0;background:#fff;color:#202933;
font:14px/1.5 Arial,Helvetica,sans-serif;letter-spacing:0}
main{padding:10px 14px 16px;max-width:100%;width:100%}
table{border-collapse:separate;border-spacing:0;width:100%;table-layout:auto;
border:1px solid #cdd6df;border-radius:4px;overflow:hidden}
thead{background:#eef3f7}th{font-size:13px;font-weight:700;color:#344453;
padding:13px 14px;text-align:left;vertical-align:middle;border-bottom:2px solid #c8d3de}
th.numeric{text-align:right}
td{padding:12px 14px;border-bottom:1px solid #e0e6ed;vertical-align:middle;overflow-wrap:anywhere}
tbody tr:nth-child(even){background:#f7f9fb}tbody tr:last-child td{border-bottom:0}
td:first-child{font-weight:600}th,td{min-width:0}td.numeric{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.positive,.et{color:#006e58;font-weight:700}.negative{color:#a52e46;font-weight:700}.qrc{color:#245cc4;font-weight:700}
@media(max-width:900px){main{padding:6px}th,td{padding:10px 8px}}
@media(max-width:620px){thead{display:none}table,tbody,tr,td{display:block;width:100%}
table{border:0;border-radius:0}tr{border:1px solid #cdd6df;margin-bottom:12px}
td{display:grid;grid-template-columns:minmax(110px,42%) minmax(0,1fr);gap:12px;text-align:right;padding:8px 12px}
td:before{content:attr(data-label);font-weight:600;text-align:left;color:#526170}
td.numeric{white-space:normal}tbody tr:last-child td,tbody td:last-child{border-bottom:0}}
</style></head><body><main><table aria-label=""" + '"' + clean(part["title"]) + '"' + ">" + \
        "<thead><tr>" + cells + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></main></body></html>"


def table_height(part):
    # Include W&B's media-panel header and controls above the actual table.
    row_height = sum(max(47, 24 + math.ceil(max(map(len, row)) / 38) * 21)
                     for row in part["rows"])
    return max(8, (row_height + 165 + 31) // 32)


def presentation_digest(tables):
    return hashlib.sha256("\n".join(table_html(t) for t in tables).encode("utf-8")).hexdigest()
