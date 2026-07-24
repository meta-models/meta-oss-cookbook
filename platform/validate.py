"""
validate.py — check partner vendor pages before they're sent, and before they're merged.

    python platform/validate.py                # every vendor page in platform/
    python platform/validate.py nvidia.md      # one page (a bare name also works)
    python platform/validate.py --index        # print the index table for platform/README.md
    python platform/validate.py --write-index  # splice that table into platform/README.md

A contribution is one file, `platform/<vendor>.md`. This checks its frontmatter,
section structure, status vocabulary, precision naming, and tables. Stdlib only,
no network.

Exit code 0 = clean, 1 = at least one error.
"""
from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------------------
# The contract. Everything below is what "consistent across partners" actually means.
# --------------------------------------------------------------------------------------
REQUIRED_SECTIONS = ["## Platforms", "## Performance", "## Platform details", "## Support"]

# Inside each `### <Platform>` block, in this order.
REQUIRED_BLOCKS = ["**Snapshot**", "**Prerequisites**", "**Deploy**",
                   "**Supported precisions**", "**Troubleshooting**"]

REQUIRED_FRONTMATTER = ["vendor", "contact", "updated"]

PLATFORMS_COLUMNS = ["Platform", "Class", "Accelerator", "Memory", "Precisions verified"]

CLASSES = {"consumer", "workstation", "server", "edge", "cloud-instance"}

STATUSES = {"Verified", "Works, not verified", "Planned", "Not supported"}

# Examples, not a closed set: partners use the artifact's published name, and new
# quantization schemes appear faster than this list can track them. An unrecognized
# label is a warning (likely a typo or a casing slip), never an error.
KNOWN_PRECISIONS = {
    "bf16", "fp16", "fp8", "int8", "int4", "nvfp4", "mxfp4", "mxfp8",
    "awq-int4", "gptq-int4", "q4_k_m", "q5_k_m", "q8_0",
    "mlx-4bit", "mlx-8bit",
}

# Literal strings from the template. If any survive, the page wasn't filled in.
TEMPLATE_MARKERS = [
    "<Your Org>", "<Platform A>", "<Platform B>", "<YYYY-MM-DD>", "<url>",
    "<chip name", "<Name (", "<128 GB unified>", "<workstation>", "<consumer>",
    "One file per organization. Rename this",
]

INDEX_BEGIN = "<!-- BEGIN:INDEX"
INDEX_END = "<!-- END:INDEX -->"


class Report:
    def __init__(self) -> None:
        self.items: list[tuple[str, str, str]] = []

    def error(self, where: Path | str, msg: str) -> None:
        self._add("error", where, msg)

    def warn(self, where: Path | str, msg: str) -> None:
        self._add("warn", where, msg)

    def _add(self, level: str, where: Path | str, msg: str) -> None:
        item = (level, str(where), msg)
        if item not in self.items:  # the same mistake repeated in one file is one finding
            self.items.append(item)

    @property
    def errors(self) -> int:
        return sum(1 for level, _, _ in self.items if level == "error")

    @property
    def warnings(self) -> int:
        return sum(1 for level, _, _ in self.items if level == "warn")

    def print(self) -> None:
        grouped: dict[str, list[tuple[str, str]]] = {}
        for level, where, msg in self.items:
            grouped.setdefault(where, []).append((level, msg))
        for where, found in grouped.items():
            print(f"\n{where}")
            for level, msg in found:
                print(f"  {'ERROR' if level == 'error' else 'warn '}  {msg}")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PLATFORM_DIR.parent))
    except ValueError:
        return str(path)


def slugify(text: str) -> str:
    """GitHub-style anchor slug, also used for vendor filenames."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------
def parse_frontmatter(text: str) -> dict[str, str] | None:
    """Minimal YAML-ish frontmatter: `key: value`."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    out: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return out
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip()] = value.strip().strip('"').strip("'")
    return None  # no closing delimiter


def fenced(lines: list[str]) -> list[bool]:
    """Mark lines inside ``` / ~~~ code fences.

    Serve commands routinely contain `# comment` and `|` at the start of a line, which
    would otherwise read as a heading or a table row and cut a section short.
    """
    mask, marker = [], ""
    for line in lines:
        stripped = line.strip()
        if not marker and (stripped.startswith("```") or stripped.startswith("~~~")):
            marker = stripped[:3]
            mask.append(True)
        elif marker and stripped.startswith(marker):
            marker = ""
            mask.append(True)
        else:
            mask.append(bool(marker))
    return mask


def heading_level(line: str) -> int:
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return 0
    level = len(stripped) - len(stripped.lstrip("#"))
    return level if stripped[level:level + 1] == " " else 0


def headings(text: str, level: int) -> list[tuple[int, str]]:
    lines = text.splitlines()
    mask = fenced(lines)
    return [(i, line.strip()) for i, line in enumerate(lines)
            if not mask[i] and heading_level(line) == level]


def section_text(text: str, heading: str, level: int = 2) -> str | None:
    """Everything under a heading, up to the next heading of the same or higher level."""
    lines = text.splitlines()
    mask = fenced(lines)
    start = next((i for i, line in enumerate(lines)
                  if not mask[i] and heading_level(line) == level
                  and line.strip().startswith(heading)), None)
    if start is None:
        return None
    out = []
    for i in range(start + 1, len(lines)):
        found = 0 if mask[i] else heading_level(lines[i])
        if found and found <= level:
            break
        out.append(lines[i])
    return "\n".join(out)


def table_rows(chunk: str, after: str | None = None) -> list[list[str]]:
    """Rows of the first markdown table in `chunk`, or of the one following `after`."""
    lines = chunk.splitlines()
    mask = fenced(lines)
    if after is not None:
        start = next((i for i, line in enumerate(lines)
                      if not mask[i] and line.strip() == after), None)
        if start is None:
            return []
        lines, mask = lines[start + 1:], mask[start + 1:]
    rows, seen_table = [], False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if mask[i]:
            continue
        if stripped.startswith("|"):
            seen_table = True
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            rows.append(cells)
        elif seen_table and stripped and not stripped.startswith("<!--"):
            break  # table ended
    return rows


def is_iso_date(value: str) -> bool:
    try:
        datetime.date.fromisoformat(value)
        return True
    except ValueError:
        return False


def check_precision(label: str, where: str, context: str, report: Report) -> None:
    """Nudge toward the published name and lowercase, without blocking new schemes.

    Compound labels are common (weights in one format, activations in another), so
    `nvfp4/mxfp8` is checked a part at a time.
    """
    for part in (p.strip() for p in label.split("/")):
        if not part or part in KNOWN_PRECISIONS:
            continue
        if part.lower() in KNOWN_PRECISIONS:
            report.warn(where, f"{context}: write precision `{part}` in lowercase "
                               f"(`{part.lower()}`) so pages stay comparable")
        else:
            report.warn(where, f"{context}: precision `{part}` isn't a label we've seen before "
                               f"— check it matches the artifact's published name")


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------
def check_vendor_page(path: Path, report: Report) -> dict | None:
    text = path.read_text(encoding="utf-8")
    where = rel(path)

    fm = parse_frontmatter(text)
    if fm is None:
        report.error(where, "no frontmatter block (file must open with `---` and close with `---`)")
        return None
    for key in REQUIRED_FRONTMATTER:
        if not fm.get(key):
            report.error(where, f"frontmatter: missing or empty `{key}`")
    if fm.get("updated") and not is_iso_date(fm["updated"]):
        report.error(where, f"frontmatter: updated `{fm['updated']}` is not YYYY-MM-DD")
    vendor = fm.get("vendor", "")
    if vendor and slugify(vendor) != path.stem and not is_scaffolding(path):
        report.warn(where, f"filename `{path.name}` doesn't match vendor `{vendor}` "
                           f"(expected `{slugify(vendor)}.md`)")

    # The four sections, present and in order.
    found = {}
    cursor = -1
    for required in REQUIRED_SECTIONS:
        match = next((i for i, h in headings(text, 2) if i > cursor and h.startswith(required)),
                     None)
        if match is not None:
            found[required] = match
            cursor = match
        elif any(h.startswith(required) for _, h in headings(text, 2)):
            report.error(where, f"section `{required}` is out of order — keep the order in "
                                f"platform/_template/VENDOR_TEMPLATE.md")
        else:
            report.error(where, f"missing section `{required}`")
    for i, heading in headings(text, 2):
        if i not in set(found.values()):
            report.warn(where, f"unexpected section `{heading}` — vendor pages should stick to "
                               f"the four shared sections")

    listed = check_platforms_table(text, where, report)
    check_performance(text, where, report)
    detailed = check_platform_details(text, where, listed, report)

    left = [m for m in TEMPLATE_MARKERS if m in text]
    if left:
        report.error(where, f"{len(left)} piece(s) of unfilled template text still present: "
                            + ", ".join(f"`{m}`" for m in left))

    return {"vendor": vendor, "file": path.name, "platforms": detailed}


def check_platforms_table(text: str, where: str, report: Report) -> list[dict]:
    chunk = section_text(text, "## Platforms")
    if chunk is None:
        return []
    rows = table_rows(chunk)
    if not rows:
        report.error(where, "`## Platforms` has no table — the index is generated from it")
        return []
    if [c.strip() for c in rows[0]] != PLATFORMS_COLUMNS:
        report.error(where, "`## Platforms` header must be exactly: | "
                            + " | ".join(PLATFORMS_COLUMNS) + " |")
    listed = []
    for row in rows[1:]:
        if len(row) < len(PLATFORMS_COLUMNS):
            report.error(where, f"`## Platforms` row `{row[0] if row else ''}` is missing columns")
            continue
        name, klass = row[0].strip("`* "), row[1].strip()
        if klass and klass not in CLASSES:
            report.error(where, f"`## Platforms`: class `{klass}` for {name} not one of "
                                f"{sorted(CLASSES)}")
        for precision in (p.strip() for p in row[4].split(",")):
            if precision:
                check_precision(precision, where, f"`## Platforms` ({name})", report)
        listed.append({"platform": name, "class": klass, "accelerator": row[2].strip(),
                       "memory": row[3].strip(), "precisions": row[4].strip()})
    return listed


def check_performance(text: str, where: str, report: Report) -> None:
    chunk = section_text(text, "## Performance")
    if chunk is None:
        return
    for row in table_rows(chunk)[1:]:
        if len(row) < 2:
            continue
        precision = row[1].strip("`* ")
        if precision and not set(precision) <= set("—- "):
            check_precision(precision, where, "`## Performance` (column 2 is the precision)",
                            report)


def check_platform_details(text: str, where: str, listed: list[dict],
                           report: Report) -> list[dict]:
    chunk = section_text(text, "## Platform details")
    if chunk is None:
        return listed
    blocks = headings(chunk, 3)
    named = [h.lstrip("# ").strip() for _, h in blocks]
    if not blocks:
        report.error(where, "`## Platform details` has no `###` platform blocks")

    for entry in listed:
        if entry["platform"] not in named:
            report.error(where, f"`{entry['platform']}` is in the Platforms table but has no "
                                f"`### {entry['platform']}` block")
    for name in named:
        if name not in [e["platform"] for e in listed]:
            report.error(where, f"`### {name}` has no row in the `## Platforms` table")

    for name in named:
        block = section_text(chunk, f"### {name}", level=3)
        if block is None:
            continue
        cursor = -1
        lines = block.splitlines()
        for label in REQUIRED_BLOCKS:
            at = next((i for i, line in enumerate(lines)
                       if i > cursor and line.strip() == label), None)
            if at is None:
                if any(line.strip() == label for line in lines):
                    report.error(where, f"`### {name}`: `{label}` is out of order")
                else:
                    report.error(where, f"`### {name}`: missing `{label}`")
            else:
                cursor = at
        for row in table_rows(block, after="**Supported precisions**")[1:]:
            if len(row) < 2:
                continue
            precision, status = row[0].strip("`* "), row[1].strip()
            if precision:
                check_precision(precision, where, f"`### {name}` supported precisions", report)
            if status and status not in STATUSES:
                report.error(where, f"`### {name}` supported precisions: status `{status}` "
                                    f"not one of {sorted(STATUSES)}")
    return listed


# --------------------------------------------------------------------------------------
# Discovery and index
# --------------------------------------------------------------------------------------
def is_scaffolding(path: Path) -> bool:
    return path.name == "README.md" or path.parent.name == "_template"


def vendor_pages() -> list[Path]:
    return sorted(p for p in PLATFORM_DIR.glob("*.md") if not is_scaffolding(p))


def resolve(name: str) -> Path:
    candidate = Path(name)
    if candidate.is_file():
        return candidate.resolve()
    for guess in (PLATFORM_DIR / name, PLATFORM_DIR / f"{name}.md"):
        if guess.is_file():
            return guess
    raise SystemExit(f"No such vendor page: {name}")


def build_index(pages: list[dict]) -> str:
    rows = [(p["vendor"], p["file"], entry) for p in pages for entry in p["platforms"]]
    if not rows:
        return "_No partner contributions merged yet._"
    lines = ["| Vendor | Platform | Class | Accelerator | Memory | Precisions verified | Page |",
             "|---|---|---|---|---|---|---|"]
    for vendor, file, entry in sorted(rows, key=lambda r: (r[0], r[2]["platform"])):
        anchor = f"{file}#{slugify(entry['platform'])}"
        lines.append(
            f"| {vendor} | {entry['platform']} | {entry['class']} | {entry['accelerator']} "
            f"| {entry['memory']} | {entry['precisions']} | [`{file}`]({anchor}) |"
        )
    return "\n".join(lines)


def write_index(table: str) -> None:
    readme = PLATFORM_DIR / "README.md"
    text = readme.read_text(encoding="utf-8")
    start, end = text.find(INDEX_BEGIN), text.find(INDEX_END)
    if start == -1 or end == -1:
        raise SystemExit(f"{rel(readme)}: index markers not found.")
    marker_end = text.find("\n", start)
    readme.write_text(f"{text[:start]}{text[start:marker_end + 1]}\n{table}\n\n{text[end:]}",
                      encoding="utf-8")
    print(f"Wrote index into {rel(readme)}.")


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Check partner vendor pages in platform/.")
    ap.add_argument("pages", nargs="*", help="Vendor pages to check. Default: all of them.")
    ap.add_argument("--index", action="store_true", help="Print the index table and exit.")
    ap.add_argument("--write-index", action="store_true",
                    help="Splice the index table into platform/README.md.")
    args = ap.parse_args()

    paths = [resolve(name) for name in args.pages] if args.pages else vendor_pages()

    report = Report()
    pages = [p for p in (check_vendor_page(path, report) for path in paths) if p]

    if args.index or args.write_index:
        table = build_index(pages)
        write_index(table) if args.write_index else print(table)
        return

    if not paths:
        print("No vendor pages yet. Partners: copy platform/_template/VENDOR_TEMPLATE.md to "
              "platform/<your-org>.md and see platform/README.md.")
        return

    report.print()
    print(f"\nChecked {len(paths)} page(s) [{', '.join(p.name for p in paths)}]: "
          f"{report.errors} error(s), {report.warnings} warning(s).")
    sys.exit(1 if report.errors else 0)


if __name__ == "__main__":
    main()
