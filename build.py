#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["markdown>=3.5"]
# ///
"""Build topic pages from sources + a shared template.

Usage:
    uv run build.py              # build everything
    uv run build.py NAME         # build only topics/_src/NAME.{md,html}
    uv run build.py --watch      # rebuild whenever a source or the template changes

(Plain `python3 build.py` also works if `markdown` is importable.)

Sources live in topics/_src/, either as legacy `.html` or as new `.md`.

Legacy .html format (three HTML-comment-marked sections):

    <!-- TOPIC META -->
    key: value
    key: value

    <!-- TOPIC CONTENT -->
    <section class="topic-level" ...>...</section>

    <!-- TOPIC SIDEBAR -->
    <div class="learn-more">...</div>

New .md format (YAML-ish frontmatter, then markdown body, sidebar split by
the same HTML comment marker — markdown is processed inside any HTML block
that has the `markdown="1"` attribute):

    ---
    title: ...
    eyebrow_text: ...
    eyebrow_href: {{root}}theory.html
    heading: ...
    lead: ...
    ---

    <section class="topic-level active" data-level="intuition">

    <div class="key-idea" markdown="1">
    <span class="key-idea-label">Key idea</span>

    Markdown prose here, with **bold**, *italics*, [links](...), and `code`.
    </div>

    </section>

    <!-- TOPIC SIDEBAR -->

    <div class="learn-more" markdown="1">
    ...
    </div>

Meta fields (required in both formats):
    title         - <title> tag content
    eyebrow_text  - breadcrumb text shown in the eyebrow pill
    eyebrow_href  - link target for the eyebrow
    heading       - <h1> on the page
    lead          - one-sentence summary under the heading
    prev_href     - (optional) filename for the "previous" nav link
    prev_title    - (optional) display text for the previous nav link
    next_href     - (optional) filename for the "next" nav link
    next_title    - (optional) display text for the next nav link
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import markdown as _md_lib
except ImportError:
    _md_lib = None

ROOT     = Path(__file__).resolve().parent
SRC_DIR  = ROOT / "topics" / "_src"
TEMPLATE = ROOT / "templates" / "topic.html"
OUT_DIR  = ROOT / "topics"

REQUIRED_META = ("title", "eyebrow_text", "eyebrow_href", "heading", "lead")


def parse_source(text: str) -> tuple[dict[str, str], str, str]:
    """Split a source file into (meta dict, content html, sidebar html)."""
    parts = re.split(r"<!--\s*TOPIC\s+(META|CONTENT|SIDEBAR)\s*-->", text)
    # parts: ['', 'META', meta_body, 'CONTENT', content_body, 'SIDEBAR', sidebar_body]
    if len(parts) < 7:
        raise ValueError(
            "Source must contain <!-- TOPIC META -->, <!-- TOPIC CONTENT -->, "
            "and <!-- TOPIC SIDEBAR --> markers, in that order."
        )

    sections: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        sections[parts[i].lower()] = parts[i + 1]

    meta: dict[str, str] = {}
    for line in sections["meta"].splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Bad meta line (expected 'key: value'): {line!r}")
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()

    for k in REQUIRED_META:
        if k not in meta:
            raise ValueError(f"Missing required meta field: {k!r}")

    return meta, sections["content"].strip(), sections["sidebar"].strip()


def _make_markdown():
    """Build a markdown.Markdown instance configured for our authoring needs."""
    if _md_lib is None:
        raise RuntimeError(
            "markdown library is not installed. "
            "Run via `uv run build.py` (auto-installs) or `pip install markdown`."
        )
    return _md_lib.Markdown(
        extensions=[
            "md_in_html",   # process markdown inside <div markdown="1">…</div>
            "fenced_code",  # ```…``` code blocks
            "attr_list",    # {#id .class} attribute syntax
            "tables",       # | a | b | tables
        ],
        # Keep our existing $math$ delimiters intact — markdown shouldn't
        # touch them. The smarty extension is off for the same reason
        # (would mangle quotes inside KaTeX).
    )


def parse_markdown_source(text: str) -> tuple[dict[str, str], str, str]:
    """Split a `.md` source file into (meta dict, content html, sidebar html).

    Format:
        ---
        key: value
        ...
        ---

        [markdown + HTML body]

        <!-- TOPIC SIDEBAR -->

        [markdown + HTML sidebar]
    """
    m = re.match(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", text, flags=re.DOTALL)
    if not m:
        raise ValueError(
            "Markdown source must start with YAML-style frontmatter "
            "between `---` lines."
        )
    meta_body, body = m.group(1), m.group(2)

    meta: dict[str, str] = {}
    for line in meta_body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Bad meta line (expected 'key: value'): {line!r}")
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()

    for k in REQUIRED_META:
        if k not in meta:
            raise ValueError(f"Missing required meta field: {k!r}")

    # Split body on the sidebar marker. Sidebar is optional.
    parts = re.split(r"<!--\s*TOPIC\s+SIDEBAR\s*-->", body, maxsplit=1)
    content_md = parts[0].strip()
    sidebar_md = parts[1].strip() if len(parts) == 2 else ""

    md = _make_markdown()
    content_html = _convert_md_preserving_math(md, content_md)
    md.reset()
    sidebar_html = _convert_md_preserving_math(md, sidebar_md) if sidebar_md else ""

    return meta, content_html, sidebar_html


# Python-Markdown's inline parser treats `\` + ASCII punctuation as an
# escape and drops the backslash, which destroys KaTeX commands like
# `\!`, `\{`, `\,`, `\(`, `\\`. Stash math blocks before conversion and
# restore them verbatim afterwards.
_MATH_PLACEHOLDER = "\x00KATEX{idx}\x00"
_MATH_RE = re.compile(r"\$\$.*?\$\$|\$(?!\s)[^$\n]+?(?<!\s)\$", re.DOTALL)

# HTML void elements have no close tag.
_HTML_VOID = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "source", "track", "wbr",
}
_HTML_OPEN_RE = re.compile(r"^<(\w+)\b")


def _dedent_html_blocks(text: str) -> str:
    """Strip leading whitespace from the inside of column-0 HTML blocks.

    Python-Markdown's `md_in_html` extension misparses indented HTML lines
    (4+ leading spaces) as code blocks. This preprocessor finds top-level
    HTML elements that start at column 0, reads forward to the matching
    close tag at column 0, and dedents every line in between — so authors
    can indent their HTML for readability in `.md` sources.

    Fenced code blocks pass through unchanged so embedded HTML in code
    examples is preserved.

    Limitations:
      * The opening tag and closing tag must each be on their own line at
        column 0. Single-line blocks (`<tag>…</tag>` on one line) and
        self-closing tags are passed through untouched.
      * HTML void elements (`<br>`, `<hr>`, `<img>`, etc.) are passed
        through as-is.
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    in_fenced = False

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # Fenced code blocks: pass through (and toggle state on the fence line).
        if stripped.startswith("```"):
            in_fenced = not in_fenced
            out.append(line)
            i += 1
            continue
        if in_fenced:
            out.append(line)
            i += 1
            continue

        m = _HTML_OPEN_RE.match(line)
        if not m:
            out.append(line)
            i += 1
            continue

        tag = m.group(1)

        # Void / self-closing / single-line cases — nothing to dedent.
        if (tag.lower() in _HTML_VOID
                or re.match(rf"^<{re.escape(tag)}\b[^>]*/>", line)
                or re.search(rf"</{re.escape(tag)}>", line)):
            out.append(line)
            i += 1
            continue

        # Multi-line block: read until matching close at column 0.
        close_re = re.compile(rf"^</{re.escape(tag)}>\s*$")
        block = [line]
        i += 1
        found = False
        while i < len(lines):
            l = lines[i]
            block.append(l)
            i += 1
            if close_re.match(l):
                found = True
                break

        if found:
            # Dedent every line; preserve blank lines as-is. Lines inside a
            # nested fenced code block (```) pass through verbatim so we
            # don't destroy the Python/etc. indentation the code relies on.
            in_block_fence = False
            for bl in block:
                if bl.lstrip().startswith("```"):
                    in_block_fence = not in_block_fence
                    out.append(bl.lstrip() if bl.strip() else bl)
                    continue
                if in_block_fence:
                    out.append(bl)            # preserve code indentation
                else:
                    out.append(bl.lstrip() if bl.strip() else bl)
        else:
            # No matching close — give up and pass through verbatim.
            out.extend(block)

    return "\n".join(out)


def _convert_md_preserving_math(md, text: str) -> str:
    # Dedent first so indented HTML inside the source doesn't trip md_in_html
    text = _dedent_html_blocks(text)

    blocks: list[str] = []

    def stash(m: re.Match) -> str:
        blocks.append(m.group(0))
        return _MATH_PLACEHOLDER.format(idx=len(blocks) - 1)

    stashed = _MATH_RE.sub(stash, text)
    html = md.convert(stashed)

    def restore(m: re.Match) -> str:
        return blocks[int(m.group(1))]

    return re.sub(r"\x00KATEX(\d+)\x00", restore, html)


def render_nav(meta: dict[str, str], side: str) -> str:
    """Build the prev/next anchor for the topic-nav footer."""
    href  = meta.get(f"{side}_href", "")
    title = meta.get(f"{side}_title", "")
    if not href or not title:
        return ""
    label = "Previous" if side == "prev" else "Next"
    return (
        f'<a href="{href}" class="{side}">\n'
        f'                        <span>\n'
        f'                            <span class="topic-nav-label">{label}</span>\n'
        f'                            <span class="topic-nav-title">{title}</span>\n'
        f'                        </span>\n'
        f'                    </a>'
    )


def render(template: str, meta: dict[str, str], content: str, sidebar: str, root: str) -> str:
    """Render the template + content. `root` is the relative path back to the
    site root (e.g. "../" for topics/foo.html, "../../" for topics/foo/bar.html)."""
    substitutions = {
        "{{title}}":        meta["title"],
        "{{eyebrow_text}}": meta["eyebrow_text"],
        "{{eyebrow_href}}": meta["eyebrow_href"],
        "{{heading}}":      meta["heading"],
        "{{lead}}":         meta["lead"],
        "{{content}}":      content,
        "{{sidebar}}":      sidebar,
        "{{prev_link}}":    render_nav(meta, "prev"),
        "{{next_link}}":    render_nav(meta, "next"),
    }
    out = template
    for needle, value in substitutions.items():
        out = out.replace(needle, value)

    # Set the active navbar item. Source declares `active_nav: theory|practice|resources`
    # (default theory). One of the three placeholders becomes "active", the others "".
    active = meta.get("active_nav", "theory")
    for section in ("theory", "engineering", "resources"):
        out = out.replace(f"{{{{{section}_active}}}}",
                          "active" if section == active else "")

    # Substitute {{root}} last so it works in both template and content.
    out = out.replace("{{root}}", root)

    # Sanity: complain if any unfilled {{placeholder}} slipped through
    leftover = re.findall(r"\{\{[a-zA-Z_]+\}\}", out)
    if leftover:
        raise ValueError(f"Unfilled placeholders in template output: {set(leftover)}")
    return out


def build_one(src: Path, template: str) -> Path:
    if src.suffix == ".md":
        meta, content, sidebar = parse_markdown_source(src.read_text())
    else:
        meta, content, sidebar = parse_source(src.read_text())
    rel = src.relative_to(SRC_DIR).with_suffix(".html")
    # "../" per directory level back to the site root
    root = "../" * len(rel.parts)
    rendered = render(template, meta, content, sidebar, root)
    out = OUT_DIR / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered)
    return out


def _all_sources() -> list[Path]:
    """All source files, sorted, both .html and .md."""
    return sorted(list(SRC_DIR.rglob("*.html")) + list(SRC_DIR.rglob("*.md")))


def build_all(template: str) -> int:
    """Build every source file (recursively); return number of errors."""
    sources = _all_sources()
    if not sources:
        print(f"error: no sources in {SRC_DIR}", file=sys.stderr)
        return 1

    errors = 0
    for src in sources:
        try:
            out = build_one(src, template)
            print(f"✓ built {out.relative_to(ROOT)}")
        except Exception as e:
            errors += 1
            print(f"✗ {src.relative_to(ROOT)}: {e}", file=sys.stderr)
    return errors


def watch_loop(template_path: Path) -> int:
    """Rebuild whenever sources or the template change. Polls every 0.4s."""
    import time

    def snapshot() -> dict[Path, float]:
        m: dict[Path, float] = {}
        for f in _all_sources():
            m[f] = f.stat().st_mtime
        if template_path.exists():
            m[template_path] = template_path.stat().st_mtime
        return m

    print(f"[watch] watching {SRC_DIR.relative_to(ROOT)}/ and {template_path.relative_to(ROOT)}")
    print(f"[watch] press Ctrl+C to stop")

    # Initial build
    template = template_path.read_text()
    build_all(template)
    mtimes = snapshot()

    try:
        while True:
            time.sleep(0.4)
            current = snapshot()
            changed = [p for p, m in current.items() if mtimes.get(p) != m]
            if changed:
                names = ", ".join(p.name for p in changed)
                print(f"\n[change] {names}")
                template = template_path.read_text()
                build_all(template)
                mtimes = current
    except KeyboardInterrupt:
        print("\n[watch] stopped")
        return 0


def main() -> int:
    if not TEMPLATE.exists():
        print(f"error: template not found at {TEMPLATE}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        return watch_loop(TEMPLATE)

    template = TEMPLATE.read_text()

    if len(sys.argv) > 1:
        # Build a single named topic (supports nested: "neural-networks/cnn").
        # Prefer .md if both exist (legacy .html source can be left behind
        # until it's deleted manually).
        name = sys.argv[1].removesuffix(".html").removesuffix(".md")
        src_md   = SRC_DIR / f"{name}.md"
        src_html = SRC_DIR / f"{name}.html"
        if src_md.exists():
            src = src_md
        elif src_html.exists():
            src = src_html
        else:
            print(f"error: source not found: {src_md} or {src_html}", file=sys.stderr)
            return 1
        try:
            out = build_one(src, template)
            print(f"✓ built {out.relative_to(ROOT)}")
            return 0
        except Exception as e:
            print(f"✗ {src.relative_to(ROOT)}: {e}", file=sys.stderr)
            return 1

    return 0 if build_all(template) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
