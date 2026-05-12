#!/usr/bin/env python3
"""Build topic pages from sources + a shared template.

Usage:
    python3 build.py             # build everything
    python3 build.py NAME        # build only topics/_src/NAME.html
    python3 build.py --watch     # rebuild whenever a source or the template changes

Sources live in topics/_src/*.html, written as three sections separated by
HTML comment markers:

    <!-- TOPIC META -->
    key: value
    key: value
    ...

    <!-- TOPIC CONTENT -->
    <section class="topic-level" data-level="basic">...</section>
    <section class="topic-level active" data-level="standard">...</section>
    <section class="topic-level" data-level="advanced">...</section>

    <!-- TOPIC SIDEBAR -->
    <div class="learn-more">...</div>

Meta fields:
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


def render(template: str, meta: dict[str, str], content: str, sidebar: str) -> str:
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

    # Sanity: complain if any unfilled {{placeholder}} slipped through
    leftover = re.findall(r"\{\{[a-zA-Z_]+\}\}", out)
    if leftover:
        raise ValueError(f"Unfilled placeholders in template output: {set(leftover)}")
    return out


def build_one(src: Path, template: str) -> Path:
    meta, content, sidebar = parse_source(src.read_text())
    rendered = render(template, meta, content, sidebar)
    out = OUT_DIR / src.name
    out.write_text(rendered)
    return out


def build_all(template: str) -> int:
    """Build every source file; return number of errors."""
    sources = sorted(SRC_DIR.glob("*.html"))
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
        for f in SRC_DIR.glob("*.html"):
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
        # Build a single named topic
        name = sys.argv[1].removesuffix(".html")
        src = SRC_DIR / f"{name}.html"
        if not src.exists():
            print(f"error: source not found: {src}", file=sys.stderr)
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
