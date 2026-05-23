#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["beautifulsoup4>=4.12"]
# ///
"""Convert a legacy topic .html source into the new .md format.

Legacy format (one file, three comment-delimited sections):

    <!-- TOPIC META -->
    title: …
    eyebrow_text: …
    …

    <!-- TOPIC CONTENT -->
    <section class="topic-level …">
        <div class="key-idea">…</div>
        …
    </section>

    <!-- TOPIC SIDEBAR -->
    <div class="learn-more">
        …
    </div>

New format (a single .md file):

    ---
    title: …
    eyebrow_text: …
    …
    ---

    <section class="topic-level …" markdown="1">

    <div class="key-idea" markdown="1">
    <span class="key-idea-label">Key idea</span>

    **Predict…** *as a weighted sum of the inputs…*
    </div>

    …

    <!-- TOPIC SIDEBAR -->

    <div class="learn-more" markdown="1">
    …
    </div>

The conversion:

1. The META block becomes YAML-style frontmatter between `---` lines.
2. Containers whose **inner text** should be authored as markdown get the
   attribute `markdown="1"` (so python-markdown's `md_in_html` extension
   processes them). For each such container, the text-level HTML inside
   (<p>, <h2-4>, <ul><li>, <strong>, <em>, <code>, <a>) is converted to its
   markdown equivalent.
3. Containers that should stay as raw HTML (the viz-embed, script tags,
   level-next buttons, the controls inside formula-block, etc.) are
   preserved verbatim — except where they contain a markdown-authored
   container, which gets a recursive conversion.
4. The output is pretty-printed so the markdown text reads naturally.

The point isn't to wring every drop of HTML into markdown — it's to make the
text inside paragraphs / lists / headings editable as plain markdown while
leaving the layout machinery alone.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag


# ---------------------------------------------------------------- containers

# Outer containers whose inner text should be authored as markdown. We add
# `markdown="1"` to each and convert the inner text-level HTML.
MARKDOWN_CONTAINERS = {
    # Outer page structure
    "section": {"topic-level"},
    "article": {"tldr-body"},
    "div": {
        "key-idea",                  # the intro card (also formula-block sometimes)
        "tldr-body",
        "use-cases",
        "yes",
        "no",
        "code-snippet",
        "fig-explainer",
        "learn-more",
        "formula-block",
        "notation-standard",
        "notation-plain",
    },
    "ul": {"formula-legend"},        # legend is a list of li with <code> + text
    "li": set(),                     # only inside formula-legend (handled below)
}

# Within a markdown="1" container, only this list of children stays as raw
# HTML — everything else is converted. Containers in this set are themselves
# recursed-into so any nested markdown="1" target is still processed.
PASSTHROUGH_TAGS_INSIDE_MD = {
    "span", "i", "kbd", "sub", "sup", "br", "img", "button", "input", "select",
    "option", "label", "canvas", "script", "summary", "details",
}


# ---------------------------------------------------------------- helpers

def text_to_md_inline(node) -> str:
    """Walk a BeautifulSoup node, returning a markdown string for its inline
    content. Block-level children are NOT handled here (they're rendered by
    the block walker)."""
    out = []
    for child in node.children:
        if isinstance(child, NavigableString):
            out.append(str(child))
            continue
        name = child.name
        if name == "strong" or name == "b":
            out.append(f"**{text_to_md_inline(child)}**")
        elif name == "em" or name == "i" and not child.get("class"):
            # Plain <em>/<i> = italics. But <i class="fa…"> is a font-awesome
            # icon — preserve it as raw HTML.
            out.append(f"*{text_to_md_inline(child)}*")
        elif name == "i" and child.get("class"):
            out.append(str(child))                # icon — keep raw
        elif name == "code":
            txt = child.get_text()
            out.append(f"`{txt}`")
        elif name == "a":
            href = child.get("href", "")
            label = text_to_md_inline(child).strip()
            # Extra attributes we want to preserve (like target="_blank") get
            # tacked on via attr_list. Plain href-only links stay clean.
            extra_attrs = []
            for k, v in child.attrs.items():
                if k in ("href", "class"):
                    continue
                if isinstance(v, list):
                    v = " ".join(v)
                extra_attrs.append(f'{k}="{v}"')
            cls = child.get("class") or []
            if cls:
                extra_attrs.append(" ".join("." + c for c in cls))
            attr_suffix = f'{{: {" ".join(extra_attrs)} }}' if extra_attrs else ""
            out.append(f"[{label}]({href}){attr_suffix}")
        elif name in PASSTHROUGH_TAGS_INSIDE_MD:
            out.append(str(child))
        elif name in ("p", "h1", "h2", "h3", "h4", "h5", "ul", "ol", "li", "div", "blockquote"):
            # Block-level element nested in inline context — render block-style
            # but inline its representation. Caller is responsible for
            # handling this. We fall through to text representation here.
            out.append(text_to_md_inline(child))
        else:
            out.append(str(child))
    return "".join(out)


def render_block(node, indent: int = 0) -> str:
    """Render a block-level container's children as a markdown document.
    Each block element produces its own paragraph separated by blank lines."""
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            t = str(child)
            if t.strip():
                # Loose text in a block — treat as a paragraph
                parts.append(t.strip())
            continue
        rendered = render_one_block(child, indent)
        if rendered is not None:
            parts.append(rendered)
    return "\n\n".join(p for p in parts if p)


def render_one_block(child: Tag, indent: int) -> str | None:
    """Render a single block element. Returns the markdown chunk (without
    trailing blank line) or None if the element produced nothing."""
    name = child.name
    cls = " ".join(child.get("class") or [])

    if name == "p":
        attr = format_attr_list(child)
        body = text_to_md_inline(child).strip()
        if attr:
            return f"{body}\n{{: {attr} }}"
        return body

    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        level = int(name[1])
        hashes = "#" * level
        body = text_to_md_inline(child).strip()
        return f"{hashes} {body}"

    if name == "ul":
        lis = child.find_all("li", recursive=False)
        # If the UL has any class (e.g. "formula-legend") OR any <li> has
        # attributes beyond "class" (e.g. data-tier), emit the whole list as
        # raw HTML with markdown="1" so those attributes survive untouched.
        ul_has_attrs = bool(child.attrs)
        li_has_attrs = any(set(li.attrs) - {"class"} for li in lis)
        if ul_has_attrs or li_has_attrs:
            pieces = [build_open_tag(child, with_markdown=True)]
            for li in lis:
                pieces.append(build_open_tag(li, with_markdown=True))
                pieces.append("")
                pieces.append(render_list_item(li))
                pieces.append("")
                pieces.append("</li>")
            pieces.append("</ul>")
            return "\n".join(pieces)
        items = []
        for li in lis:
            body = render_list_item(li)
            items.append(f"- {body}")
        return "\n".join(items)

    if name == "ol":
        items = []
        for n, li in enumerate(child.find_all("li", recursive=False), start=1):
            body = render_list_item(li)
            items.append(f"{n}. {body}")
        return "\n".join(items)

    if name == "blockquote":
        inner = render_block(child).splitlines()
        return "\n".join("> " + line for line in inner)

    if name == "pre":
        # Code block — pull out the <code> contents and emit as fenced
        code = child.find("code")
        if code is None:
            return str(child)
        lang_class = " ".join(code.get("class") or [])
        m = re.search(r"language-(\S+)", lang_class)
        lang = m.group(1) if m else ""
        code_text = code.get_text()
        return f"```{lang}\n{code_text.rstrip()}\n```"

    # Otherwise — a container element. If it's a known markdown target,
    # recurse and emit as a markdown="1" block. If it's anything else,
    # serialize as raw HTML (but recurse into it so nested md targets are
    # still processed).
    if is_md_container(child):
        return render_md_container(child)
    # Recurse: serialize as raw HTML but convert any nested md targets.
    return serialize_html_with_md(child)


def render_list_item(li: Tag) -> str:
    """Render the contents of a <li>. Most list items are simple inline
    text, but some contain nested structure (e.g. formula-legend items have
    <code>…</code> followed by a description)."""
    # If the li contains block-level children, render with newlines.
    has_block = any(
        isinstance(c, Tag) and c.name in ("p", "ul", "ol", "div", "h3", "h4")
        for c in li.children
    )
    if has_block:
        return render_block(li).replace("\n", "\n  ")
    return text_to_md_inline(li).strip()


def format_attr_list(tag: Tag) -> str | None:
    """Format extra <p> attributes as a markdown attr_list."""
    cls = tag.get("class") or []
    if cls:
        classes = " ".join("." + c for c in cls)
        return classes
    return None


def is_md_container(tag: Tag) -> bool:
    """Should this tag get markdown='1' and have its inner text converted?"""
    name = tag.name
    if name not in MARKDOWN_CONTAINERS:
        return False
    classes_we_handle = MARKDOWN_CONTAINERS[name]
    if not classes_we_handle:
        # Any tag of this name qualifies (rare — currently empty set)
        return False
    tag_classes = set(tag.get("class") or [])
    return bool(classes_we_handle & tag_classes)


def render_md_container(tag: Tag) -> str:
    """Render a container as `<tag class=… markdown="1">` + markdown body
    + `</tag>` — so python-markdown processes the inside on build."""
    inner_md = render_block(tag)
    open_tag = build_open_tag(tag, with_markdown=True)
    close_tag = f"</{tag.name}>"
    return f"{open_tag}\n\n{inner_md}\n\n{close_tag}"


def build_open_tag(tag: Tag, *, with_markdown: bool) -> str:
    """Reconstruct the opening tag string, optionally adding markdown='1'."""
    parts = [tag.name]
    for k, v in tag.attrs.items():
        if k == "markdown":
            continue
        if isinstance(v, list):
            v = " ".join(v)
        # Boolean attrs render bare
        if v is True or v == "":
            parts.append(k)
        else:
            v_quoted = v.replace('"', "&quot;")
            parts.append(f'{k}="{v_quoted}"')
    if with_markdown:
        parts.append('markdown="1"')
    return "<" + " ".join(parts) + ">"


def serialize_html_with_md(tag: Tag) -> str:
    """Serialise a non-markdown tag as raw HTML, but recurse so any nested
    markdown="1" container is still processed and emitted as markdown."""
    open_tag = build_open_tag(tag, with_markdown=False)
    pieces = [open_tag]
    for child in tag.children:
        if isinstance(child, NavigableString):
            pieces.append(str(child))
        elif isinstance(child, Tag):
            if is_md_container(child):
                pieces.append("\n")
                pieces.append(render_md_container(child))
                pieces.append("\n")
            else:
                pieces.append(serialize_html_with_md(child))
    pieces.append(f"</{tag.name}>")
    return "".join(pieces)


# ---------------------------------------------------------------- entry point

META_RE = re.compile(r"<!--\s*TOPIC\s+META\s*-->")
CONTENT_RE = re.compile(r"<!--\s*TOPIC\s+CONTENT\s*-->")
SIDEBAR_RE = re.compile(r"<!--\s*TOPIC\s+SIDEBAR\s*-->")


def split_sections(text: str) -> tuple[str, str, str]:
    parts = re.split(r"<!--\s*TOPIC\s+(META|CONTENT|SIDEBAR)\s*-->", text)
    if len(parts) < 7:
        raise ValueError("source missing META/CONTENT/SIDEBAR markers")
    sections: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        sections[parts[i].lower()] = parts[i + 1]
    return sections["meta"], sections["content"], sections["sidebar"]


def meta_to_yaml(meta_body: str) -> str:
    lines = []
    for raw in meta_body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"bad meta line: {line!r}")
        key, _, value = line.partition(":")
        lines.append(f"{key.strip()}: {value.strip()}")
    return "---\n" + "\n".join(lines) + "\n---"


def html_block_to_md(html: str) -> str:
    """Take an HTML body, return markdown body."""
    # Parse as a fragment. BeautifulSoup will add <html><body> if we use the
    # html parser, so wrap in a sentinel root and walk children of that root.
    soup = BeautifulSoup(f"<root>{html}</root>", "html.parser")
    root = soup.find("root")
    return render_block(root)


def convert(text: str) -> str:
    meta_body, content_body, sidebar_body = split_sections(text)
    yaml = meta_to_yaml(meta_body)
    content_md = html_block_to_md(content_body)
    sidebar_md = html_block_to_md(sidebar_body)
    out = (
        yaml + "\n\n"
        + content_md + "\n\n"
        + "<!-- TOPIC SIDEBAR -->\n\n"
        + sidebar_md + "\n"
    )
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: html_to_md.py <file.html> [file2.html …]", file=sys.stderr)
        print("       html_to_md.py --all   # convert every legacy source", file=sys.stderr)
        return 2

    targets: list[Path]
    if sys.argv[1] == "--all":
        targets = sorted(Path("topics/_src").rglob("*.html"))
    else:
        targets = [Path(a) for a in sys.argv[1:]]

    for src in targets:
        if not src.exists():
            print(f"✗ {src}: not found", file=sys.stderr)
            continue
        if src.suffix != ".html":
            print(f"✗ {src}: not an .html source", file=sys.stderr)
            continue
        try:
            md = convert(src.read_text())
        except Exception as e:
            print(f"✗ {src}: {e}", file=sys.stderr)
            continue
        dst = src.with_suffix(".md")
        dst.write_text(md)
        src.unlink()
        print(f"✓ {src} → {dst.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
