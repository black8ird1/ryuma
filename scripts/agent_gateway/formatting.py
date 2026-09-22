"""Message chunking + Markdown -> Telegram-HTML helpers.

Self-contained (stdlib only) so the bot has no external formatting dependency.
Telegram renders raw markdown literally, so model output must be converted to its
supported HTML subset (b/i/s/code/pre/a) and sent with parse_mode=HTML.
"""

from __future__ import annotations

import html
import re

TELEGRAM_LIMIT = 4096
CHUNK_SIZE = 3900  # headroom so a chunk + any marker never crosses the hard limit


def _split_plain(text: str, limit: int) -> list[str]:
    """Split on natural boundaries (paragraph > line > word > hard cut)."""
    text = text.rstrip()
    if len(text) <= limit:
        return [text] if text else [""]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        cut = remaining.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = remaining.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    return chunks


def _split_fence(block: str, limit: int) -> list[str]:
    """Split one ```fenced block that is itself over the limit, re-opening the
    fence on every piece so each renders as its own copyable <pre>."""
    lines = block.split("\n")
    opener = lines[0] if lines[0].startswith("```") else "```"
    body = lines[1:]
    if body and body[-1].strip() == "```":
        body = body[:-1]
    out, cur = [], []
    # -4 for the closing "\n```" we always append.
    budget = limit - len(opener) - 4
    size = 0
    for line in body:
        if cur and size + len(line) + 1 > budget:
            out.append(opener + "\n" + "\n".join(cur) + "\n```")
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        out.append(opener + "\n" + "\n".join(cur) + "\n```")
    return out or [block]


def split_message(text: str, limit: int = CHUNK_SIZE) -> list[str]:
    """Split on natural boundaries, NEVER inside a ``` fenced block.

    Why the fence rule (founder, 2026-08-06, third time of asking): the YT
    bundle we hand over after every pack — title, description, tags — is
    delivered in fenced blocks so Telegram renders each as a tap-to-copy
    <pre>. Splitting happens on the RAW markdown before md_to_html() sees it
    (telegram.py:179), so a cut inside a fence left one chunk with an
    unterminated ``` and the next with no opener — the _FENCE_RE match fails on
    both and NEITHER renders as a copy block. The founder was pasting two
    Telegram messages into Notes and reassembling the description by hand.

    Fenced blocks are therefore atomic: they move to the next chunk whole. A
    block that is over the limit on its own is split fence-aware, re-opening
    the fence on each piece, so the worst case is still N copyable blocks and
    never a broken one.
    """
    text = text.rstrip()
    if len(text) <= limit:
        return [text] if text else [""]

    # Tokenise into alternating plain / fenced segments. An unterminated fence
    # at EOF is treated as running to the end, which is what the renderer does.
    segments: list[tuple[str, str]] = []
    pos = 0
    for m in re.finditer(r"```[\w+-]*\n.*?(?:\n```|\Z)", text, re.DOTALL):
        if m.start() > pos:
            segments.append(("plain", text[pos:m.start()]))
        segments.append(("fence", m.group(0)))
        pos = m.end()
    if pos < len(text):
        segments.append(("plain", text[pos:]))
    if not any(kind == "fence" for kind, _ in segments):
        return _split_plain(text, limit)

    chunks: list[str] = []
    cur = ""

    def flush() -> None:
        nonlocal cur
        if cur.strip():
            chunks.append(cur.rstrip())
        cur = ""

    for kind, seg in segments:
        if kind == "fence":
            if len(seg) > limit:
                flush()
                chunks.extend(_split_fence(seg, limit))
                continue
            if len(cur) + len(seg) + 1 > limit:
                flush()
            cur = (cur + "\n" + seg) if cur else seg
            continue
        for piece in _split_plain(seg, limit):
            if not piece:
                continue
            if len(cur) + len(piece) + 2 > limit:
                flush()
            cur = (cur + "\n\n" + piece) if cur else piece
    flush()
    return chunks or [""]


_FENCE_RE = re.compile(r"```(?:[\w+-]*)\n?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")  # non-greedy so **bold with *italic* inside** works
_ITALIC_RE = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])")
_STRIKE_RE = re.compile(r"~~([^~]+)~~")
_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_HEADER_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_BARE_URL_RE = re.compile(r"https?://[^\s<>`\[\]]+")
_URL_ONLY_RE = re.compile(r"https?://\S+\Z")
_TRAILING = ".,;:!?)]}'\""


def _anchor(url: str) -> str:
    return f'<a href="{html.escape(url, quote=True)}">{html.escape(url)}</a>'


def md_to_html(text: str) -> str:
    """Convert a self-contained Markdown chunk to Telegram-safe HTML. Code spans,
    links and URLs are stashed before the global escape so they are never mangled,
    then restored. Only Telegram's supported tag subset is emitted."""
    placeholders: list[str] = []

    def _stash(inner_html: str) -> str:
        placeholders.append(inner_html)
        return f"\x00{len(placeholders) - 1}\x00"

    text = _FENCE_RE.sub(lambda m: _stash(f"<pre>{html.escape(m.group(1).rstrip())}</pre>"), text)

    def _inline(m: re.Match) -> str:
        inner = m.group(1)
        stripped = inner.strip()
        if _URL_ONLY_RE.match(stripped):
            return _stash(_anchor(stripped))
        return _stash(f"<code>{html.escape(inner)}</code>")

    text = _INLINE_CODE_RE.sub(_inline, text)
    text = _LINK_RE.sub(
        lambda m: _stash(f'<a href="{html.escape(m.group(2), quote=True)}">{html.escape(m.group(1))}</a>'),
        text,
    )

    def _bare(m: re.Match) -> str:
        url = m.group(0)
        trail = ""
        while url and url[-1] in _TRAILING:
            trail = url[-1] + trail
            url = url[:-1]
        if not url:
            return m.group(0)
        return _stash(_anchor(url)) + trail

    text = _BARE_URL_RE.sub(_bare, text)

    text = html.escape(text)
    text = _HEADER_RE.sub(lambda m: f"<b>{m.group(1)}</b>", text)
    text = _BOLD_RE.sub(lambda m: f"<b>{m.group(1)}</b>", text)
    text = _STRIKE_RE.sub(lambda m: f"<s>{m.group(1)}</s>", text)
    text = _ITALIC_RE.sub(lambda m: f"<i>{m.group(1)}</i>", text)

    def _restore(m: re.Match) -> str:
        return placeholders[int(m.group(1))]

    return re.sub(r"\x00(\d+)\x00", _restore, text)
