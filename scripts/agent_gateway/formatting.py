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


def split_message(text: str, limit: int = CHUNK_SIZE) -> list[str]:
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
