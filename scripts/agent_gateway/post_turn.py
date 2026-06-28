"""End-of-turn automation for the shared agent gateway.

Backends can ask for hidden post-turn work with a final line like:

    POSTTURN: {"commit":{"message":"...","paths":["scripts/x.py"]},"push":false,"merge":false}

The transport strips that marker before Telegram sees the final answer. This
keeps the public command set small while preserving explicit path-scoped git
operations. Push and merge are policy-gated by environment variables.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import ROOT


_POSTTURN_RE = re.compile(r"^\s*POSTTURN:\s*(?P<payload>\{.*\})\s*$", re.MULTILINE)
DEFAULT_MERGE_QUEUE_DIR = ROOT / "state" / "telegram-agent" / "merge-requests"


@dataclass(frozen=True)
class CommitRequest:
    message: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class PostTurnRequest:
    commit: CommitRequest | None = None
    push: bool = False
    merge: bool = False
    target_branch: str = "dev"


@dataclass(frozen=True)
class PostTurnPolicy:
    auto_commit: bool = True
    allow_push: bool = False
    allow_merge: bool = False
    remote: str = "origin"
    merge_queue_dir: Path = DEFAULT_MERGE_QUEUE_DIR

    @classmethod
    def from_env(cls) -> "PostTurnPolicy":
        return cls(
            auto_commit=_env_bool("AGENT_GATEWAY_AUTO_COMMIT", True),
            allow_push=_env_bool("AGENT_GATEWAY_ALLOW_PUSH", False),
            allow_merge=_env_bool("AGENT_GATEWAY_ALLOW_MERGE", False),
            remote=os.environ.get("AGENT_GATEWAY_GIT_REMOTE", "origin").strip() or "origin",
            merge_queue_dir=Path(
                os.environ.get(
                    "AGENT_GATEWAY_MERGE_QUEUE_DIR",
                    str(DEFAULT_MERGE_QUEUE_DIR),
                )
            ),
        )


@dataclass(frozen=True)
class PostTurnResult:
    lines: tuple[str, ...]

    def render(self) -> str:
        return "\n".join(self.lines)


def extract_post_turn_request(text: str) -> tuple[str, PostTurnRequest | None]:
    """Strip a POSTTURN marker and parse its JSON payload."""
    matches = list(_POSTTURN_RE.finditer(text))
    if not matches:
        return text.strip(), None
    payload = matches[-1].group("payload")
    cleaned = _POSTTURN_RE.sub("", text).strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return cleaned, None
    if not isinstance(data, dict):
        return cleaned, None
    return cleaned, _request_from_payload(data)


class PostTurnRunner:
    def __init__(self, *, cwd: Path = ROOT, policy: PostTurnPolicy | None = None) -> None:
        self.cwd = cwd
        self.policy = policy or PostTurnPolicy.from_env()

    def run(self, request: PostTurnRequest) -> PostTurnResult:
        lines: list[str] = []
        committed = False
        branch = self._git(["branch", "--show-current"]).strip() or "HEAD"
        if request.commit:
            line, committed = self._commit(request.commit)
            lines.append(line)
        if request.push:
            lines.append(self._push() if self.policy.allow_push else "push skipped: disabled by gateway policy")
        if request.merge:
            if self.policy.allow_merge:
                lines.append(self._merge(request.target_branch, branch))
            else:
                lines.append(self._queue_merge_request(request.target_branch, branch, committed))
        return PostTurnResult(tuple(line for line in lines if line))

    def _commit(self, request: CommitRequest) -> tuple[str, bool]:
        if not self.policy.auto_commit:
            return "commit skipped: disabled by gateway policy", False
        try:
            paths = self._safe_paths(request.paths)
        except ValueError as exc:
            return f"commit skipped: {exc}", False
        if not paths:
            return "commit skipped: no paths supplied", False
        status = self._git(["status", "--porcelain", "--", *paths]).strip()
        if not status:
            return "commit skipped: no scoped changes", False
        self._git(["add", "--", *paths])
        staged = self._run(["git", "diff", "--cached", "--quiet", "--", *paths], check=False)
        if staged.returncode == 0:
            return "commit skipped: no staged scoped changes", False
        message = request.message.strip() or "Agent gateway checkpoint"
        self._git(["commit", "-m", message, "--", *paths])
        sha = self._git(["rev-parse", "--short", "HEAD"]).strip()
        return f"committed {sha}: {message}", True

    def _push(self) -> str:
        remote = self.policy.remote
        proc = self._run(["git", "push", "-u", remote, "HEAD"], check=False)
        if proc.returncode == 0:
            return f"pushed HEAD to {remote}"
        return f"push failed: {_tail(proc.stdout)}"

    def _merge(self, target_branch: str, source_branch: str) -> str:
        if self._git(["status", "--porcelain"]).strip():
            return "merge skipped: checkout is dirty"
        current = source_branch
        target = _safe_branch(target_branch)
        try:
            self._git(["checkout", target])
            self._git(["merge", "--ff-only", current])
            sha = self._git(["rev-parse", "--short", "HEAD"]).strip()
            return f"merged {current} into {target} at {sha}"
        finally:
            self._run(["git", "checkout", current], check=False)

    def _queue_merge_request(self, target_branch: str, source_branch: str, committed: bool) -> str:
        target = _safe_branch(target_branch)
        self.policy.merge_queue_dir.mkdir(parents=True, exist_ok=True)
        sha = self._git(["rev-parse", "--short", "HEAD"]).strip()
        request_id = f"{int(time.time())}-{source_branch.replace('/', '-')}-{sha}"
        payload = {
            "id": request_id,
            "source_branch": source_branch,
            "target_branch": target,
            "head": sha,
            "repo": str(self.cwd.resolve()),
            "committed_this_turn": committed,
            "created_ts": int(time.time()),
            "updated_ts": int(time.time()),
            "status": "pending",
        }
        path = self.policy.merge_queue_dir / f"{request_id}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return f"merge queued: {source_branch} -> {target} ({sha})"

    def _safe_paths(self, paths: tuple[str, ...]) -> list[str]:
        safe: list[str] = []
        for raw in paths:
            path = raw.strip()
            if not path:
                continue
            p = Path(path)
            if p.is_absolute() or ".." in p.parts or ".git" in p.parts:
                raise ValueError(f"unsafe path {path!r}")
            resolved = (self.cwd / p).resolve()
            try:
                resolved.relative_to(self.cwd.resolve())
            except ValueError as exc:
                raise ValueError(f"path escapes repo {path!r}") from exc
            safe.append(path)
        return safe

    def _git(self, args: list[str]) -> str:
        proc = self._run(["git", *args], check=True)
        return proc.stdout

    def _run(self, cmd: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            cmd,
            cwd=str(self.cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if check and proc.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed: {_tail(proc.stdout)}")
        return proc


def _request_from_payload(data: dict[str, Any]) -> PostTurnRequest:
    commit = _commit_from_payload(data.get("commit"), data)
    push = bool(data.get("push", False))
    merge_value = data.get("merge", False)
    merge = bool(merge_value)
    target = str(data.get("target_branch") or data.get("target") or "dev")
    if isinstance(merge_value, str) and merge_value.strip():
        target = merge_value
    return PostTurnRequest(commit=commit, push=push, merge=merge, target_branch=_safe_branch(target))


def _commit_from_payload(value: Any, data: dict[str, Any]) -> CommitRequest | None:
    if value is False or value is None:
        return None
    if isinstance(value, dict):
        message = str(value.get("message") or data.get("message") or "Agent gateway checkpoint")
        raw_paths = value.get("paths") or value.get("files") or data.get("paths") or data.get("files") or []
    else:
        message = str(data.get("message") or "Agent gateway checkpoint")
        raw_paths = data.get("paths") or data.get("files") or []
    if isinstance(raw_paths, str):
        paths = (raw_paths,)
    elif isinstance(raw_paths, list):
        paths = tuple(str(item) for item in raw_paths)
    else:
        paths = ()
    return CommitRequest(message=message, paths=paths)


def _safe_branch(name: str) -> str:
    cleaned = name.strip()
    if not cleaned or cleaned.startswith("-") or ".." in cleaned or any(ch.isspace() for ch in cleaned):
        return "dev"
    return cleaned


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _tail(text: str) -> str:
    return " ".join(text.strip().splitlines())[-500:] or "no output"
