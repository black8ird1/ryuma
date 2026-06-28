"""Automatic task worktree broker for write-capable gateway turns."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .core import AgentTurn, ROOT


@dataclass(frozen=True)
class WorktreePolicy:
    enabled: bool = True
    repo: Path = ROOT
    root: Path = Path(os.environ.get("AGENT_GATEWAY_WORKTREE_ROOT", str(Path.home() / ".ninja-bot" / "worktrees")))
    base: str = "HEAD"
    branch_prefix: str = "agent"

    @classmethod
    def from_env(cls) -> "WorktreePolicy":
        # One knob: AGENT_GATEWAY_BASE_BRANCH is both the fork point AND the merge
        # target (see merge_gate.base_branch_from_env), so "set it to dev" means
        # agents branch from dev and land back into dev. Falls back to the older
        # WORKTREE_BASE var, then HEAD (fork from whatever the repo is on).
        base = (
            os.environ.get("AGENT_GATEWAY_BASE_BRANCH")
            or os.environ.get("AGENT_GATEWAY_WORKTREE_BASE")
            or "HEAD"
        ).strip() or "HEAD"
        return cls(
            enabled=_env_bool("AGENT_GATEWAY_AUTO_WORKTREE", True),
            repo=Path(os.environ.get("AGENT_GATEWAY_REPO_ROOT", str(ROOT))),
            root=Path(os.environ.get("AGENT_GATEWAY_WORKTREE_ROOT", str(Path.home() / ".ninja-bot" / "worktrees"))),
            base=base,
            branch_prefix=os.environ.get("AGENT_GATEWAY_BRANCH_PREFIX", "agent").strip() or "agent",
        )


@dataclass(frozen=True)
class WorktreeAssignment:
    id: str
    backend: str
    branch: str
    path: Path
    reused: bool = False

    def status_line(self) -> str:
        action = "reusing" if self.reused else "ready"
        return f"worktree {action}: {self.branch} at {self.path}"


class WorktreeBroker:
    def __init__(self, policy: WorktreePolicy | None = None) -> None:
        self.policy = policy or WorktreePolicy.from_env()
        self._assignments: dict[tuple[int, str], WorktreeAssignment] = {}

    def prepare(self, turn: AgentTurn) -> WorktreeAssignment | None:
        if not self.policy.enabled or turn.mode != "write":
            return None
        key = (turn.chat_id, turn.backend)
        existing = self._assignments.get(key)
        if existing is not None and existing.path.exists():
            return WorktreeAssignment(
                id=existing.id,
                backend=existing.backend,
                branch=existing.branch,
                path=existing.path,
                reused=True,
            )
        assignment = self._create(turn)
        self._assignments[key] = assignment
        return assignment

    def reset(self, chat_id: int) -> None:
        for key in [key for key in self._assignments if key[0] == chat_id]:
            self._assignments.pop(key, None)

    def workdir_for(self, chat_id: int, backend: str) -> Path | None:
        assignment = self._assignments.get((chat_id, backend))
        return assignment.path if assignment else None

    def remove_branch(self, branch: str) -> None:
        """Drop the worktree DIR for a branch (e.g. after it lands). The branch and
        its commits are untouched — only the working copy is removed, so a finished
        worktree stops leaking. Forgets the in-memory assignment too."""
        for key, a in list(self._assignments.items()):
            if a.branch == branch:
                self._remove_path(a.path)
                self._assignments.pop(key, None)
                return
        # not in-memory (e.g. after a restart) — best-effort remove by branch lookup.
        for path, b in _list_worktrees(self.policy.repo):
            if b == branch:
                self._remove_path(Path(path))
                return

    def discard_branch(self, branch: str) -> str:
        """Throw away an agent worktree AND delete its branch — the operator's
        'reject this turn' escape hatch when they don't want to land the work.

        Safe by construction: it refuses any branch that isn't one of OUR agent
        branches (prefix-gated), so it can never touch the base/prod branch. The
        commits linger in the reflog for a while, so a discard is recoverable.
        Returns a short human note."""
        prefix = f"{_safe_prefix(self.policy.branch_prefix)}/"
        if not branch or not branch.startswith(prefix):
            return f"refused: {branch or '(none)'} is not an agent worktree branch"
        path: Path | None = None
        for key, a in list(self._assignments.items()):
            if a.branch == branch:
                path = a.path
                self._assignments.pop(key, None)
                break
        if path is None:
            for p, b in _list_worktrees(self.policy.repo):
                if b == branch:
                    path = Path(p)
                    break
        if path is not None:
            self._remove_path(path)
        _git(self.policy.repo, ["worktree", "prune"], check=False)
        deleted = _git(self.policy.repo, ["branch", "-D", branch], check=False).returncode == 0
        if deleted:
            return f"discarded {branch} (worktree + branch removed)"
        return f"discarded {branch} (worktree removed; branch already gone)"

    def sweep_startup(self, base: str) -> list[dict[str, object]]:
        """On bot start: prune stale git registrations, then for each of OUR agent
        worktrees — remove the ones with no unlanded commits (true orphans; branch
        preserved), and KEEP + REPORT the ones still ahead of `base` (parallel work
        that survived the restart). Returns the live ones so the operator sees them
        the moment the bot comes back — restart is when that awareness matters most."""
        if not self.policy.enabled:
            return []
        repo = self.policy.repo
        _git(repo, ["worktree", "prune"], check=False)
        root = str(self.policy.root.resolve())
        live: list[dict[str, object]] = []
        for path, branch in _list_worktrees(repo):
            try:
                under_root = str(Path(path).resolve()).startswith(root)
            except OSError:
                under_root = False
            if not under_root or not branch:
                continue  # not one of ours — leave it alone
            ahead = _count_ahead(repo, base, branch)
            if ahead > 0:
                live.append({"branch": branch, "path": path, "ahead": ahead})
            else:
                self._remove_path(Path(path))  # landed/empty → reclaim
        return live

    def _remove_path(self, path: Path) -> None:
        _git(self.policy.repo, ["worktree", "remove", "--force", str(path)], check=False)

    def _create(self, turn: AgentTurn) -> WorktreeAssignment:
        backend = _safe_component(turn.backend)
        task_id = _task_id(turn.text)
        wid = f"{backend}-{turn.chat_id}-{task_id}"
        branch = f"{_safe_prefix(self.policy.branch_prefix)}/{backend}/{turn.chat_id}-{task_id}"
        path = (self.policy.root / backend / f"{turn.chat_id}-{task_id}").resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            if _branch_exists(self.policy.repo, branch):
                _git(self.policy.repo, ["worktree", "add", str(path), branch])
            else:
                _git(self.policy.repo, ["worktree", "add", "-b", branch, str(path), self.policy.base])
        return WorktreeAssignment(id=wid, backend=backend, branch=branch, path=path)


def _task_id(text: str) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), text.strip())
    return hashlib.sha1(first_line.encode("utf-8")).hexdigest()[:12]


def _safe_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip().lower())
    return cleaned.strip("-_") or "agent"


def _safe_prefix(value: str) -> str:
    cleaned = "/".join(_safe_component(part) for part in value.split("/") if part.strip())
    return cleaned or "agent"


def _branch_exists(repo: Path, branch: str) -> bool:
    return _git(repo, ["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"], check=False).returncode == 0


def _list_worktrees(repo: Path) -> list[tuple[str, str]]:
    """Parse `git worktree list --porcelain` → [(path, branch)]. Branch is the
    short name ('' if detached)."""
    out = _git(repo, ["worktree", "list", "--porcelain"], check=False).stdout
    items: list[tuple[str, str]] = []
    path = ""
    for line in out.splitlines():
        if line.startswith("worktree "):
            path = line[len("worktree "):].strip()
        elif line.startswith("branch "):
            ref = line[len("branch "):].strip()
            items.append((path, ref.replace("refs/heads/", "", 1)))
            path = ""
        elif not line.strip() and path:
            items.append((path, ""))  # detached worktree
            path = ""
    if path:
        items.append((path, ""))
    return items


def _count_ahead(repo: Path, base: str, branch: str) -> int:
    """Commits on `branch` not yet in `base` — i.e. unlanded work. 0 if either ref
    is missing (treat as nothing to keep)."""
    if not base or not branch:
        return 0
    out = _git(repo, ["rev-list", "--count", f"{base}..{branch}"], check=False).stdout.strip()
    return int(out) if out.isdigit() else 0


def _git(repo: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stdout.strip()}")
    return proc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
