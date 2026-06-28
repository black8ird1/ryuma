"""Safe merge gate for landing agent worktree branches into the base branch.

Two layers, on purpose:

  * assess()  — READ-ONLY, best-effort. Is a merge needed? Will it fast-forward?
                Is the base checkout clean? Are conflicts likely? This drives the
                tap-to-confirm dialog. It never mutates anything, so it is safe to
                call any time, as often as we like.

  * land()    — AUTHORITATIVE, runs only on explicit confirm (a tap, or auto-land
                when the operator opted in). It is atomic: it merges in the base
                checkout and on ANY conflict it `merge --abort`s and reports the
                conflicting files. It NEVER auto-resolves. Because of the abort,
                land() is safe even if assess() guessed wrong.

The base branch (e.g. `dev`) is only ever merged in the one checkout that owns it
(the operator repo). Agents only ever produce commits on their own branch — they
never touch the base branch or each other's worktrees. That isolation, not a lock,
is what makes parallel agents safe. The bot never goes near the production branch
(`main`); promoting `dev` -> `main` stays the operator's existing manual step.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .core import ROOT


def protected_branches_from_env() -> set[str]:
    """Branches the bot must NEVER land into — prod, basically. Branch-agnostic:
    the operator (or the project's brain/config) sets the list. Defaults to the
    universal prod conventions main/master so an UN-configured bot fails SAFE rather
    than merging to prod. To allow landing to one of these, drop it from
    AGENT_GATEWAY_PROTECTED_BRANCHES (e.g. set it to '' or 'master')."""
    raw = os.environ.get("AGENT_GATEWAY_PROTECTED_BRANCHES", "main,master")
    return {b.strip() for b in raw.split(",") if b.strip()}


def base_branch_from_env(default: str = "main") -> str:
    """The branch agents fork from and land back into. One knob per project.

    Default `main` works for any repo; a project can set it to an integration
    branch (e.g. `dev`) so landed work flows there and never touches prod.
    """
    value = (
        os.environ.get("AGENT_GATEWAY_BASE_BRANCH")
        or os.environ.get("AGENT_GATEWAY_WORKTREE_BASE")
        or default
    ).strip()
    return value or default


@dataclass(frozen=True)
class MergeAssessment:
    needed: bool
    branch: str
    base: str
    ahead: int = 0
    fast_forward: bool = False
    dirty: bool = False
    likely_conflict: bool = False
    diffstat: str = ""
    reason: str = ""

    @property
    def landable(self) -> bool:
        """True when a one-tap land is safe to offer (work exists, base clean)."""
        return self.needed and not self.dirty

    def summary(self) -> str:
        if not self.needed:
            return self.reason or f"nothing to land into {self.base}"
        kind = "fast-forward" if self.fast_forward else "merge"
        bits = [f"{self.ahead} commit(s) → {self.base} ({kind})"]
        if self.diffstat:
            bits.append(self.diffstat)
        if self.dirty:
            bits.append("⚠ base checkout dirty")
        if self.likely_conflict:
            bits.append("⚠ likely conflicts")
        return " · ".join(bits)


@dataclass(frozen=True)
class MergeResult:
    ok: bool
    branch: str
    base: str
    sha: str = ""
    conflicts: tuple[str, ...] = ()
    message: str = ""


@dataclass(frozen=True)
class _Git:
    code: int
    out: str


class MergeGate:
    def __init__(self, *, repo: Path | str = ROOT, base: str | None = None, protected: set[str] | None = None) -> None:
        self.repo = Path(repo)
        self.base = base or base_branch_from_env()
        self.protected = protected if protected is not None else protected_branches_from_env()

    # ---- read-only assessment (drives the confirm dialog) ----
    def assess(self, branch: str) -> MergeAssessment:
        base = self.base
        if base in self.protected:
            # Never offer to land into a protected branch (prod). No button, no auto-land.
            return MergeAssessment(False, branch, base, reason=f"base '{base}' is protected — the bot never lands to prod; set AGENT_GATEWAY_BASE_BRANCH to an integration branch")
        if not self._branch_exists(branch):
            return MergeAssessment(False, branch, base, reason=f"branch {branch} not found")
        if not self._branch_exists(base):
            return MergeAssessment(False, branch, base, reason=f"base {base} not found")
        ahead = self._count(f"{base}..{branch}")
        if ahead == 0:
            return MergeAssessment(False, branch, base, reason=f"{branch} has no commits ahead of {base}")
        fast_forward = self._is_ancestor(base, branch)
        dirty = bool(self._git(["status", "--porcelain"]).out.strip())
        diffstat = self._compact_diffstat(base, branch)
        likely_conflict = False if fast_forward else self._likely_conflict(base, branch)
        return MergeAssessment(
            needed=True,
            branch=branch,
            base=base,
            ahead=ahead,
            fast_forward=fast_forward,
            dirty=dirty,
            likely_conflict=likely_conflict,
            diffstat=diffstat,
        )

    # ---- authoritative land (runs only on confirm) ----
    def land(self, branch: str, *, message: str | None = None, push: bool = False, remote: str = "origin") -> MergeResult:
        base = self.base
        if base in self.protected:  # defense in depth — authoritative refusal, even if assess is bypassed
            return MergeResult(False, branch, base, message=f"refusing to land into protected branch '{base}' (prod). Set AGENT_GATEWAY_BASE_BRANCH to an integration branch, or remove it from AGENT_GATEWAY_PROTECTED_BRANCHES.")
        assessment = self.assess(branch)
        if not assessment.needed:
            return MergeResult(False, branch, base, message=assessment.reason or "nothing to land")
        if assessment.dirty:
            return MergeResult(False, branch, base, message=f"{base} checkout is dirty — commit or stash it before landing")
        if self._current_branch() != base:
            checkout = self._git(["checkout", base])
            if checkout.code != 0:
                return MergeResult(False, branch, base, message=f"could not checkout {base}: {_tail(checkout.out)}")
        if assessment.fast_forward:
            merged = self._git(["merge", "--ff-only", branch])
        else:
            msg = message or f"Land {branch} into {base}"
            merged = self._git(["merge", "--no-ff", "-m", msg, branch])
        if merged.code != 0:
            conflicts = tuple(self._git(["diff", "--name-only", "--diff-filter=U"]).out.split())
            if self._git(["rev-parse", "-q", "--verify", "MERGE_HEAD"], allow_fail=True).code == 0:
                self._git(["merge", "--abort"])
            note = "conflicts — left for you to resolve" if conflicts else f"merge failed: {_tail(merged.out)}"
            return MergeResult(False, branch, base, conflicts=conflicts, message=note)
        sha = self._git(["rev-parse", "--short", "HEAD"]).out.strip()
        result = MergeResult(True, branch, base, sha=sha, message=f"landed {branch} → {base} at {sha}")
        if push:
            pushed = self._git(["push", remote, base])
            if pushed.code != 0:
                return MergeResult(True, branch, base, sha=sha, message=f"landed at {sha}, but push failed: {_tail(pushed.out)}")
            result = MergeResult(True, branch, base, sha=sha, message=f"landed {branch} → {base} at {sha} and pushed to {remote}")
        return result

    # ---- helpers ----
    def _likely_conflict(self, base: str, branch: str) -> bool:
        merge_base = self._git(["merge-base", base, branch]).out.strip()
        if not merge_base:
            return False
        out = self._git(["merge-tree", merge_base, base, branch]).out
        return "<<<<<<<" in out or "changed in both" in out

    def _branch_exists(self, branch: str) -> bool:
        return self._git(["rev-parse", "--verify", "--quiet", branch], allow_fail=True).code == 0

    def _count(self, rng: str) -> int:
        out = self._git(["rev-list", "--count", rng]).out.strip()
        return int(out) if out.isdigit() else 0

    def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return self._git(["merge-base", "--is-ancestor", ancestor, descendant], allow_fail=True).code == 0

    def _current_branch(self) -> str:
        return self._git(["branch", "--show-current"]).out.strip()

    def _compact_diffstat(self, base: str, branch: str) -> str:
        out = self._git(["diff", "--shortstat", f"{base}...{branch}"]).out.strip()
        return " ".join(out.split())

    def _git(self, args: list[str], *, allow_fail: bool = False) -> _Git:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(self.repo),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return _Git(proc.returncode, proc.stdout)


def _tail(text: str) -> str:
    return " ".join(text.strip().splitlines())[-300:] or "no output"
