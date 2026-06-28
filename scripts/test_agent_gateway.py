import os
import stat
import subprocess
import textwrap
import threading
import time
import unittest
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_gateway.backends import ClaudePrintBackend, CodexExecBackend, MockBackend
from agent_gateway.core import (
    AgentEvent,
    AgentTurn,
    BackendCapabilities,
    GatewayRuntime,
    InjectionBuffer,
    LiveCard,
    ROOT,
    ReplyContext,
    build_interface_report,
    extract_suggestions,
    is_status_frame,
)
from agent_gateway.post_turn import (
    CommitRequest,
    PostTurnPolicy,
    PostTurnRequest,
    PostTurnRunner,
    extract_post_turn_request,
)
from agent_gateway.core import Attachment
from agent_gateway.media import AlbumBuffer, extract_media, outbound_image_paths, save_path, MediaRef
from agent_gateway.merge_gate import MergeGate, base_branch_from_env
from agent_gateway.skills import activation_prompt, discover_skills
from agent_gateway import project as gw_project
from agent_gateway.backends import _gateway_prompt
from agent_gateway.hooks import AugmentationHook, NoOpHook, load_hook
from agent_gateway.profiles import apply_profile, write_profile_template
from agent_gateway.telegram import TelegramGatewayApp, TelegramGatewayConfig, _redact_token
from agent_gateway.worktrees import WorktreeBroker, WorktreePolicy

# The example project (a private brain hook) is excluded from the shipped edition.
# Tests that exercise it skip cleanly when it isn't present, so the shared suite
# stays green both here and in the packaged standalone repo.
_NC_EXAMPLE = ROOT / "scripts" / "agent_gateway" / "examples" / "private" / "brain_hook.py"
_has_example = _NC_EXAMPLE.exists()


class SlowNoSteerBackend:
    capabilities = BackendCapabilities(name="slow", label="Slow Agent", steering=False)

    def __init__(self):
        self.seen = []

    def run(self, turn, emit, stop_event, injections):
        self.seen.append(turn.text)
        time.sleep(0.05)
        return f"done {turn.text}"

    def reset(self, chat_id):
        pass


class FakeTelegramClient:
    def __init__(self):
        self.sent = []
        self.edited = []
        self.downloaded = []
        self.files = []

    def send(self, chat_id, text, reply_markup=None):
        self.sent.append((chat_id, text, reply_markup))
        return len(self.sent)

    def edit(self, chat_id, message_id, text, reply_markup=None):
        self.edited.append((chat_id, message_id, text, reply_markup))

    def send_file(self, chat_id, path, caption=""):
        self.files.append((chat_id, str(path), caption))
        return 100 + len(self.files)

    def answer_callback(self, callback_id):
        pass

    def chat_action(self, chat_id, action="typing"):
        pass  # no-op; the live typing indicator is cosmetic

    def file_path(self, file_id):
        return f"remote/{file_id}.bin"

    def download(self, file_path, dest):
        Path(dest).write_bytes(b"fake-bytes")
        self.downloaded.append((file_path, str(dest)))


class AgentGatewayTests(unittest.TestCase):
    def test_interface_report_captures_best_features(self):
        text = build_interface_report()
        self.assertIn("Reply injection", text)
        self.assertIn("Live progress", text)
        self.assertIn("Skills menu", text)
        self.assertIn("Codex compact", text)
        self.assertIn("Claude warm-worker", text)
        self.assertIn("Continuation buttons", text)
        self.assertIn("Post-turn hook", text)
        self.assertIn("Worktree broker", text)
        self.assertIn("Merge checkpoint", text)

    def test_status_frames_are_not_reply_context(self):
        self.assertTrue(is_status_frame("Agent live · codex · write · 3s\n\n/stop cancels"))
        self.assertTrue(is_status_frame("Codex live · queue: 1"))
        self.assertFalse(is_status_frame("This is a real prior answer."))

    def test_reply_context_renders_into_prompt(self):
        turn = AgentTurn(
            chat_id=1,
            user_id=2,
            text="continue",
            backend="mock",
            reply_context=ReplyContext(author="Claude", text="previous answer"),
        )
        self.assertIn("[Context - replying to Claude]:", turn.prompt_text())
        self.assertIn("previous answer", turn.prompt_text())
        self.assertTrue(turn.prompt_text().endswith("continue"))

    def test_suggestions_are_stripped_and_capped(self):
        clean, suggestions = extract_suggestions(
            "Done.\n\nSUGGEST: run tests | commit scoped diff | queue merge | ignored"
        )
        self.assertEqual(clean, "Done.")
        self.assertEqual(suggestions, ["run tests", "commit scoped diff", "queue merge"])

    def test_post_turn_request_is_stripped_and_parsed(self):
        clean, request = extract_post_turn_request(
            'Done.\nPOSTTURN: {"commit":{"message":"gateway commit","paths":["scripts/x.py"]},"push":true,"merge":"dev"}'
        )
        self.assertEqual(clean, "Done.")
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.commit, CommitRequest(message="gateway commit", paths=("scripts/x.py",)))
        self.assertTrue(request.push)
        self.assertTrue(request.merge)
        self.assertEqual(request.target_branch, "dev")

    def test_telegram_final_controls_strip_markers_and_build_buttons(self):
        runtime = GatewayRuntime({"mock": MockBackend()}, default_backend="mock")
        app = TelegramGatewayApp(TelegramGatewayConfig(token="1:test", allowed_user_ids={1}), runtime)
        text, markup = app._final_text_and_markup(
            10,
            "plan",
            'Done.\nSUGGEST: run tests | queue merge\nPOSTTURN: {"push":true}',
        )
        self.assertNotIn("SUGGEST:", text)
        self.assertNotIn("POSTTURN:", text)
        self.assertIn("post-turn skipped", text)
        self.assertIsNotNone(markup)
        assert markup is not None
        self.assertEqual(len(markup["inline_keyboard"]), 2)
        self.assertIn("10.1", app.suggestions)
        self.assertEqual(app.suggestions["10.1"], "run tests")

    def test_fixed_profile_blocks_agent_switching(self):
        runtime = GatewayRuntime({"mock": MockBackend(), "slow": SlowNoSteerBackend()}, default_backend="mock")
        app = TelegramGatewayApp(TelegramGatewayConfig(token="1:test", allowed_user_ids={1}, fixed_backend=True), runtime)
        fake = FakeTelegramClient()
        app.client = fake  # type: ignore[assignment]

        app._handle_command(10, 1, "/agent slow")

        self.assertEqual(runtime.backend_for_chat(10), "mock")
        self.assertIn("Fixed agent: mock", fake.sent[-1][1])

    def test_profile_loader_maps_simple_bot_file_to_gateway_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "users.env").write_text("ALLOWED_USER_IDS=42\n")
            profile = root / "codex.env"
            profile.write_text(
                "TELEGRAM_TOKEN=123:abc\n"
                "AGENT=codex\n"
            )
            env = {}

            result = apply_profile(str(profile), environ=env)

            self.assertEqual(result.name, "codex")
            self.assertEqual(env["AGENT_GATEWAY_TELEGRAM_TOKEN"], "123:abc")
            self.assertEqual(env["AGENT_GATEWAY_ALLOWED_USER_IDS"], "42")
            self.assertEqual(env["AGENT_GATEWAY_DEFAULT_BACKEND"], "codex")
            # Every bot runs all agents and can switch live (FIXED_BACKEND=0); AGENT is
            # only the default the bot opens on.
            self.assertEqual(env["AGENT_GATEWAY_BACKENDS"], "mock,codex,claude")
            self.assertEqual(env["AGENT_GATEWAY_FIXED_BACKEND"], "0")
            self.assertEqual(env["AGENT_GATEWAY_ALLOW_PUSH"], "0")
            self.assertEqual(env["AGENT_GATEWAY_ALLOW_MERGE"], "0")
            self.assertTrue(env["AGENT_GATEWAY_STATE_DIR"].endswith("state/agent-gateway/runtime/codex"))

    def test_profile_file_can_override_shared_user_ids_for_debugging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "users.env").write_text("ALLOWED_USER_IDS=42\n")
            profile = root / "codex.env"
            profile.write_text(
                "TELEGRAM_TOKEN=123:abc\n"
                "ALLOWED_USER_IDS=99\n"
                "AGENT=codex\n"
            )
            env = {}

            apply_profile(str(profile), environ=env)

            self.assertEqual(env["AGENT_GATEWAY_ALLOWED_USER_IDS"], "99")

    def test_profile_file_can_override_runtime_state_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "users.env").write_text("ALLOWED_USER_IDS=42\n")
            profile = root / "codex.env"
            profile.write_text(
                "TELEGRAM_TOKEN=123:abc\n"
                "AGENT=codex\n"
                "AGENT_GATEWAY_STATE_DIR=/tmp/custom-gateway-state\n"
            )
            env = {}

            apply_profile(str(profile), environ=env)

            self.assertEqual(env["AGENT_GATEWAY_STATE_DIR"], "/tmp/custom-gateway-state")

    def test_profile_template_is_small_and_secret_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_profile_template("claude", profiles_dir=Path(tmp))

            text = path.read_text()
            self.assertIn("TELEGRAM_TOKEN=", text)
            self.assertNotIn("ALLOWED_USER_IDS=", text)
            self.assertIn("AGENT=claude", text)
            # The profile is now just token + default agent; no per-bot machinery.
            self.assertNotIn("FIXED_BACKEND", text)
            self.assertIn("/model", text)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

            shared = Path(tmp) / "users.env"
            self.assertTrue(shared.exists())
            self.assertIn("ALLOWED_USER_IDS=", shared.read_text())
            self.assertEqual(shared.stat().st_mode & 0o777, 0o600)

    def test_telegram_app_uses_env_state_dir_at_runtime(self):
        runtime = GatewayRuntime({"mock": MockBackend()}, default_backend="mock")
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("AGENT_GATEWAY_STATE_DIR")
            os.environ["AGENT_GATEWAY_STATE_DIR"] = tmp
            try:
                app = TelegramGatewayApp(
                    TelegramGatewayConfig(token="1:test", allowed_user_ids={1}, fixed_backend=True),
                    runtime,
                )
            finally:
                if old is None:
                    os.environ.pop("AGENT_GATEWAY_STATE_DIR", None)
                else:
                    os.environ["AGENT_GATEWAY_STATE_DIR"] = old

        self.assertEqual(app.state_dir, Path(tmp))
        self.assertEqual(app.offset_file, Path(tmp) / "offset")

    def test_telegram_error_redaction_hides_bot_token(self):
        text = "https://api.telegram.org/bot123:secret/getUpdates failed"

        redacted = _redact_token(text, "123:secret")

        self.assertNotIn("123:secret", redacted)
        self.assertIn("<telegram-token>", redacted)

    def test_gateway_online_message_is_sent_by_runtime_process(self):
        runtime = GatewayRuntime({"mock": MockBackend()}, default_backend="mock")
        app = TelegramGatewayApp(TelegramGatewayConfig(token="1:test", allowed_user_ids={2, 1}), runtime)
        fake = FakeTelegramClient()
        app.client = fake  # type: ignore[assignment]

        app._send_online_message()

        self.assertEqual([chat_id for chat_id, _, _ in fake.sent], [1, 2])
        self.assertIn("Ryuma", fake.sent[0][1])
        self.assertIn("is live", fake.sent[0][1])

    def test_runtime_steers_when_backend_supports_injection(self):
        events = []
        runtime = GatewayRuntime({"mock": MockBackend(delay=0.08)}, default_backend="mock")

        def emit(event: AgentEvent):
            events.append(event)

        first = AgentTurn(chat_id=1, user_id=1, text="first", backend="mock")
        second = AgentTurn(chat_id=1, user_id=1, text="second", backend="mock")
        self.assertEqual(runtime.submit(first, emit), "started")
        self.assertEqual(runtime.submit(second, emit), "steered")
        self.assertTrue(runtime.wait_for_idle(1, timeout=2))
        finals = [e.text for e in events if e.kind == "final"]
        self.assertEqual(len(finals), 1)
        self.assertIn("Injected:", finals[0])
        self.assertIn("second", finals[0])

    def test_runtime_queues_when_backend_cannot_steer(self):
        backend = SlowNoSteerBackend()
        events = []
        runtime = GatewayRuntime({"slow": backend}, default_backend="slow")

        def emit(event: AgentEvent):
            events.append(event)

        first = AgentTurn(chat_id=1, user_id=1, text="one", backend="slow")
        second = AgentTurn(chat_id=1, user_id=1, text="two", backend="slow")
        self.assertEqual(runtime.submit(first, emit), "started")
        self.assertEqual(runtime.submit(second, emit), "queued")
        self.assertTrue(runtime.wait_for_idle(1, timeout=2))
        self.assertEqual(backend.seen, ["one", "two"])
        finals = [e.text for e in events if e.kind == "final"]
        self.assertEqual(finals, ["done one", "done two"])

    def test_live_card_renders_compact_live_and_final(self):
        card = LiveCard(backend="mock", mode="write", label="Mock Agent")
        card.update(AgentEvent("thinking", "checking repo", data={"stream": True}))
        live = card.render_live()
        self.assertIn("Mock Agent", live)
        self.assertIn("checking repo", live)
        card.update(AgentEvent("tool", "running: pytest", backend="mock", data={"category": "cmd", "cmd": "pytest", "phase": "tests"}))
        live = card.render_live()
        self.assertIn("tests", live)             # phase in header
        self.assertIn("running: pytest", live)   # action line in the live feed
        card.update(AgentEvent("final", "done"))
        # The neutral gateway imposes NO signature — the operator owns the final text.
        self.assertEqual(card.render_final(), "done")

    def _resolve_app(self):
        runtime = GatewayRuntime({"mock": MockBackend()}, default_backend="mock")
        app = TelegramGatewayApp(TelegramGatewayConfig(token="1:test", allowed_user_ids={1}), runtime)
        fake = FakeTelegramClient()
        app.client = fake  # type: ignore[assignment]
        return app, fake

    def test_resolve_in_place_short_answer_morphs_the_card(self):
        # A short answer edits the live card itself — no extra messages, buttons attached.
        app, fake = self._resolve_app()
        markup = {"inline_keyboard": [[{"text": "🌿 LAND", "callback_data": "land:1"}]]}
        app._resolve_in_place(1, 99, "the answer", markup)
        self.assertEqual(len(fake.edited), 1)
        self.assertEqual(fake.edited[0][1], 99)              # edited the live card's message_id
        self.assertEqual(fake.edited[0][3], markup)          # buttons ride the card
        self.assertEqual(fake.sent, [])                      # nothing sent separately

    def test_resolve_in_place_long_answer_never_doubles_the_card(self):
        # A long answer: card becomes chunk 1 (no buttons, no '✅ done' stub),
        # overflow is a bare continuation, buttons sit on the LAST chunk only.
        app, fake = self._resolve_app()
        markup = {"inline_keyboard": [[{"text": "🌿 LAND", "callback_data": "land:1"}]]}
        long_answer = ("A" * 3000) + "\n\n" + ("B" * 3000)
        app._resolve_in_place(1, 99, long_answer, markup)
        self.assertEqual(len(fake.edited), 1)                # exactly ONE card, edited in place
        self.assertEqual(fake.edited[0][1], 99)
        self.assertIsNone(fake.edited[0][3])                 # chunk 1 carries no buttons
        self.assertNotIn("✅ done", fake.edited[0][2])       # the stub is gone for good
        self.assertEqual(len(fake.sent), 1)                  # one bare continuation message
        self.assertEqual(fake.sent[0][2], markup)            # buttons ride the LAST chunk

    def test_resolve_in_place_falls_back_to_send_without_a_card(self):
        # No live card to edit (e.g. a queued turn) → send fresh, buttons on the last.
        app, fake = self._resolve_app()
        markup = {"inline_keyboard": [[{"text": "🌿 LAND", "callback_data": "land:1"}]]}
        app._resolve_in_place(1, None, "the answer", markup)
        self.assertEqual(fake.edited, [])
        self.assertEqual(len(fake.sent), 1)
        self.assertEqual(fake.sent[0][2], markup)

    def test_status_noise_never_reaches_feed_prose_leads_once_flowing(self):
        # Once prose is flowing the feed is thinking-led: status noise NEVER reaches
        # it, and an action that lands AFTER prose started stays in the data panel.
        card = LiveCard(backend="claude", mode="write", label="Claude Code")
        for _ in range(6):
            card.update(AgentEvent("status", "session active", backend="claude"))
        card.update(AgentEvent("thinking", "the real answer", backend="claude", data={"stream": True}))
        card.update(AgentEvent("tool", "$ git status", backend="claude", data={"category": "cmd"}))
        live = card.render_live()
        self.assertIn("the real answer", card.feed)
        self.assertNotIn("session active", card.feed)
        self.assertNotIn("$ git status", card.feed)  # action AFTER prose → panel only
        self.assertIn("$ git status", live)           # ...still visible in the ➡️ now line
        self.assertIn("the real answer", live)
        self.assertNotIn("session active", live)      # pure noise gone entirely

    def test_action_feed_keeps_a_silent_backend_alive(self):
        # Codex app-server streams NO reasoning prose and only emits its agent message
        # at the very end — for minutes the feed would otherwise be a frozen "…thinking".
        # While no prose has streamed, actions seed the feed so the card visibly moves.
        card = LiveCard(backend="codex", mode="write", label="Codex CLI")
        card.update(AgentEvent("status", "session active", backend="codex", data={"phase": "session"}))
        card.update(AgentEvent("tool", "running: npm test", backend="codex", data={"category": "cmd"}))
        card.update(AgentEvent("tool", "editing telegram.py", backend="codex", data={"category": "file"}))
        live = card.render_live()
        self.assertNotIn("…thinking", live)            # no longer frozen
        self.assertIn("running: npm test", card.feed)  # actions are the live feed now
        self.assertIn("editing telegram.py", card.feed)
        self.assertNotIn("session active", card.feed)  # status noise still excluded
        # Once prose finally streams, it flows into the SAME feed below the actions.
        card.update(AgentEvent("thinking", "Here is the result.", backend="codex", data={"stream": True}))
        self.assertIn("Here is the result.", card.feed)

    def test_claude_streams_reasoning_and_answer(self):
        from agent_gateway.backends import _parse_claude_event
        think = '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"let me reason"}}}'
        ev, _ = _parse_claude_event(think)
        self.assertEqual(ev.kind, "thinking")
        self.assertTrue(ev.data.get("stream"))
        self.assertIn("let me reason", ev.text)
        ans = '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"the answer"}}}'
        ev2, _ = _parse_claude_event(ans)
        self.assertTrue(ev2.data.get("stream"))
        self.assertIn("the answer", ev2.text)

    def test_codex_sandbox_grants_extra_writable_roots_when_configured(self):
        from agent_gateway.backends import CodexAppServerBackend
        be = CodexAppServerBackend(sandbox="workspace-write")
        self.assertEqual(be._sandbox_policy(), {"type": "workspaceWrite"})  # default: no holes
        os.environ["AGENT_GATEWAY_CODEX_WRITABLE_ROOTS"] = "/opt/x/state, /tmp/extra"
        try:
            pol = be._sandbox_policy()
            self.assertEqual(pol["type"], "workspaceWrite")
            self.assertEqual(pol["writableRoots"], ["/opt/x/state", "/tmp/extra"])
            # read-only sandboxes are NEVER widened
            self.assertNotIn("writableRoots", CodexAppServerBackend(sandbox="read-only")._sandbox_policy())
        finally:
            del os.environ["AGENT_GATEWAY_CODEX_WRITABLE_ROOTS"]

    def test_gateway_prompt_can_skip_system_for_token_savings(self):
        from agent_gateway.backends import _gateway_prompt
        os.environ["AGENT_GATEWAY_SYSTEM_PROMPT"] = "SYSCTX-MARKER"
        try:
            turn = AgentTurn(1, 1, "do x", "claude")
            self.assertIn("SYSCTX-MARKER", _gateway_prompt(turn, include_system=True))
            self.assertNotIn("SYSCTX-MARKER", _gateway_prompt(turn, include_system=False))
            self.assertIn("do x", _gateway_prompt(turn, include_system=False))
        finally:
            del os.environ["AGENT_GATEWAY_SYSTEM_PROMPT"]

    def test_start_background_seam_noop_and_gated(self):
        from agent_gateway.hooks import NoOpHook
        NoOpHook().start_background(lambda t: None)  # neutral engine runs nothing — must not raise
        if not _has_example:
            return  # shipped edition: no example hook to exercise
        import importlib.util, os
        spec = importlib.util.spec_from_file_location("bh", str(_NC_EXAMPLE))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        os.environ.pop("AGENT_GATEWAY_PRODUCT_WATCH", None)
        m.make_hook().start_background(lambda t: None)  # flag off → no watcher, must not raise

    def test_neutral_bot_imposes_no_final_footer(self):
        # The signature is a PERSONAL-BRAIN feature, not the neutral gateway's. The
        # default NoOpHook imposes nothing — the operator's final message is theirs.
        turn = AgentTurn(1, 1, "hi", "claude", model="opus-4-8", effort="high")
        self.assertEqual(NoOpHook().final_footer(turn), "")

    @unittest.skipUnless(_has_example, "example project excluded from shipped edition")
    def test_personal_brain_signs_with_agent_model_effort(self):
        # The example brain opts into the model · effort · agent signature and,
        # knowing its own setup, prints the honest model even when none is pinned.
        import importlib.util
        path = _NC_EXAMPLE
        spec = importlib.util.spec_from_file_location("_nc_brain_hook", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        hook = mod.make_hook()
        foot = hook.final_footer(AgentTurn(1, 1, "x", "claude"))
        self.assertIn("claude", foot)        # agent
        self.assertIn("opus-4-8", foot)      # honest model from the brain's own config
        self.assertIn("auto", foot)          # effort fallback
        self.assertIn("gpt-5.5", hook.final_footer(AgentTurn(1, 1, "x", "codex")))

    def test_codex_answer_streams_into_feed(self):
        # Parity step: Codex sends its answer as one completed item (no token deltas),
        # so we stream it into the card feed — it must not sit empty until the final.
        from agent_gateway.backends import _parse_codex_event
        line = '{"type":"item.completed","item":{"type":"agent_message","text":"codex answer here"}}'
        ev, reply = _parse_codex_event(line)
        self.assertEqual(reply, "codex answer here")          # still the reply
        self.assertIsNotNone(ev)
        self.assertEqual(ev.data.get("phase"), "writing")     # streamed as the answer
        self.assertTrue(ev.data.get("stream"))
        card = LiveCard(backend="codex", mode="write", label="Codex CLI")
        card.update(ev)
        self.assertIn("codex answer here", card.render_live())  # visible in the live card

    def test_codex_reasoning_event_keeps_thinking_phase(self):
        from agent_gateway.backends import _parse_codex_event
        ev, _ = _parse_codex_event('{"type":"reasoning"}')
        self.assertEqual(ev.data.get("phase"), "thinking")
        ev2, _ = _parse_codex_event('{"type":"agent_reasoning","text":"weighing options"}')
        self.assertTrue(ev2.data.get("stream"))               # streams text when present
        self.assertEqual(ev2.text, "weighing options")

    def test_timeline_interleaves_reasoning_actions_answer(self):
        # The Mini App's flowing stream: consecutive prose of one kind merges into a
        # paragraph; an action waypoint between them breaks it — that's the rhythm.
        card = LiveCard(backend="claude", mode="write", label="Claude Code")
        card.update(AgentEvent("thinking", "let me ", backend="claude", data={"stream": True}))
        card.update(AgentEvent("thinking", "check the file", backend="claude", data={"stream": True}))
        card.update(AgentEvent("tool", "read buckets.ts", backend="claude", data={"category": "tool"}))
        card.update(AgentEvent("thinking", "found it", backend="claude", data={"stream": True}))
        card.update(AgentEvent("thinking", "the answer", backend="claude", data={"stream": True, "phase": "writing"}))
        tl = card.snapshot()["timeline"]
        kinds = [e["t"] for e in tl]
        self.assertEqual(kinds, ["reason", "act", "reason", "answer"])  # merged prose, waypoint splits it
        self.assertEqual(tl[0]["text"], "let me check the file")        # consecutive reason merged
        self.assertEqual(tl[1]["text"], "read buckets.ts")             # the action waypoint
        self.assertIn("started_at", card.snapshot())                    # stable per-turn id for the web view

    def test_stream_chars_gauge_present(self):
        card = LiveCard(backend="claude", mode="write", label="X")
        card.update(AgentEvent("thinking", "hello there friend", backend="claude", data={"stream": True}))
        self.assertIn("hello there friend", card.render_live())  # streamed into the feed
        self.assertEqual(card.snapshot()["chars"], len("hello there friend"))

    def test_live_feed_streams_answer_as_it_writes(self):
        card = LiveCard(backend="claude", mode="write", label="Claude Code")
        for frag in ("Here ", "are 3 ", "ideas."):
            card.update(AgentEvent("thinking", frag, backend="claude", data={"stream": True}))
        live = card.render_live()
        self.assertIn("Here are 3 ideas.", live)  # streamed prose accumulated inline, read-as-it-writes

    def test_feed_marks_think_to_answer_transition(self):
        card = LiveCard(backend="claude", mode="write", label="Claude Code")
        card.update(AgentEvent("thinking", "let me reason about this", backend="claude", data={"stream": True}))
        card.update(AgentEvent("thinking", "the answer", backend="claude", data={"stream": True, "phase": "writing"}))
        self.assertIn("💬", card.feed)  # think→answer break is visible in the chat feed
        self.assertIn("the answer", card.answer)
        self.assertEqual(card.feed.count("💬"), 1)  # only once per turn
        card.update(AgentEvent("thinking", " continues", backend="claude", data={"stream": True, "phase": "writing"}))
        self.assertEqual(card.feed.count("💬"), 1)

    def test_feed_no_answer_marker_without_prior_reasoning(self):
        card = LiveCard(backend="claude", mode="write", label="Claude Code")
        card.update(AgentEvent("thinking", "straight to it", backend="claude", data={"stream": True, "phase": "writing"}))
        self.assertNotIn("💬", card.feed)  # no stray glyph when a turn has no reasoning lead-up

    def test_try_steer_folds_into_active_turn_else_false(self):
        runtime = GatewayRuntime({"mock": MockBackend(delay=0.25)}, default_backend="mock")
        events = []
        self.assertEqual(runtime.submit(AgentTurn(1, 1, "first", "mock"), events.append), "started")
        time.sleep(0.03)
        self.assertTrue(runtime.try_steer(AgentTurn(1, 1, "fold me", "mock")))  # active+steering → folds
        self.assertTrue(runtime.wait_for_idle(1, timeout=2))
        self.assertFalse(runtime.try_steer(AgentTurn(1, 1, "nothing running", "mock")))  # idle → no fold

    # ---- merge gate ----
    def _make_base_repo(self, tmp):
        repo = Path(tmp)
        self._git(repo, "init", "-b", "dev")
        self._git(repo, "config", "user.email", "t@t.t")
        self._git(repo, "config", "user.name", "t")
        (repo / "base.txt").write_text("base\n")
        self._git(repo, "add", "-A")
        self._git(repo, "commit", "-m", "base")
        return repo

    def test_merge_gate_fast_forward_lands(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_base_repo(tmp)
            self._git(repo, "checkout", "-b", "agent/x")
            (repo / "feature.txt").write_text("feature\n")
            self._git(repo, "add", "-A")
            self._git(repo, "commit", "-m", "feature")
            self._git(repo, "checkout", "dev")
            gate = MergeGate(repo=repo, base="dev")
            a = gate.assess("agent/x")
            self.assertTrue(a.needed and a.fast_forward and a.landable)
            self.assertEqual(a.ahead, 1)
            self.assertFalse(a.likely_conflict)
            result = gate.land("agent/x")
            self.assertTrue(result.ok, result.message)
            self.assertTrue((repo / "feature.txt").exists())

    def test_merge_gate_diverged_clean_lands(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_base_repo(tmp)
            self._git(repo, "checkout", "-b", "agent/x")
            (repo / "feature.txt").write_text("feature\n")
            self._git(repo, "add", "-A")
            self._git(repo, "commit", "-m", "feature")
            self._git(repo, "checkout", "dev")
            (repo / "other.txt").write_text("other\n")  # base advances on a different file
            self._git(repo, "add", "-A")
            self._git(repo, "commit", "-m", "advance dev")
            gate = MergeGate(repo=repo, base="dev")
            a = gate.assess("agent/x")
            self.assertTrue(a.needed)
            self.assertFalse(a.fast_forward)
            self.assertFalse(a.likely_conflict)
            self.assertTrue(gate.land("agent/x").ok)

    def test_merge_gate_conflict_aborts_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_base_repo(tmp)
            self._git(repo, "checkout", "-b", "agent/x")
            (repo / "base.txt").write_text("agent edit\n")
            self._git(repo, "add", "-A")
            self._git(repo, "commit", "-m", "agent change base")
            self._git(repo, "checkout", "dev")
            (repo / "base.txt").write_text("dev edit\n")  # same file, conflicting
            self._git(repo, "add", "-A")
            self._git(repo, "commit", "-m", "dev change base")
            gate = MergeGate(repo=repo, base="dev")
            result = gate.land("agent/x")
            self.assertFalse(result.ok)
            self.assertIn("base.txt", result.conflicts)
            # repo must be left clean — no merge in progress, base unchanged
            self.assertFalse((repo / ".git" / "MERGE_HEAD").exists())
            self.assertEqual((repo / "base.txt").read_text(), "dev edit\n")

    def test_base_branch_env_prefers_explicit(self):
        os.environ["AGENT_GATEWAY_BASE_BRANCH"] = "dev"
        try:
            self.assertEqual(base_branch_from_env(), "dev")
        finally:
            del os.environ["AGENT_GATEWAY_BASE_BRANCH"]
        self.assertEqual(base_branch_from_env(default="main"), "main")

    def _land_app(self, repo, branch="agent/x"):
        self._git(repo, "checkout", "-b", branch)
        (repo / "feature.txt").write_text("feature\n")
        self._git(repo, "add", "-A")
        self._git(repo, "commit", "-m", "feature")
        self._git(repo, "checkout", "dev")
        runtime = GatewayRuntime({"mock": MockBackend()}, default_backend="mock")
        app = TelegramGatewayApp(TelegramGatewayConfig(token="1:test", allowed_user_ids={1}), runtime)
        app.client = FakeTelegramClient()  # type: ignore[assignment]
        app.merge_gate = MergeGate(repo=repo, base="dev")
        return app

    def test_merge_gate_refuses_protected_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_base_repo(tmp)
            gate = MergeGate(repo=repo, base="main")  # main protected by default → never land
            a = gate.assess("agent/x")
            self.assertFalse(a.needed)
            self.assertIn("protected", a.reason)            # no button offered
            r = gate.land("agent/x")
            self.assertFalse(r.ok)
            self.assertIn("protected", r.message)           # authoritative refusal too

    def test_protected_branches_configurable(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_base_repo(tmp)
            # dev is not protected by default; explicitly protecting it blocks landing
            self.assertIn("protected", MergeGate(repo=repo, base="dev", protected={"dev"}).assess("agent/x").reason)
            # main is landable when removed from the protected set (generic-repo case)
            self.assertNotIn("protected", MergeGate(repo=repo, base="main", protected=set()).assess("agent/x").reason)

    def test_smart_merge_auto_lands_clean_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_base_repo(tmp)
            app = self._land_app(repo)
            app.smart_merge = True  # pin ON — don't inherit the operator's ambient SMART_MERGE env
            note, row = app._land_controls(1, "agent/x")
            self.assertIn("Saved", note)                    # clean branch saves itself
            self.assertIsNone(row)                          # no tap needed
            self.assertTrue((repo / "feature.txt").exists())  # already on dev

    def test_land_button_gates_when_smart_merge_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_base_repo(tmp)
            app = self._land_app(repo)
            app.smart_merge = False  # fall back to the manual gate
            note, row = app._land_controls(1, "agent/x")
            self.assertIn("keep them or throw them away", note)
            self.assertEqual(len(row), 2)                       # ✅ Keep + 🗑 Throw away
            self.assertTrue(row[0]["callback_data"].startswith("land:"))
            self.assertTrue(row[1]["callback_data"].startswith("disc:"))
            self.assertFalse((repo / "feature.txt").exists())  # nothing landed yet
            key = row[0]["callback_data"].split(":", 1)[1]
            app._handle_land(1, key)
            self.assertTrue((repo / "feature.txt").exists())
            self.assertIn("Saved", app.client.sent[-1][1])

    def test_discard_destroys_worktree_and_branch(self):
        from agent_gateway.worktrees import WorktreeBroker, WorktreePolicy
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_base_repo(tmp)
            wt_root = Path(tmp) / "wt"
            # a live agent worktree with unlanded work
            live = wt_root / "claude" / "live"
            live.parent.mkdir(parents=True, exist_ok=True)
            self._git(repo, "worktree", "add", "-b", "agent/claude/live", str(live), "dev")
            (live / "wip.txt").write_text("wip\n")
            self._git(live, "add", "-A")
            self._git(live, "commit", "-m", "wip")
            broker = WorktreeBroker(WorktreePolicy(enabled=True, repo=repo, root=wt_root, base="dev"))
            note = broker.discard_branch("agent/claude/live")
            self.assertIn("discarded", note)
            self.assertFalse(live.exists())                          # worktree gone
            self.assertNotIn(
                "agent/claude/live",
                self._git(repo, "branch", "--list", "agent/claude/live"),
            )                                                        # branch gone too
            # never touches a non-agent branch (safe by construction)
            self.assertIn("refused", broker.discard_branch("dev"))

    def test_discard_two_tap_confirm_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_base_repo(tmp)
            app = self._land_app(repo)
            app.smart_merge = False
            _, row = app._land_controls(1, "agent/x")
            disc_key = row[1]["callback_data"].split(":", 1)[1]
            # first tap → confirm prompt with a dscok button (no destruction yet)
            app._handle_callback({"id": "c1", "from": {"id": 1}, "message": {"chat": {"id": 1}}, "data": f"disc:{disc_key}"})
            confirm_markup = app.client.sent[-1][2]
            ok = confirm_markup["inline_keyboard"][0][0]["callback_data"]
            self.assertTrue(ok.startswith("dscok:"))

    def test_sweep_startup_keeps_unlanded_prunes_landed(self):
        from agent_gateway.worktrees import WorktreeBroker, WorktreePolicy
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_base_repo(tmp)
            wt_root = Path(tmp) / "wt"
            broker = WorktreeBroker(WorktreePolicy(enabled=True, repo=repo, root=wt_root, base="dev"))
            # worktree with unlanded work → KEEP + report
            live = wt_root / "claude" / "live"
            live.parent.mkdir(parents=True, exist_ok=True)
            self._git(repo, "worktree", "add", "-b", "agent/claude/live", str(live), "dev")
            (live / "wip.txt").write_text("wip\n")
            self._git(live, "add", "-A")
            self._git(live, "commit", "-m", "wip")
            # worktree with no commits ahead of dev → PRUNE (true orphan, branch kept)
            done = wt_root / "claude" / "done"
            done.parent.mkdir(parents=True, exist_ok=True)
            self._git(repo, "worktree", "add", "-b", "agent/claude/done", str(done), "dev")
            reported = broker.sweep_startup("dev")
            self.assertEqual({wt["branch"] for wt in reported}, {"agent/claude/live"})
            self.assertTrue(live.exists())   # unlanded preserved + surfaced
            self.assertFalse(done.exists())  # landed/empty reclaimed

    def test_startup_message_offers_tappable_land(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_base_repo(tmp)
            app = self._land_app(repo)  # agent/x ahead of dev; merge_gate on repo
            app._startup_worktrees = [{"branch": "agent/x", "path": str(repo), "ahead": 1}]
            app._send_online_message()
            markup = app.client.sent[-1][2]
            self.assertIsNotNone(markup)  # restart message carries a keyboard
            btns = [b for row in markup["inline_keyboard"] for b in row]
            self.assertTrue(any(b["callback_data"].startswith("land:") for b in btns))
            # tapping the LAND button merges the surviving branch
            key = btns[0]["callback_data"].split(":", 1)[1]
            app._handle_land(1, key)
            self.assertTrue((repo / "feature.txt").exists())

    # ---- media intake (Phase 2) ----
    def test_extract_media_detects_photo_doc_voice(self):
        photo = extract_media({"photo": [{"file_id": "a"}, {"file_id": "big"}], "caption": "look"})
        self.assertEqual(len(photo), 1)
        self.assertEqual(photo[0].file_id, "big")  # largest size
        self.assertEqual(photo[0].kind, "image")
        self.assertEqual(photo[0].caption, "look")
        doc = extract_media({"document": {"file_id": "d", "mime_type": "image/png", "file_name": "x.png"}})
        self.assertEqual(doc[0].kind, "image")
        self.assertEqual(doc[0].suffix, ".png")
        voice = extract_media({"voice": {"file_id": "v"}})
        self.assertEqual(voice[0].kind, "voice")
        self.assertEqual(extract_media({"text": "hi"}), [])
        # non-image documents are ignored
        self.assertEqual(extract_media({"document": {"file_id": "z", "mime_type": "application/pdf"}}), [])

    def test_handle_media_downloads_and_submits_attachment(self):
        runtime = GatewayRuntime({"mock": MockBackend()}, default_backend="mock")
        app = TelegramGatewayApp(TelegramGatewayConfig(token="1:test", allowed_user_ids={1}), runtime)
        app.client = FakeTelegramClient()  # type: ignore[assignment]
        captured = {}

        def fake_submit(chat_id, user_id, text, reply_context, attachments=()):
            captured["text"] = text
            captured["attachments"] = attachments

        app._submit = fake_submit  # type: ignore[assignment]
        refs = [MediaRef(file_id="big", kind="image", suffix=".jpg", caption="steal this")]
        app._handle_media(1, 1, refs, "steal this")

        self.assertEqual(len(app.client.downloaded), 1)
        self.assertEqual(captured["text"], "steal this")
        self.assertEqual(len(captured["attachments"]), 1)
        self.assertEqual(captured["attachments"][0].kind, "image")

    def test_handle_media_default_text_when_no_caption(self):
        runtime = GatewayRuntime({"mock": MockBackend()}, default_backend="mock")
        app = TelegramGatewayApp(TelegramGatewayConfig(token="1:test", allowed_user_ids={1}), runtime)
        app.client = FakeTelegramClient()  # type: ignore[assignment]
        captured = {}
        app._submit = lambda *a, **k: captured.update(text=a[2], attachments=k.get("attachments", ()))  # type: ignore[assignment]
        app._handle_media(1, 1, [MediaRef("big", "image", ".jpg", "")], "")
        self.assertIn("Read the attached", captured["text"])
        self.assertEqual(len(captured["attachments"]), 1)

    def test_album_buffer_collects_then_flushes_once(self):
        timers = []

        class FakeTimer:
            def __init__(self, delay, fn):
                self.fn = fn
                self.cancelled = False
                timers.append(self)

            def start(self):
                pass

            def cancel(self):
                self.cancelled = True

        flushed = []
        buf = AlbumBuffer(
            lambda chat, user, atts, cap: flushed.append((chat, user, atts, cap)),
            timer_factory=lambda d, fn: FakeTimer(d, fn),
        )
        a1 = Attachment(path=Path("/tmp/a.jpg"), kind="image", caption="one")
        a2 = Attachment(path=Path("/tmp/b.jpg"), kind="image", caption="")
        buf.add("grp", 1, 1, [a1], "one")
        buf.add("grp", 1, 1, [a2], "")  # second photo re-arms; first timer cancelled
        self.assertTrue(timers[0].cancelled)
        # nothing flushed until the debounce fires
        self.assertEqual(flushed, [])
        timers[-1].fn()  # debounce elapses
        self.assertEqual(len(flushed), 1)
        chat, user, atts, cap = flushed[0]
        self.assertEqual(len(atts), 2)  # both photos in ONE turn
        self.assertEqual(cap, "one")

    def test_album_groups_are_independent(self):
        flushed = []
        buf = AlbumBuffer(
            lambda chat, user, atts, cap: flushed.append((atts, cap)),
            timer_factory=lambda d, fn: type("T", (), {"start": lambda s: None, "cancel": lambda s: None})(),
        )
        buf.add("g1", 1, 1, [Attachment(Path("/tmp/x.jpg"), "image")], "")
        buf.add("g2", 1, 1, [Attachment(Path("/tmp/y.jpg"), "image")], "g2cap")
        buf.flush("g2")
        self.assertEqual(len(flushed), 1)
        self.assertEqual(flushed[0][1], "g2cap")

    def test_outbound_image_paths_detects_explicit_and_fresh_generated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "generated"
            root.mkdir()
            explicit = Path(tmp) / "logo.png"
            explicit.write_bytes(b"png")
            old = root / "old.png"
            old.write_bytes(b"old")
            since = time.time()
            fresh = root / "fresh.png"
            fresh.write_bytes(b"fresh")
            os.utime(old, (since - 20, since - 20))
            os.utime(fresh, (since + 1, since + 1))

            paths = outbound_image_paths(f"saved at {explicit}", since=since, roots=[root])
            self.assertEqual(paths, [explicit.resolve(), fresh.resolve()])

    def test_gateway_sends_recent_generated_image_after_final(self):
        class ImageBackend:
            capabilities = BackendCapabilities(name="img", label="Image Agent", steering=False)

            def __init__(self, root: Path):
                self.root = root

            def run(self, turn, emit, stop_event, injections):
                self.root.mkdir(parents=True, exist_ok=True)
                (self.root / "dragon.png").write_bytes(b"fake-png")
                return "Generated the logo."

            def reset(self, chat_id):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "generated"
            old_env = os.environ.get("AGENT_GATEWAY_GENERATED_IMAGE_ROOTS")
            old_wt = os.environ.get("AGENT_GATEWAY_AUTO_WORKTREE")
            os.environ["AGENT_GATEWAY_GENERATED_IMAGE_ROOTS"] = str(root)
            os.environ["AGENT_GATEWAY_AUTO_WORKTREE"] = "0"
            try:
                runtime = GatewayRuntime({"img": ImageBackend(root)}, default_backend="img")
                app = TelegramGatewayApp(TelegramGatewayConfig(token="1:test", allowed_user_ids={1}), runtime)
                app.client = FakeTelegramClient()  # type: ignore[assignment]
                app._submit(1, 1, "make logo", None)
                self.assertTrue(runtime.wait_for_idle(1, timeout=2))
                time.sleep(0.1)
                self.assertEqual(len(app.client.files), 1)
                self.assertTrue(app.client.files[0][1].endswith("dragon.png"))
            finally:
                if old_env is None:
                    os.environ.pop("AGENT_GATEWAY_GENERATED_IMAGE_ROOTS", None)
                else:
                    os.environ["AGENT_GATEWAY_GENERATED_IMAGE_ROOTS"] = old_env
                if old_wt is None:
                    os.environ.pop("AGENT_GATEWAY_AUTO_WORKTREE", None)
                else:
                    os.environ["AGENT_GATEWAY_AUTO_WORKTREE"] = old_wt

    # ---- skill auto-discovery (Phase 2) ----
    def _make_skill(self, base, folder, name, desc):
        d = base / folder
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\nDo the thing.\n")

    def test_discover_skills_parses_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "skills"
            self._make_skill(base, "alpha", "alpha-skill", "does alpha")
            self._make_skill(base, "beta", "beta-skill", "does beta")
            (base / "no-md").mkdir()  # ignored — no SKILL.md
            skills = discover_skills(base, repo=Path(tmp))
            self.assertEqual([s.name for s in skills], ["alpha-skill", "beta-skill"])
            self.assertEqual(skills[0].description, "does alpha")
            self.assertEqual(skills[0].path, "skills/alpha/SKILL.md")
            prompt = activation_prompt(skills[0])
            self.assertIn("alpha-skill", prompt)
            self.assertIn("skills/alpha/SKILL.md", prompt)

    def test_discover_skills_missing_dir_is_empty(self):
        self.assertEqual(discover_skills(Path("/nonexistent/skills/dir")), [])

    def test_skills_menu_and_tap_submits_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "skills"
            self._make_skill(base, "alpha", "alpha-skill", "does alpha")
            os.environ["AGENT_GATEWAY_SKILLS_DIR"] = str(base)
            try:
                runtime = GatewayRuntime({"mock": MockBackend()}, default_backend="mock")
                app = TelegramGatewayApp(TelegramGatewayConfig(token="1:test", allowed_user_ids={1}), runtime)
                app.client = FakeTelegramClient()  # type: ignore[assignment]
                captured = {}
                app._submit = lambda *a, **k: captured.update(text=a[2])  # type: ignore[assignment]
                app._skills_menu(1)
                markup = app.client.sent[-1][2]
                buttons = markup["inline_keyboard"]
                self.assertEqual(buttons[0][0]["text"], "✨ alpha-skill")
                key = buttons[0][0]["callback_data"].split(":", 1)[1]
                app._handle_callback({"id": "c", "from": {"id": 1}, "message": {"chat": {"id": 1}}, "data": f"skill:{key}"})
                self.assertIn("alpha-skill", captured["text"])
            finally:
                del os.environ["AGENT_GATEWAY_SKILLS_DIR"]

    # ---- project decoupling (Phase 3) ----
    def test_core_system_prompt_is_project_neutral_by_default(self):
        for var in ("AGENT_GATEWAY_SYSTEM_PROMPT", "AGENT_GATEWAY_SYSTEM_PROMPT_FILE"):
            os.environ.pop(var, None)
        prompt = gw_project.full_system_prompt()
        self.assertNotIn("ExampleProject", prompt)
        self.assertNotIn("AGENTS.md", prompt)
        # functional contract is always present (it's how the bot works)
        self.assertIn("SUGGEST:", prompt)
        self.assertIn("POSTTURN:", prompt)

    def test_system_prompt_inline_env_override(self):
        os.environ["AGENT_GATEWAY_SYSTEM_PROMPT"] = "Custom project context here."
        try:
            prompt = gw_project.full_system_prompt()
            self.assertIn("Custom project context here.", prompt)
            self.assertIn("SUGGEST:", prompt)  # contract still appended
        finally:
            del os.environ["AGENT_GATEWAY_SYSTEM_PROMPT"]

    def test_system_prompt_file_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "sp.md"
            f.write_text("From a file: serve the Acme repo.")
            os.environ["AGENT_GATEWAY_SYSTEM_PROMPT_FILE"] = str(f)
            os.environ.pop("AGENT_GATEWAY_SYSTEM_PROMPT", None)
            try:
                self.assertIn("serve the Acme repo", gw_project.project_system_prompt())
            finally:
                del os.environ["AGENT_GATEWAY_SYSTEM_PROMPT_FILE"]

    def test_gateway_prompt_carries_project_context_and_request(self):
        os.environ["AGENT_GATEWAY_SYSTEM_PROMPT"] = "ACME-CONTEXT"
        try:
            turn = AgentTurn(chat_id=1, user_id=1, text="fix the bug", backend="codex", mode="write")
            prompt = _gateway_prompt(turn)
            self.assertIn("ACME-CONTEXT", prompt)
            self.assertIn("fix the bug", prompt)
            self.assertNotIn("Mode:", prompt)  # no mode priming — pure raw
        finally:
            del os.environ["AGENT_GATEWAY_SYSTEM_PROMPT"]

    def test_brand_defaults_and_override(self):
        os.environ.pop("AGENT_GATEWAY_BRAND_NAME", None)
        self.assertEqual(gw_project.brand_name(), "Ryuma")
        os.environ["AGENT_GATEWAY_BRAND_NAME"] = "Acme Bot"
        try:
            self.assertEqual(gw_project.brand_name(), "Acme Bot")
        finally:
            del os.environ["AGENT_GATEWAY_BRAND_NAME"]

    # ---- brain augmentation hooks (Phase 3.5) ----
    def test_noop_hook_is_default_and_inert(self):
        os.environ.pop("AGENT_GATEWAY_HOOK", None)
        hook = load_hook()
        self.assertIsInstance(hook, NoOpHook)
        turn = AgentTurn(chat_id=1, user_id=1, text="hi", backend="mock")
        self.assertEqual(hook.before_turn(turn), "")
        self.assertIsNone(hook.after_turn(turn, "done"))

    def test_load_hook_from_file_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            mod = Path(tmp) / "myhook.py"
            mod.write_text(textwrap.dedent(
                """\
                class H:
                    def before_turn(self, turn): return "CTX:" + turn.text
                    def after_turn(self, turn, final_text): pass
                def make_hook(): return H()
                """
            ))
            os.environ["AGENT_GATEWAY_HOOK"] = f"{mod}:make_hook"
            try:
                hook = load_hook()
                self.assertNotIsInstance(hook, NoOpHook)
                self.assertEqual(hook.before_turn(AgentTurn(1, 1, "x", "mock")), "CTX:x")
            finally:
                del os.environ["AGENT_GATEWAY_HOOK"]

    def test_broken_hook_degrades_to_noop(self):
        os.environ["AGENT_GATEWAY_HOOK"] = "does.not.exist:make_hook"
        try:
            self.assertIsInstance(load_hook(), NoOpHook)
        finally:
            del os.environ["AGENT_GATEWAY_HOOK"]

    def test_hook_injects_context_and_records_turn(self):
        calls = {"before": 0, "after": []}

        class RecordingHook:
            def before_turn(self, turn):
                calls["before"] += 1
                return "INJECTED-BRAIN-CONTEXT"

            def after_turn(self, turn, final_text):
                calls["after"].append((turn.text, final_text))

        runtime = GatewayRuntime({"mock": MockBackend()}, default_backend="mock")
        app = TelegramGatewayApp(TelegramGatewayConfig(token="1:test", allowed_user_ids={1}), runtime)
        app.client = FakeTelegramClient()  # type: ignore[assignment]
        app.hook = RecordingHook()  # type: ignore[assignment]
        seen = {}
        orig = runtime.submit

        def spy(turn, emit):
            seen["augment"] = turn.augment
            return orig(turn, emit)

        runtime.submit = spy  # type: ignore[assignment]
        app._submit(1, 1, "do the thing", None)
        self.assertTrue(runtime.wait_for_idle(1, timeout=3))
        self.assertEqual(calls["before"], 1)
        self.assertEqual(seen["augment"], "INJECTED-BRAIN-CONTEXT")  # context reached the turn
        self.assertEqual(len(calls["after"]), 1)  # turn recorded
        self.assertEqual(calls["after"][0][0], "do the thing")

    def test_augment_flows_into_gateway_prompt(self):
        os.environ.pop("AGENT_GATEWAY_SYSTEM_PROMPT", None)
        turn = AgentTurn(chat_id=1, user_id=1, text="fix bug", backend="codex", mode="write", augment="REPO-STATE-XYZ")
        prompt = _gateway_prompt(turn)
        self.assertIn("REPO-STATE-XYZ", prompt)
        self.assertIn("Project context:", prompt)

    @unittest.skipUnless(_has_example, "example project excluded from shipped edition")
    def test_private_example_brain_hook_loads_and_injects(self):
        hook_file = _NC_EXAMPLE
        os.environ["AGENT_GATEWAY_HOOK"] = f"{hook_file}:make_hook"
        try:
            hook = load_hook()
            self.assertNotIsInstance(hook, NoOpHook)
            # before_turn returns a string (real repo state; never raises)
            ctx = hook.before_turn(AgentTurn(1, 1, "x", "claude"))
            self.assertIsInstance(ctx, str)
        finally:
            del os.environ["AGENT_GATEWAY_HOOK"]

    def test_template_example_hook_conforms_and_loads(self):
        # The generic template SHIPS, so this runs everywhere — it proves the
        # worked example actually matches the AugmentationHook contract.
        hook_file = ROOT / "scripts" / "agent_gateway" / "examples" / "template" / "hook.py"
        self.assertTrue(hook_file.is_file())
        os.environ["AGENT_GATEWAY_HOOK"] = f"{hook_file}:make_hook"
        try:
            hook = load_hook()
            self.assertNotIsInstance(hook, NoOpHook)  # the example loaded, not a fallback
            for method in ("before_turn", "after_turn", "final_footer", "start_background"):
                self.assertTrue(callable(getattr(hook, method, None)), method)
            turn = AgentTurn(1, 1, "x", "claude", model="opus-4-8")
            self.assertIsInstance(hook.before_turn(turn), str)   # read-only, never raises
            self.assertIn("claude", hook.final_footer(turn))     # signs with the agent
            self.assertIsNone(hook.start_background(lambda t: None))  # no-op by default
        finally:
            del os.environ["AGENT_GATEWAY_HOOK"]

    # ---- raw power (no modes) + model/effort pass-through (UX round 2) ----
    def test_prompt_has_no_mode_priming(self):
        os.environ.pop("AGENT_GATEWAY_SYSTEM_PROMPT", None)
        turn = AgentTurn(chat_id=1, user_id=1, text="fix the bug", backend="codex")
        prompt = _gateway_prompt(turn)
        self.assertNotIn("Mode:", prompt)
        self.assertNotIn("Read-only", prompt)
        self.assertNotIn("Write-capable", prompt)

    def test_set_mode_rejects_removed_ask(self):
        runtime = GatewayRuntime({"mock": MockBackend()}, default_backend="mock")
        with self.assertRaises(ValueError):
            runtime.set_mode(1, "ask")  # type: ignore[arg-type]
        runtime.set_mode(1, "plan")
        self.assertEqual(runtime.mode_for_chat(1), "plan")

    def test_plan_mode_gets_no_worktree(self):
        broker = WorktreeBroker(WorktreePolicy(enabled=True))
        plan_turn = AgentTurn(chat_id=1, user_id=1, text="just analyze", backend="codex", mode="plan")
        self.assertIsNone(broker.prepare(plan_turn))

    def test_model_effort_per_chat_and_status(self):
        runtime = GatewayRuntime({"mock": MockBackend()}, default_backend="mock")
        runtime.set_model(5, "claude-opus-4-8")
        runtime.set_effort(5, "high")
        self.assertEqual(runtime.model_for_chat(5), "claude-opus-4-8")
        self.assertEqual(runtime.effort_for_chat(5), "high")
        st = runtime.status(5)
        self.assertEqual(st["model"], "claude-opus-4-8")
        self.assertEqual(st["effort"], "high")
        # untouched chat shows backend-default
        self.assertEqual(runtime.status(6)["model"], "(backend default)")

    def test_codex_build_cmd_honors_per_turn_model_and_effort(self):
        backend = CodexExecBackend(model="gpt-5.5", effort="medium")
        cmd = backend.build_cmd("do it", out_path=Path("/tmp/o.txt"), resume_id=None, persistent=True, model="gpt-9-ultra", effort="xhigh")
        self.assertIn("gpt-9-ultra", cmd)
        self.assertIn('model_reasoning_effort="xhigh"', cmd)

    def test_claude_plan_mode_locks_permission_reverts_to_write(self):
        b = ClaudePrintBackend()
        plan_cmd, _, _ = b.build_cmd(1, "claude-opus-4-8", "plan")
        self.assertEqual(plan_cmd[plan_cmd.index("--permission-mode") + 1], "plan")  # hard read-only lock
        build_cmd, _, _ = b.build_cmd(1, "claude-opus-4-8", b.permission_mode)
        self.assertNotEqual(build_cmd[build_cmd.index("--permission-mode") + 1], "plan")  # back to build default

    def test_claude_effort_maps_to_thinking_tokens(self):
        from agent_gateway.backends import _claude_thinking_tokens
        self.assertEqual(_claude_thinking_tokens("", 4000), 4000)  # default
        self.assertEqual(_claude_thinking_tokens("high", 4000), 10000)
        self.assertEqual(_claude_thinking_tokens("12345", 4000), 12345)  # numeric pass-through
        self.assertEqual(_claude_thinking_tokens("off", 4000), 0)

    def test_telegram_model_and_effort_commands(self):
        runtime = GatewayRuntime({"mock": MockBackend()}, default_backend="mock")
        app = TelegramGatewayApp(TelegramGatewayConfig(token="1:test", allowed_user_ids={1}), runtime)
        app.client = FakeTelegramClient()  # type: ignore[assignment]
        app._handle_command(1, 1, "/model claude-opus-4-8")
        self.assertEqual(runtime.model_for_chat(1), "claude-opus-4-8")
        app._handle_command(1, 1, "/effort high")
        self.assertEqual(runtime.effort_for_chat(1), "high")

    def test_skills_command_removed_model_agent_suggest(self):
        from agent_gateway.telegram import BOT_COMMANDS
        from agent_gateway.backends import ClaudePrintBackend
        self.assertNotIn("skills", {c for c, _ in BOT_COMMANDS})  # decluttered
        self.assertIn("claude-opus-4-8", ClaudePrintBackend().capabilities.model_suggestions)  # tap-buttons
        runtime = GatewayRuntime({"mock": MockBackend()}, default_backend="mock")
        app = TelegramGatewayApp(TelegramGatewayConfig(token="1:test", allowed_user_ids={1}), runtime)
        app.client = FakeTelegramClient()  # type: ignore[assignment]
        # the agent→model tree leaf (pm:backend:model) sets BOTH in one tap
        app._handle_callback({"id": "c", "from": {"id": 1}, "message": {"chat": {"id": 1}, "message_id": 7}, "data": "pm:mock:test-model"})
        self.assertEqual(runtime.backend_for_chat(1), "mock")
        self.assertEqual(runtime.model_for_chat(1), "test-model")

    def test_plan_registered_but_no_sticky_mode(self):
        # /plan is a one-shot read-only turn; build is the DEFAULT (no command);
        # no sticky /mode or /write toggle (state to track + clashes with /model).
        from agent_gateway.telegram import BOT_COMMANDS
        names = {c for c, _ in BOT_COMMANDS}
        self.assertIn("plan", names)
        self.assertNotIn("write", names)
        self.assertNotIn("mode", names)

    def test_plan_directive_only_in_plan_mode(self):
        plan = _gateway_prompt(AgentTurn(1, 1, "refactor X", "mock", mode="plan"))
        build = _gateway_prompt(AgentTurn(1, 1, "refactor X", "mock", mode="write"))
        self.assertIn("PLAN MODE", plan)        # read-only directive present
        self.assertNotIn("PLAN MODE", build)    # default build is unconstrained

    def test_plan_command_then_build_button_executes(self):
        runtime = GatewayRuntime({"mock": MockBackend(delay=0.02)}, default_backend="mock")
        app = TelegramGatewayApp(TelegramGatewayConfig(token="1:test", allowed_user_ids={1}), runtime)
        app.client = FakeTelegramClient()  # type: ignore[assignment]
        app.worktrees.prepare = lambda turn: None  # don't create real git worktrees in tests
        # A final answer resolves IN PLACE (edits the live card) or, when there is no
        # card, is sent fresh — so a test must scan BOTH outbound channels.
        def outbound():
            msgs = [(t, m) for _, t, m in app.client.sent]
            msgs += [(t, m) for _, _, t, m in app.client.edited]
            return msgs
        # /plan runs a PLAN-mode turn (mock echoes the mode)
        app._handle_command(1, 1, "/plan add a flag")
        self.assertTrue(runtime.wait_for_idle(1, timeout=3))
        self.assertTrue(any("Mock final for `plan`" in t for t, _ in outbound()))
        # the plan result carries a ⚡ Build this button
        build_key = None
        for _, markup in outbound():
            for row in (markup or {}).get("inline_keyboard", []):
                for btn in row:
                    if str(btn.get("callback_data", "")).startswith("build:"):
                        build_key = btn["callback_data"].split(":", 1)[1]
        self.assertIsNotNone(build_key, "plan result should offer ⚡ Build this")
        # tapping it runs a WRITE-mode build turn
        app._handle_callback({"id": "c", "from": {"id": 1}, "message": {"chat": {"id": 1}}, "data": f"build:{build_key}"})
        self.assertTrue(runtime.wait_for_idle(1, timeout=3))
        self.assertTrue(any("Mock final for `write`" in t for t, _ in outbound()))

    # ---- markdown rendering + steering card (UX round 3) ----
    def test_markdown_renders_to_telegram_html(self):
        from agent_gateway.formatting import md_to_html
        self.assertEqual(md_to_html("**bold**"), "<b>bold</b>")
        self.assertEqual(md_to_html("*it*"), "<i>it</i>")
        self.assertEqual(md_to_html("**a *b* c**"), "<b>a <i>b</i> c</b>")  # nested italic in bold
        self.assertIn("<b>3. Heading</b>", md_to_html("## 3. Heading"))
        self.assertIn("<code>x</code>", md_to_html("`x`"))
        self.assertIn('<a href="https://a.io">', md_to_html("see https://a.io"))
        # HTML special chars in prose are escaped (no injection / parse breakage)
        self.assertIn("&lt;tag&gt;", md_to_html("a <tag> b"))

    def test_long_text_splits_into_html_chunks(self):
        from agent_gateway.formatting import split_message
        big = "para\n\n" * 2000  # well over 4096
        chunks = split_message(big)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 3900 for c in chunks))

    def test_card_shows_steering_for_steering_backend(self):
        steer = LiveCard(backend="claude", mode="write", label="Claude Code", steering=True)
        live = steer.render_live()
        steer.queue_depth = 2
        live = steer.render_live()
        self.assertIn("reply to steer", live)             # steering hint when supported
        self.assertIn("📦 q2", live)                       # queue shown only when non-zero
        noselect = LiveCard(backend="codex", mode="write", label="Codex CLI", steering=False)
        live2 = noselect.render_live()
        self.assertNotIn("steer", live2)                  # no steering hint when unsupported
        self.assertNotIn("📦", live2)                      # queue hidden at zero (feed-forward, no clutter)

    # ---- steering / orphan-drain fidelity vs the specialized bot (UX round 4) ----
    def test_two_phase_drain_absorbs_slow_orphan(self):
        import queue as _q
        from types import SimpleNamespace
        backend = ClaudePrintBackend(claude_bin="x", orphan_quiet_sec=0.05, orphan_drain_sec=2.0)
        lines: _q.Queue = _q.Queue()
        lines.put('{"type": "assistant", "message": {"content": [{"type": "text", "text": "orphan answer"}]}}')
        lines.put('{"type": "result", "result": "orphan answer"}')
        worker = SimpleNamespace(lines=lines)
        chunks = backend._drain_claude_trailing(worker, lambda e: None, deadline=time.time() + 2.0)  # type: ignore[arg-type]
        self.assertIn("orphan answer", "\n\n".join(chunks))

    def test_quiet_window_exits_fast_when_no_orphan(self):
        import queue as _q
        from types import SimpleNamespace
        backend = ClaudePrintBackend(claude_bin="x", orphan_quiet_sec=0.05, orphan_drain_sec=5.0)
        worker = SimpleNamespace(lines=_q.Queue())
        t0 = time.time()
        chunks = backend._drain_claude_trailing(worker, lambda e: None, deadline=time.time() + 5.0)  # type: ignore[arg-type]
        self.assertEqual(chunks, [])
        self.assertLess(time.time() - t0, 1.0)  # phase-1 quiet window, not the full 5s deadline

    def test_drain_queue_empties_stale_lines(self):
        import queue as _q
        from agent_gateway.backends import _drain_queue
        q: _q.Queue = _q.Queue()
        for i in range(3):
            q.put(f"stale-{i}")
        self.assertEqual(_drain_queue(q), ["stale-0", "stale-1", "stale-2"])
        self.assertTrue(q.empty())

    def test_claude_tool_labels_show_real_actions(self):
        from agent_gateway.backends import _claude_tool_label
        self.assertEqual(_claude_tool_label("Bash", {"command": "git status"}), "$ git status")
        self.assertEqual(_claude_tool_label("Edit", {"file_path": "/a/b/core.py"}), "✎ core.py")
        self.assertEqual(_claude_tool_label("Read", {"file_path": "/a/b/foo.py"}), "read foo.py")
        self.assertTrue(_claude_tool_label("Grep", {"pattern": "TODO"}).startswith("grep TODO"))

    def test_cockpit_panel_above_feed(self):
        card = LiveCard(backend="claude", mode="write", label="Claude Code")
        card.usage = "in 6.2k · out 1.6k"
        card.diff = "+40 −8 3f"
        card.update(AgentEvent("thinking", "writing the answer", backend="claude", data={"stream": True}))
        live = card.render_live()
        self.assertIn("⏱", live)               # elapsed clock present
        self.assertIn("📊 +40 −8 3f", live)     # live diff in the panel
        self.assertNotIn("6.2k", live)          # tokens in/out deliberately NOT shown (founder's call)
        self.assertIn("writing the answer", live)  # streamed feed below

    # ---- Mini App live web view (spike) ----
    def test_snapshot_has_live_fields(self):
        card = LiveCard(backend="claude", mode="write", label="Claude Code")
        card.diff = "+1 −0"
        card.usage = "in 1k"
        card.update(AgentEvent("thinking", "hi", backend="claude", data={"stream": True}))
        s = card.snapshot()
        for k in ("label", "clock", "phase", "diff", "usage", "feed", "done"):
            self.assertIn(k, s)
        self.assertFalse(s["done"])
        self.assertIn("hi", s["feed"])

    def test_webapp_button_retired(self):
        # The Mini App / Live view box is retired — the chat is the live view now, so
        # the bot never offers a Live view button regardless of webapp env.
        runtime = GatewayRuntime({"mock": MockBackend()}, default_backend="mock")
        app = TelegramGatewayApp(TelegramGatewayConfig(token="1:t", allowed_user_ids={1}), runtime)
        self.assertIsNone(app._webapp_button(5))
        app.webapp_enabled = True
        app.webapp_url = "https://x.trycloudflare.com"
        app.webapp_token = "k"
        self.assertIsNone(app._webapp_button(5))  # still none — retired for good

    def test_webapp_server_serves_html_and_token_gated_state(self):
        import urllib.request, urllib.error, json as _json
        from agent_gateway.webserver import start_webapp_server
        states = {7: {"label": "Claude Code", "phase": "editing"}}
        srv = start_webapp_server(port=0, token="sek", snapshot_for=states.get)
        try:
            port = srv.server_address[1]
            html = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
            self.assertIn("Ryuma", html)
            body = urllib.request.urlopen(f"http://127.0.0.1:{port}/state?chat=7&key=sek").read().decode()
            self.assertEqual(_json.loads(body)["phase"], "editing")
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/state?chat=7&key=bad")
            self.assertEqual(ctx.exception.code, 403)
        finally:
            srv.shutdown()

    def test_telegram_auto_land_when_enabled_and_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_base_repo(tmp)
            self._git(repo, "checkout", "-b", "agent/x")
            (repo / "feature.txt").write_text("feature\n")
            self._git(repo, "add", "-A")
            self._git(repo, "commit", "-m", "feature")
            self._git(repo, "checkout", "dev")
            runtime = GatewayRuntime({"mock": MockBackend()}, default_backend="mock")
            app = TelegramGatewayApp(TelegramGatewayConfig(token="1:test", allowed_user_ids={1}), runtime)
            app.client = FakeTelegramClient()  # type: ignore[assignment]
            app.merge_gate = MergeGate(repo=repo, base="dev")
            app.auto_land = True
            app.smart_merge = True  # pin ON — don't inherit the operator's ambient SMART_MERGE env
            note, row = app._land_controls(1, "agent/x")
            self.assertIsNone(row)  # no button — it saved itself
            self.assertIn("Saved", note)
            self.assertTrue((repo / "feature.txt").exists())

    def test_post_turn_runner_commits_scoped_paths_and_queues_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._git(root, "init")
            self._git(root, "config", "user.email", "agent@example.test")
            self._git(root, "config", "user.name", "Agent Gateway")
            (root / "scoped.txt").write_text("base\n")
            (root / "dirty.txt").write_text("base\n")
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "base")

            (root / "scoped.txt").write_text("changed\n")
            (root / "new.txt").write_text("new\n")
            (root / "dirty.txt").write_text("still dirty\n")
            queue_dir = root / "merge-queue"
            runner = PostTurnRunner(
                cwd=root,
                policy=PostTurnPolicy(auto_commit=True, allow_push=False, allow_merge=False, merge_queue_dir=queue_dir),
            )
            result = runner.run(
                PostTurnRequest(
                    commit=CommitRequest("gateway scoped commit", ("scoped.txt", "new.txt")),
                    push=True,
                    merge=True,
                    target_branch="dev",
                )
            ).render()

            self.assertIn("committed", result)
            self.assertIn("push skipped", result)
            self.assertIn("merge queued", result)
            self.assertEqual(self._git(root, "log", "-1", "--pretty=%s").strip(), "gateway scoped commit")
            status = self._git(root, "status", "--short")
            self.assertIn(" M dirty.txt", status)
            self.assertNotIn("scoped.txt", status)
            self.assertNotIn("new.txt", status)
            self.assertTrue(list(queue_dir.glob("*.json")))

    def test_worktree_broker_creates_and_reuses_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            wt_root = root / "worktrees"
            repo.mkdir()
            self._git(repo, "init")
            self._git(repo, "config", "user.email", "agent@example.test")
            self._git(repo, "config", "user.name", "Agent Gateway")
            (repo / "README.md").write_text("base\n")
            self._git(repo, "add", "README.md")
            self._git(repo, "commit", "-m", "base")

            broker = WorktreeBroker(WorktreePolicy(enabled=True, repo=repo, root=wt_root))
            turn = AgentTurn(chat_id=42, user_id=1, text="build worktree broker", backend="codex", mode="write")
            first = broker.prepare(turn)
            self.assertIsNotNone(first)
            assert first is not None
            self.assertTrue(first.path.is_dir())
            self.assertTrue(first.branch.startswith("agent/codex/42-"))

            second = broker.prepare(turn)
            self.assertIsNotNone(second)
            assert second is not None
            self.assertTrue(second.reused)
            self.assertEqual(second.path, first.path)

    def _git(self, cwd: Path, *args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        return proc.stdout

    def test_runtime_backend_and_mode_prefs(self):
        runtime = GatewayRuntime({"mock": MockBackend()}, default_backend="mock")
        runtime.set_mode(7, "plan")
        runtime.select_backend(7, "mock")
        status = runtime.status(7)
        self.assertEqual(status["mode"], "plan")
        self.assertEqual(status["backend"], "mock")

    def test_codex_backend_persists_thread_id_from_fake_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "fake_codex.py"
            fake.write_text(textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import json
                import os
                import pathlib
                import sys

                out = None
                if "--output-last-message" in sys.argv:
                    out = pathlib.Path(sys.argv[sys.argv.index("--output-last-message") + 1])
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text("codex file reply")
                pathlib.Path("cwd.txt").write_text(os.getcwd())
                print(json.dumps({"type": "thread.started", "thread_id": "thread-a"}), flush=True)
                print(json.dumps({"type": "turn.started"}), flush=True)
                print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "codex stream reply"}}), flush=True)
                print(json.dumps({"type": "turn.completed", "usage": {"total_tokens": 3}}), flush=True)
                """
            ))
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            backend = CodexExecBackend(
                codex_bin=str(fake),
                model="gpt-test",
                sandbox="workspace-write",
                timeout_sec=5,
                workdir=root,
                runs_dir=root / "runs",
            )
            events = []
            workdir = root / "assigned"
            workdir.mkdir()
            turn = AgentTurn(chat_id=9, user_id=1, text="write it", backend="codex", mode="write", workdir=workdir)
            reply = backend.run(turn, events.append, threading.Event(), InjectionBuffer())
            self.assertEqual(reply, "codex file reply")
            self.assertEqual((workdir / "cwd.txt").read_text(), str(workdir))
            self.assertEqual(backend.sessions[9]["thread_id"], "thread-a")
            self.assertEqual(backend.sessions[9]["turns"], 1)

            cmd = backend.build_cmd("again", out_path=root / "out.txt", resume_id="thread-a", persistent=True)
            self.assertEqual(cmd[:3], [str(fake), "exec", "resume"])

    def test_claude_backend_warm_worker_steers_injected_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "fake_claude.py"
            fake.write_text(textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import json
                import sys
                import time

                print(json.dumps({"type": "system", "subtype": "init"}), flush=True)
                turn = 0
                for line in sys.stdin:
                    turn += 1
                    json.loads(line)
                    time.sleep(0.18)
                    text = f"answer {turn}"
                    print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}), flush=True)
                    print(json.dumps({"type": "result", "result": text, "usage": {"total_tokens": turn}}), flush=True)
                """
            ))
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            backend = ClaudePrintBackend(
                claude_bin=str(fake),
                model="",
                permission_mode="default",
                timeout_sec=5,
                workdir=root,
                home=root.as_posix(),
                thinking_tokens=0,
            )
            runtime = GatewayRuntime({"claude": backend}, default_backend="claude")
            events = []
            first = AgentTurn(chat_id=3, user_id=1, text="first", backend="claude", mode="write")
            second = AgentTurn(chat_id=3, user_id=1, text="second", backend="claude", mode="write")
            self.assertEqual(runtime.submit(first, events.append), "started")
            time.sleep(0.03)
            self.assertEqual(runtime.submit(second, events.append), "steered")
            self.assertTrue(runtime.wait_for_idle(3, timeout=5))
            finals = [e.text for e in events if e.kind == "final"]
            self.assertEqual(len(finals), 1)
            self.assertIn("answer 1", finals[0])
            self.assertIn("answer 2", finals[0])
            self.assertTrue(backend.capabilities.steering)
            backend.reset(3)
            self.assertNotIn(3, backend._workers)

    def test_claude_backend_self_heals_on_session_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "fake_claude.py"
            fake.write_text(textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import json, sys
                from pathlib import Path
                counter = Path(__file__).with_name("spawns.txt")
                n = (int(counter.read_text()) if counter.exists() else 0) + 1
                counter.write_text(str(n))
                if n == 1:
                    # First spawn: simulate a lost resumed session and exit.
                    print("API error: No conversation found with session id abc", file=sys.stderr, flush=True)
                    sys.exit(1)
                print(json.dumps({"type": "system", "subtype": "init"}), flush=True)
                for line in sys.stdin:
                    json.loads(line)
                    print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "healed answer"}]}}), flush=True)
                    print(json.dumps({"type": "result", "result": "healed answer", "usage": {"input_tokens": 1}}), flush=True)
                """
            ))
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            backend = ClaudePrintBackend(
                claude_bin=str(fake),
                model="",
                permission_mode="default",
                timeout_sec=5,
                workdir=root,
                home=root.as_posix(),
                thinking_tokens=0,
            )
            runtime = GatewayRuntime({"claude": backend}, default_backend="claude")
            events = []
            turn = AgentTurn(chat_id=7, user_id=1, text="do it", backend="claude", mode="write")
            self.assertEqual(runtime.submit(turn, events.append), "started")
            self.assertTrue(runtime.wait_for_idle(7, timeout=8))
            finals = [e.text for e in events if e.kind == "final"]
            errors = [e.text for e in events if e.kind == "error"]
            statuses = [e.text for e in events if e.kind == "status"]
            self.assertEqual(errors, [])  # drift was healed, not surfaced
            self.assertEqual(len(finals), 1)
            self.assertIn("healed answer", finals[0])
            self.assertTrue(any("drift" in s.lower() for s in statuses))
            self.assertEqual((root / "spawns.txt").read_text(), "2")  # exactly one retry
            backend.reset(7)

    def test_claude_backend_surfaces_non_drift_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "fake_claude.py"
            fake.write_text(textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import sys
                print("API error: usage limit reached", file=sys.stderr, flush=True)
                sys.exit(1)
                """
            ))
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            backend = ClaudePrintBackend(claude_bin=str(fake), model="", timeout_sec=5, workdir=root, home=root.as_posix(), thinking_tokens=0)
            runtime = GatewayRuntime({"claude": backend}, default_backend="claude")
            events = []
            turn = AgentTurn(chat_id=8, user_id=1, text="do it", backend="claude", mode="write")
            runtime.submit(turn, events.append)
            self.assertTrue(runtime.wait_for_idle(8, timeout=8))
            errors = [e.text for e in events if e.kind == "error"]
            statuses = [e.text for e in events if e.kind == "status"]
            self.assertTrue(any("usage limit" in e.lower() for e in errors))
            self.assertFalse(any("drift" in s.lower() for s in statuses))  # non-drift: no heal attempted
            backend.reset(8)

    def test_claude_backend_restarts_after_session_turn_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "fake_claude.py"
            fake.write_text(textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import json
                import sys
                import uuid

                session = str(uuid.uuid4())
                print(json.dumps({"type": "system", "subtype": "init"}), flush=True)
                for line in sys.stdin:
                    json.loads(line)
                    print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": session}]}}), flush=True)
                    print(json.dumps({"type": "result", "result": session, "usage": {"total_tokens": 1}}), flush=True)
                """
            ))
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            backend = ClaudePrintBackend(
                claude_bin=str(fake),
                model="",
                permission_mode="default",
                timeout_sec=5,
                workdir=root,
                home=root.as_posix(),
                thinking_tokens=0,
                max_session_turns=1,
            )
            events = []
            turn = AgentTurn(chat_id=4, user_id=1, text="first", backend="claude", mode="write")
            first = backend.run(turn, events.append, threading.Event(), InjectionBuffer())
            second = backend.run(turn, events.append, threading.Event(), InjectionBuffer())
            self.assertNotEqual(first, second)
            self.assertTrue(any("turn cap reached" in event.text for event in events))
            self.assertEqual(backend._turns[4], 1)
            backend.reset(4)

    def test_gateway_start_script_prefers_independent_systemd_units(self):
        text = (Path(__file__).resolve().parent / "agent_gateway_start.sh").read_text()

        self.assertIn("systemd-run", text)
        self.assertIn("AGENT_GATEWAY_UNIT_PREFIX", text)  # unit prefix is templated, not hardcoded
        self.assertIn("AGENT_GATEWAY_SEND_ONLINE_MESSAGE=1", text)
        self.assertIn("--status", text)

    def test_claude_text_delta_is_recoverable_as_writing_stream(self):
        # The empty-final fallback recovers the answer from text_delta stream events, so
        # they MUST parse as a 'writing' stream the backend can accumulate. Locks that.
        import json
        from agent_gateway.backends import _parse_claude_event
        line = json.dumps({"type": "stream_event", "event": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "the recovered answer"}}})
        event, final = _parse_claude_event(line)
        self.assertIsNone(final)  # not in final_chunks...
        self.assertEqual(event.kind, "thinking")
        self.assertTrue(event.data.get("stream"))
        self.assertEqual(event.data.get("phase"), "writing")  # ...recovered via the streamed answer
        self.assertEqual(event.text, "the recovered answer")

    def test_codex_app_server_backend_advertises_steering(self):
        from agent_gateway.backends import CodexAppServerBackend, _codex_item_label

        be = CodexAppServerBackend()
        caps = be.capabilities
        # The whole point of app-server over exec: codex can now fold mid-turn.
        self.assertTrue(caps.steering)
        self.assertTrue(caps.streams)
        self.assertEqual(caps.name, "codex")
        self.assertIn("running:", _codex_item_label({"command": ["bash", "-lc", "ls"]}))
        self.assertIn("editing", _codex_item_label({"path": "/repo/app.py"}))

    def test_bot_id_from_token_uses_numeric_prefix(self):
        from agent_gateway.livestore import bot_id_from_token

        self.assertEqual(bot_id_from_token("8123456:AAEsecretpart"), "8123456")
        # Malformed token -> sanitized fallback (the profile name), never the secret.
        self.assertEqual(bot_id_from_token("not-a-token", fallback="claude 2"), "claude-2")
        self.assertEqual(bot_id_from_token("", fallback=""), "bot")

    def test_livestore_round_trips_and_is_keyed_by_bot_and_chat(self):
        from agent_gateway.livestore import LiveStore

        with tempfile.TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp), debounce_sec=0.0)
            store.write("8123", 42, {"phase": "writing", "clock": "0:07"})
            store.write("8999", 42, {"phase": "thinking"})  # different bot, same chat
            self.assertEqual(store.read("8123", 42)["clock"], "0:07")
            self.assertEqual(store.read("8999", 42)["phase"], "thinking")
            # Unknown bot/chat is empty, never an error.
            self.assertEqual(store.read("nope", 1), {})

    def test_livestore_debounces_but_always_flushes_terminal(self):
        from agent_gateway.livestore import LiveStore

        with tempfile.TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp), debounce_sec=60.0)
            store.write("b", 1, {"clock": "0:01"})
            store.write("b", 1, {"clock": "0:02"})  # debounced away
            self.assertEqual(store.read("b", 1)["clock"], "0:01")
            store.write("b", 1, {"clock": "0:03", "done": True})  # terminal flushes
            self.assertEqual(store.read("b", 1)["clock"], "0:03")

    def test_webserver_store_mode_reads_by_bot(self):
        import json
        import urllib.error
        import urllib.request
        from agent_gateway.livestore import LiveStore
        from agent_gateway.webserver import start_webapp_server

        with tempfile.TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp), debounce_sec=0.0)
            store.write("8123", 7, {"phase": "writing", "answer": "hi"})
            server = start_webapp_server(port=0, token="secret", store_read=store.read)
            try:
                port = server.server_address[1]
                base = f"http://127.0.0.1:{port}/state"
                with urllib.request.urlopen(f"{base}?bot=8123&chat=7&key=secret") as r:
                    self.assertEqual(json.loads(r.read())["answer"], "hi")
                # Wrong token is forbidden.
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(f"{base}?bot=8123&chat=7&key=bad")
                self.assertEqual(ctx.exception.code, 403)
            finally:
                server.shutdown()


class OnboardTest(unittest.TestCase):
    def test_validate_token_shape(self):
        from agent_gateway.onboard import validate_token_shape

        self.assertTrue(validate_token_shape("123456789:AAE-abcdefghijklmnopqrstuvwxyz0123456"))
        self.assertFalse(validate_token_shape("not-a-token"))
        self.assertFalse(validate_token_shape("123:short"))
        self.assertFalse(validate_token_shape(""))

    def test_parse_getme(self):
        from agent_gateway.onboard import parse_getme

        self.assertEqual(parse_getme({"ok": True, "result": {"username": "my_bot"}}), "my_bot")
        self.assertIsNone(parse_getme({"ok": False}))
        self.assertIsNone(parse_getme({"ok": True, "result": {}}))

    def test_first_user_id_across_shapes(self):
        from agent_gateway.onboard import first_user_id

        msg = {"ok": True, "result": [{"update_id": 1, "message": {"from": {"id": 4242}}}]}
        self.assertEqual(first_user_id(msg), 4242)
        edited = {"ok": True, "result": [{"update_id": 2, "edited_message": {"from": {"id": 7}}}]}
        self.assertEqual(first_user_id(edited), 7)
        self.assertIsNone(first_user_id({"ok": True, "result": []}))
        self.assertIsNone(first_user_id({"ok": False}))

    def test_detect_agents_appends_mock(self):
        from agent_gateway.onboard import detect_agents

        found = detect_agents(which=lambda name: "/usr/bin/claude" if name == "claude" else None)
        self.assertEqual(found, ["claude", "mock"])
        self.assertEqual(detect_agents(which=lambda name: None), ["mock"])

    def test_capture_user_id_drains_backlog_then_captures_fresh(self):
        from agent_gateway.onboard import capture_user_id

        calls = []

        def fake_http(method, token, params):
            calls.append(dict(params))
            if len(calls) == 1:  # initial drain: a stale update we must skip
                return {"ok": True, "result": [{"update_id": 100, "message": {"from": {"id": 999}}}]}
            return {"ok": True, "result": [{"update_id": 101, "message": {"from": {"id": 555}}}]}

        uid = capture_user_id("tok", http=fake_http, sleep=lambda s: None)
        self.assertEqual(uid, 555)  # fresh message, not the drained 999
        self.assertEqual(calls[1].get("offset"), 101)  # polled past the drained backlog

    def test_capture_user_id_times_out(self):
        from agent_gateway.onboard import capture_user_id

        empty = lambda *a, **k: {"ok": True, "result": []}
        self.assertIsNone(capture_user_id("tok", http=empty, sleep=lambda s: None, attempts=3))


if __name__ == "__main__":
    unittest.main()
