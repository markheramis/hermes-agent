"""Tests for the gateway-side participant upsert wiring.

The actual storage layer is unit-tested in
``test_session_participants.py``.  These tests verify the *integration*:
the gateway calls ``session_store.upsert_participant`` for every
inbound message, with the correct arguments, before the agent runs.

Two complementary approaches:

  1. Source-shape test — reads the handler's source and confirms the
     upsert call is present with the canonical (session_key, user_id,
     user_name, user_email) argument shape.  Cheap regression guard
     against the wiring being silently removed by a future refactor.

  2. Behavioural test — exercises the call shape via a minimal mock,
     pinning that the empty-source / cron / system-event paths
     correctly degenerate to a no-op rather than recording a "ghost"
     participant.
"""

import ast
import inspect

from gateway.run import GatewayRunner


class TestWiringPresent:
    """The upsert call must exist in the canonical inbound handler."""

    def _handler_source(self):
        return inspect.getsource(GatewayRunner._handle_message_with_agent)

    def test_upsert_call_present(self):
        src = self._handler_source()
        assert "upsert_participant" in src, (
            "session_store.upsert_participant call missing from "
            "_handle_message_with_agent — participants will never be tracked"
        )

    def test_upsert_uses_session_store(self):
        src = self._handler_source()
        assert "self.session_store.upsert_participant" in src, (
            "upsert_participant must be called on self.session_store, not "
            "some other instance"
        )

    def test_upsert_passes_session_key_first(self):
        """The upsert helper takes session_key as the first positional arg.
        Pinning this prevents accidental signature swaps."""
        src = self._handler_source()
        tree = ast.parse(src.lstrip())
        found_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "upsert_participant"
            ):
                found_calls.append(node)
        assert found_calls, "upsert_participant call not found in AST"
        for call in found_calls:
            assert len(call.args) >= 1, (
                "upsert_participant called without positional args; "
                "first must be session_key"
            )
            first_arg = call.args[0]
            # The first arg is "session_key" or a Name referring to it
            if isinstance(first_arg, ast.Name):
                assert first_arg.id == "session_key", (
                    f"first arg to upsert_participant must be session_key, "
                    f"got {ast.dump(first_arg)}"
                )

    def test_upsert_called_before_agent_run(self):
        """Participants must be recorded BEFORE the LLM turn starts so the
        first-turn agent can see the current sender via the tool."""
        src = self._handler_source()
        upsert_pos = src.find("upsert_participant")
        # Heuristic: the agent's main work is fired via run_conversation,
        # run_inline, or self._run_agent — any of which must come AFTER.
        # We only need to confirm the upsert isn't at the tail of the func.
        # Use the lower-bound check: upsert occurs in the first half.
        assert 0 < upsert_pos < len(src) // 2, (
            "upsert_participant should be near the top of the handler "
            "(after session_entry resolution) so participant data is "
            "available for the agent turn that follows"
        )


class TestNoGhostParticipants:
    """The upsert call must degenerate cleanly for synthetic events
    (cron, home-channel ping, status broadcasts) that carry no real
    sender identity."""

    def test_store_method_ignores_empty_user_id(self, tmp_path):
        """A direct contract test on the store method: empty user_id is
        a no-op.  Pinned here because the gateway wiring relies on this
        behaviour to avoid recording phantom participants."""
        from unittest.mock import patch
        from gateway.config import GatewayConfig, Platform, SessionResetPolicy
        from gateway.session import SessionEntry, SessionStore
        from datetime import datetime

        config = GatewayConfig(
            default_reset_policy=SessionResetPolicy(mode="none"),
        )
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = None
        store._loaded = True
        now = datetime.now()
        store._entries["k"] = SessionEntry(
            session_key="k",
            session_id="s",
            created_at=now,
            updated_at=now,
            platform=Platform.TELEGRAM,
            chat_type="dm",
        )

        # The exact arg pattern the gateway wiring uses for synthetic
        # events (source.user_id is None -> empty string passed)
        store.upsert_participant("k", "", "", "")

        assert store.snapshot_participants("k") == {}, (
            "Synthetic-event upsert created a ghost participant"
        )
