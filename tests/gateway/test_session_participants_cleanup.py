"""Tests for the cleanup paths that wipe the per-session participant roster.

Five cleanup paths reinforce the same guarantee (participant data must
not outlive its session):

  1. ``reset_session()`` — explicit ``old_entry.participants.clear()``
     before the swap.  Covers ``/new``, ``/reset``, and suspend-driven
     auto-resets.
  2. ``_session_expiry_watcher`` (gateway) — explicit
     ``clear_participants_for_session(session_id)`` after the
     ``on_session_finalize`` hook, before agent eviction.
  3. ``_finalize_shutdown_agents`` (gateway) — covered by the internal
     ``on_session_finalize`` listener (no explicit call needed since
     the process is about to exit anyway).
  4. ``prune_old_entries`` — old entries are popped from ``_entries``
     and GC-reclaims their participant dicts (no explicit clear).
  5. Internal ``on_session_finalize`` listener registered once at
     gateway init — belt-and-suspenders for any cleanup site we
     haven't enumerated.

These tests pin the explicit-call paths and the listener.  GC-driven
paths (4 above) are covered by ``test_session_participants.py::TestPersistenceExclusion``
and by Python's reference-counting model.
"""

import importlib
from datetime import datetime
from unittest.mock import patch

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionEntry, SessionStore


def _make_store(tmp_path):
    config = GatewayConfig(
        default_reset_policy=SessionResetPolicy(mode="none"),
    )
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db = None
    store._loaded = True
    return store


def _add_entry(store, session_key="k1", session_id="sess-A"):
    now = datetime.now()
    entry = SessionEntry(
        session_key=session_key,
        session_id=session_id,
        created_at=now,
        updated_at=now,
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    store._entries[session_key] = entry
    return entry


# ---------------------------------------------------------------------------
# Path 1 — reset_session explicit clear
# ---------------------------------------------------------------------------

class TestResetSessionClearsRoster:
    def test_old_entry_dict_emptied_in_place(self, tmp_path):
        """Old entry's dict is cleared in place — any stale reference
        holders see an empty roster after reset."""
        store = _make_store(tmp_path)
        old_entry = _add_entry(store, session_key="k", session_id="sess-OLD")
        store.upsert_participant("k", "u1", "Alice", "alice@x.com")
        store.upsert_participant("k", "u2", "Bob", "")
        assert len(old_entry.participants) == 2

        store.reset_session("k")

        # The dict we held a reference to is now empty
        assert old_entry.participants == {}, (
            "Old entry roster was not wiped; stale references could leak PII"
        )

    def test_new_entry_starts_empty(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store, session_key="k", session_id="sess-OLD")
        store.upsert_participant("k", "u1", "Alice", "")
        store.upsert_participant("k", "u2", "Bob", "")

        new_entry = store.reset_session("k")

        assert new_entry is not None
        assert new_entry.participants == {}
        assert new_entry.session_id != "sess-OLD"

    def test_reset_does_not_touch_other_sessions(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store, session_key="k1", session_id="sess-A")
        _add_entry(store, session_key="k2", session_id="sess-B")
        store.upsert_participant("k1", "u1", "Alice", "")
        store.upsert_participant("k2", "u2", "Bob", "")

        store.reset_session("k1")

        # k2's roster is untouched
        snap = store.snapshot_participants("k2")
        assert "u2" in snap
        assert snap["u2"]["name"] == "Bob"

    def test_reset_unknown_session_is_noop(self, tmp_path):
        store = _make_store(tmp_path)
        result = store.reset_session("does-not-exist")
        assert result is None  # existing contract preserved


# ---------------------------------------------------------------------------
# Path 5 — internal on_session_finalize listener
# ---------------------------------------------------------------------------

class TestFinalizeListener:
    """The listener registered by ``_register_participant_cleanup_listener``
    clears the participant roster when ``on_session_finalize`` fires."""

    def setup_method(self):
        # Force a fresh registration each test so we have a known state.
        import gateway.run as gateway_run
        gateway_run._PARTICIPANT_CLEANUP_LISTENER_REGISTERED = False
        # Clear any pre-existing listeners on the hook to keep counting honest
        from hermes_cli.plugins import get_plugin_manager
        get_plugin_manager()._hooks.pop("on_session_finalize", None)

    def teardown_method(self):
        # Reset state so we don't leak listeners across tests
        import gateway.run as gateway_run
        gateway_run._PARTICIPANT_CLEANUP_LISTENER_REGISTERED = False
        from hermes_cli.plugins import get_plugin_manager
        get_plugin_manager()._hooks.pop("on_session_finalize", None)

    def test_listener_clears_roster_on_hook_fire(self, tmp_path):
        from gateway.run import _register_participant_cleanup_listener
        from hermes_cli.plugins import invoke_hook

        store = _make_store(tmp_path)
        _add_entry(store, session_key="k", session_id="sess-X")
        store.upsert_participant("k", "u1", "Alice", "alice@x.com")
        assert store.snapshot_participants("k") != {}

        _register_participant_cleanup_listener(store)
        invoke_hook("on_session_finalize", session_id="sess-X", platform="telegram")

        assert store.snapshot_participants("k") == {}

    def test_listener_registered_only_once(self, tmp_path):
        from gateway.run import _register_participant_cleanup_listener
        from hermes_cli.plugins import get_plugin_manager

        store = _make_store(tmp_path)
        _register_participant_cleanup_listener(store)
        _register_participant_cleanup_listener(store)
        _register_participant_cleanup_listener(store)

        hooks = get_plugin_manager()._hooks.get("on_session_finalize", [])
        assert len(hooks) == 1, (
            f"Listener registered {len(hooks)} times; should be exactly 1 "
            "to avoid duplicate work and leak warnings"
        )

    def test_listener_ignores_missing_session_id(self, tmp_path):
        from gateway.run import _register_participant_cleanup_listener
        from hermes_cli.plugins import invoke_hook

        store = _make_store(tmp_path)
        _add_entry(store, session_key="k", session_id="sess-Y")
        store.upsert_participant("k", "u1", "Alice", "")

        _register_participant_cleanup_listener(store)
        # Missing session_id should be silently ignored — must NOT wipe
        # everyone's roster.
        invoke_hook("on_session_finalize", session_id=None, platform="telegram")
        invoke_hook("on_session_finalize", session_id="", platform="telegram")

        assert store.snapshot_participants("k") != {}

    def test_listener_ignores_unknown_session_id(self, tmp_path):
        from gateway.run import _register_participant_cleanup_listener
        from hermes_cli.plugins import invoke_hook

        store = _make_store(tmp_path)
        _add_entry(store, session_key="k", session_id="sess-Z")
        store.upsert_participant("k", "u1", "Alice", "")

        _register_participant_cleanup_listener(store)
        invoke_hook("on_session_finalize", session_id="sess-NOT-REAL", platform="telegram")

        # No collateral
        assert store.snapshot_participants("k") != {}

    def test_listener_handles_store_method_raising(self, tmp_path):
        """Hook callbacks must never propagate exceptions — even if the
        underlying store method raises, the hook system continues."""
        from gateway.run import _register_participant_cleanup_listener
        from hermes_cli.plugins import invoke_hook

        store = _make_store(tmp_path)

        def boom(session_id):
            raise RuntimeError("disk on fire")
        store.clear_participants_for_session = boom

        _register_participant_cleanup_listener(store)
        # Must not raise
        results = invoke_hook("on_session_finalize", session_id="sess-X", platform="telegram")
        # Listener returns None on error → no result entries
        assert results == []
