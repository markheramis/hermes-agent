"""Tests for SessionStore participant tracking (security-critical).

Covers the storage layer added by the
``feature/gateway-session-identity-context`` branch:

  - SessionEntry.participants field exists, defaults to empty dict
  - SessionStore.upsert_participant / snapshot_participants /
    clear_participants_for_session work correctly
  - Sanitization is enforced at the write boundary
  - FIFO eviction at MAX_PARTICIPANTS_PER_SESSION
  - Snapshot returns a deep copy (caller mutations don't leak back)
  - clear_participants_for_session is keyed by session_id (race-safe)
  - Drift-prevention: every SessionEntry field is classified as either
    persistent or transient
  - PII never leaks to sessions.json (to_dict, _save round-trip)
  - PII never returns via from_dict, even if injected into the JSON
  - Concurrent upserts don't corrupt state

The cleanup-on-finalize / cleanup-on-reset tests live in
test_session_participants_cleanup.py once Phase 1.3 wires those paths.
"""

import dataclasses
import json
import threading
from datetime import datetime
from unittest.mock import patch

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import (
    MAX_PARTICIPANTS_PER_SESSION,
    SESSION_ENTRY_PERSISTENT_FIELDS,
    SESSION_ENTRY_TRANSIENT_FIELDS,
    SessionEntry,
    SessionStore,
)


def _make_store(tmp_path):
    """Build a SessionStore bypassing SQLite/disk-load side effects."""
    config = GatewayConfig(
        default_reset_policy=SessionResetPolicy(mode="none"),
    )
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db = None
    store._loaded = True
    return store


def _add_entry(store, session_key="k1", session_id="sess-1"):
    """Insert a minimal SessionEntry directly into the store."""
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
# Drift-prevention — the security guardrail
# ---------------------------------------------------------------------------

class TestDriftPrevention:
    """Every SessionEntry field must be explicitly classified.

    The disk-persistence allowlist in SessionEntry.to_dict() is the boundary
    between in-memory PII and durable storage.  Without this guardrail a
    future PR could add a sensitive field and accidentally serialize it.
    """

    def test_every_field_classified(self):
        all_fields = {f.name for f in dataclasses.fields(SessionEntry)}
        classified = (
            SESSION_ENTRY_PERSISTENT_FIELDS | SESSION_ENTRY_TRANSIENT_FIELDS
        )
        unclassified = all_fields - classified
        assert not unclassified, (
            f"Unclassified SessionEntry fields: {sorted(unclassified)}.  "
            "Each new field MUST be added to either "
            "SESSION_ENTRY_PERSISTENT_FIELDS (round-tripped to disk) or "
            "SESSION_ENTRY_TRANSIENT_FIELDS (in-memory only)."
        )

    def test_no_stale_classifications(self):
        all_fields = {f.name for f in dataclasses.fields(SessionEntry)}
        stale_persistent = SESSION_ENTRY_PERSISTENT_FIELDS - all_fields
        stale_transient = SESSION_ENTRY_TRANSIENT_FIELDS - all_fields
        assert not stale_persistent, (
            f"Persistent allowlist mentions removed fields: {sorted(stale_persistent)}"
        )
        assert not stale_transient, (
            f"Transient list mentions removed fields: {sorted(stale_transient)}"
        )

    def test_persistent_and_transient_are_disjoint(self):
        overlap = SESSION_ENTRY_PERSISTENT_FIELDS & SESSION_ENTRY_TRANSIENT_FIELDS
        assert not overlap, f"Field cannot be both persistent and transient: {overlap}"

    def test_participants_is_transient(self):
        assert "participants" in SESSION_ENTRY_TRANSIENT_FIELDS
        assert "participants" not in SESSION_ENTRY_PERSISTENT_FIELDS


# ---------------------------------------------------------------------------
# SessionEntry default state
# ---------------------------------------------------------------------------

class TestEntryDefaults:
    def test_new_entry_has_empty_participants(self):
        e = SessionEntry(
            session_key="k", session_id="s",
            created_at=datetime.now(), updated_at=datetime.now(),
        )
        assert e.participants == {}

    def test_two_entries_have_independent_dicts(self):
        # default_factory=dict (not a shared default!) — pin this so a
        # future refactor doesn't accidentally introduce shared state.
        e1 = SessionEntry(
            session_key="k1", session_id="s1",
            created_at=datetime.now(), updated_at=datetime.now(),
        )
        e2 = SessionEntry(
            session_key="k2", session_id="s2",
            created_at=datetime.now(), updated_at=datetime.now(),
        )
        e1.participants["u"] = {"name": "x"}
        assert e2.participants == {}


# ---------------------------------------------------------------------------
# upsert_participant
# ---------------------------------------------------------------------------

class TestUpsert:
    def test_insert_new_participant(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store)
        store.upsert_participant("k1", "u1", "Alice", "alice@example.com")
        snap = store.snapshot_participants("k1")
        assert snap["u1"]["name"] == "Alice"
        assert snap["u1"]["email"] == "alice@example.com"
        assert snap["u1"]["message_count"] == 1
        assert "first_seen" in snap["u1"]

    def test_repeat_upsert_increments_count(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store)
        for _ in range(5):
            store.upsert_participant("k1", "u1", "Alice", "")
        assert store.snapshot_participants("k1")["u1"]["message_count"] == 5

    def test_repeat_upsert_preserves_first_seen(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store)
        store.upsert_participant("k1", "u1", "Alice", "")
        first_seen = store.snapshot_participants("k1")["u1"]["first_seen"]
        store.upsert_participant("k1", "u1", "Alice", "")
        assert store.snapshot_participants("k1")["u1"]["first_seen"] == first_seen

    def test_name_change_updates_in_place(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store)
        store.upsert_participant("k1", "u1", "Alice", "")
        store.upsert_participant("k1", "u1", "Alice Smith", "")
        snap = store.snapshot_participants("k1")
        assert snap["u1"]["name"] == "Alice Smith"
        assert snap["u1"]["message_count"] == 2

    def test_email_added_later(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store)
        store.upsert_participant("k1", "u1", "Alice", "")
        store.upsert_participant("k1", "u1", "Alice", "alice@x.com")
        assert store.snapshot_participants("k1")["u1"]["email"] == "alice@x.com"

    def test_multiple_users_tracked(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store)
        store.upsert_participant("k1", "u1", "Alice", "")
        store.upsert_participant("k1", "u2", "Bob", "")
        store.upsert_participant("k1", "u3", "Carol", "")
        snap = store.snapshot_participants("k1")
        assert set(snap.keys()) == {"u1", "u2", "u3"}

    def test_falls_back_to_user_id_when_name_empty(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store)
        store.upsert_participant("k1", "u1", "", "")
        assert store.snapshot_participants("k1")["u1"]["name"] == "u1"

    def test_empty_user_id_is_noop(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store)
        store.upsert_participant("k1", "", "ghost", "")
        assert store.snapshot_participants("k1") == {}

    def test_unknown_session_key_is_noop(self, tmp_path):
        store = _make_store(tmp_path)
        # No entry created.
        store.upsert_participant("missing-key", "u1", "Alice", "")
        # Doesn't raise, doesn't create the entry.
        assert "missing-key" not in store._entries

    def test_sanitization_at_write(self, tmp_path):
        """Names and emails are scrubbed before storage."""
        store = _make_store(tmp_path)
        _add_entry(store)
        # NUL + zero-width + bidi-override + fullwidth
        attacker_name = "Ｅvil\x00name​‮"
        attacker_email = "a\x00b@x.com"
        store.upsert_participant("k1", "u1", attacker_name, attacker_email)
        snap = store.snapshot_participants("k1")
        # NUL, ZW, bidi removed; fullwidth normalized.
        assert "\x00" not in snap["u1"]["name"]
        assert "​" not in snap["u1"]["name"]
        assert "‮" not in snap["u1"]["name"]
        assert snap["u1"]["name"] == "Evilname"
        assert snap["u1"]["email"] == "ab@x.com"


# ---------------------------------------------------------------------------
# FIFO eviction at cap
# ---------------------------------------------------------------------------

class TestFifoEviction:
    def test_cap_enforced(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store)
        for i in range(MAX_PARTICIPANTS_PER_SESSION + 50):
            store.upsert_participant("k1", f"u{i}", f"User{i}", "")
        snap = store.snapshot_participants("k1")
        assert len(snap) == MAX_PARTICIPANTS_PER_SESSION

    def test_oldest_evicted_first(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store)
        # First MAX users
        for i in range(MAX_PARTICIPANTS_PER_SESSION):
            store.upsert_participant("k1", f"early{i}", f"E{i}", "")
        # Push 10 more
        for i in range(10):
            store.upsert_participant("k1", f"late{i}", f"L{i}", "")
        snap = store.snapshot_participants("k1")
        # Earliest 10 are evicted; latest 10 are present
        for i in range(10):
            assert f"early{i}" not in snap, f"oldest 'early{i}' not evicted"
        for i in range(10):
            assert f"late{i}" in snap, f"newest 'late{i}' missing"

    def test_repeat_upsert_does_not_count_against_cap(self, tmp_path):
        """Bumping an existing user shouldn't trigger eviction."""
        store = _make_store(tmp_path)
        _add_entry(store)
        # Fill to cap with distinct users
        for i in range(MAX_PARTICIPANTS_PER_SESSION):
            store.upsert_participant("k1", f"u{i}", f"User{i}", "")
        # Repeatedly upsert the FIRST user — must not evict anyone
        for _ in range(50):
            store.upsert_participant("k1", "u0", "User0", "")
        snap = store.snapshot_participants("k1")
        assert len(snap) == MAX_PARTICIPANTS_PER_SESSION
        assert "u0" in snap
        # All originals still present (none evicted)
        assert all(f"u{i}" in snap for i in range(MAX_PARTICIPANTS_PER_SESSION))
        assert snap["u0"]["message_count"] == 51  # 1 initial + 50 bumps


# ---------------------------------------------------------------------------
# snapshot_participants — deep-copy isolation
# ---------------------------------------------------------------------------

class TestSnapshotIsolation:
    def test_snapshot_is_independent_dict(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store)
        store.upsert_participant("k1", "u1", "Alice", "")
        snap = store.snapshot_participants("k1")
        snap["u1"]["message_count"] = 9999
        snap["evil"] = {"name": "injected"}
        snap2 = store.snapshot_participants("k1")
        assert snap2["u1"]["message_count"] == 1
        assert "evil" not in snap2

    def test_snapshot_of_unknown_key_is_empty(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.snapshot_participants("nope") == {}

    def test_snapshot_of_empty_session_is_empty(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store)
        assert store.snapshot_participants("k1") == {}


# ---------------------------------------------------------------------------
# clear_participants_for_session — race safety via session_id key
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_by_session_id(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store, session_key="k1", session_id="sess-A")
        store.upsert_participant("k1", "u1", "Alice", "")
        store.upsert_participant("k1", "u2", "Bob", "")
        cleared = store.clear_participants_for_session("sess-A")
        assert cleared == 2
        assert store.snapshot_participants("k1") == {}

    def test_clear_wrong_session_id_is_noop(self, tmp_path):
        """Race safety: if the session has rotated to a new session_id,
        a late finalize-clear for the OLD id must NOT wipe the new roster."""
        store = _make_store(tmp_path)
        _add_entry(store, session_key="k1", session_id="sess-NEW")
        store.upsert_participant("k1", "u1", "Alice", "")
        cleared = store.clear_participants_for_session("sess-OLD")
        assert cleared == 0
        assert store.snapshot_participants("k1") != {}, (
            "Clear leaked into a session with a different session_id"
        )

    def test_clear_empty_session_returns_zero(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store, session_key="k1", session_id="sess-X")
        assert store.clear_participants_for_session("sess-X") == 0

    def test_clear_with_empty_session_id_returns_zero(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store, session_key="k1", session_id="sess-X")
        store.upsert_participant("k1", "u1", "Alice", "")
        assert store.clear_participants_for_session("") == 0
        # Roster unchanged
        assert store.snapshot_participants("k1") != {}

    def test_clear_with_none_session_id_returns_zero(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store, session_key="k1", session_id="sess-X")
        store.upsert_participant("k1", "u1", "Alice", "")
        assert store.clear_participants_for_session(None) == 0


# ---------------------------------------------------------------------------
# Persistence exclusion — PII must never reach disk
# ---------------------------------------------------------------------------

class TestPersistenceExclusion:
    def test_to_dict_omits_participants(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store)
        store.upsert_participant("k1", "u1", "Alice", "alice@example.com")
        d = store._entries["k1"].to_dict()
        assert "participants" not in d, (
            f"PII leak: participants appeared in to_dict() output: {sorted(d.keys())}"
        )

    def test_sessions_json_roundtrip_drops_participants(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store)
        store.upsert_participant("k1", "u1", "Alice", "alice@example.com")
        store._save()
        # Inspect the file directly — PII must not be present
        raw = (tmp_path / "sessions.json").read_text(encoding="utf-8")
        assert "alice@example.com" not in raw, "PII leaked to sessions.json"
        assert "Alice" not in raw, "PII leaked to sessions.json"
        # And the structured form has no key
        data = json.loads(raw)
        assert "participants" not in data["k1"]

    def test_from_dict_ignores_injected_participants(self, tmp_path):
        """If a tampered sessions.json contains participants data, the
        loader must drop it on the floor — the allowlist enforces that
        nothing ever round-trips back into memory either."""
        # Build a known-good dict and inject participants
        store = _make_store(tmp_path)
        entry = _add_entry(store)
        raw = entry.to_dict()
        raw["participants"] = {
            "smuggled": {"name": "ghost", "email": "ghost@example.com"}
        }
        reloaded = SessionEntry.from_dict(raw)
        assert reloaded.participants == {}

    def test_participants_not_in_sqlite_schema(self):
        """No table in the SQLite session DB may have a ``participants``
        column.  If a future migration adds one, this test fails and the
        author has to consciously choose to persist PII (and update the
        threat model in tests/gateway/test_session_participants.py)."""
        import sqlite3
        from hermes_state import SessionDB

        db = SessionDB()
        with sqlite3.connect(db.db_path) as conn:
            tables = [row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )]
            offenders = []
            for table in tables:
                cols = [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]
                if "participants" in cols:
                    offenders.append(table)
        assert not offenders, (
            f"SQLite session DB has 'participants' column(s) on table(s): "
            f"{offenders} — PII would land at rest"
        )

    def test_reloading_store_after_save_starts_empty_roster(self, tmp_path):
        # End-to-end: populate, save, instantiate a new store, load,
        # confirm the participant data is gone.
        store1 = _make_store(tmp_path)
        _add_entry(store1)
        store1.upsert_participant("k1", "u1", "Alice", "alice@example.com")
        store1._save()

        # New store on the same directory, real load path
        config = GatewayConfig(
            default_reset_policy=SessionResetPolicy(mode="none"),
        )
        store2 = SessionStore(sessions_dir=tmp_path, config=config)
        store2._db = None
        store2._ensure_loaded()
        assert store2.snapshot_participants("k1") == {}


# ---------------------------------------------------------------------------
# Concurrency — lock discipline holds under contention
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_concurrent_upserts_no_loss_no_crash(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store)
        n_threads = 10
        per_thread = 100
        errors = []

        def worker(tid):
            try:
                for i in range(per_thread):
                    store.upsert_participant("k1", f"t{tid}", f"T{tid}", "")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        snap = store.snapshot_participants("k1")
        # Exactly n_threads unique users; each saw per_thread upserts
        assert len(snap) == n_threads
        for tid in range(n_threads):
            assert snap[f"t{tid}"]["message_count"] == per_thread

    def test_concurrent_clear_and_upsert_no_crash(self, tmp_path):
        store = _make_store(tmp_path)
        _add_entry(store)
        errors = []
        stop_at = 200

        def upserter():
            try:
                for i in range(stop_at):
                    store.upsert_participant("k1", f"u{i}", f"U{i}", "")
            except Exception as e:
                errors.append(e)

        def clearer():
            try:
                for _ in range(stop_at):
                    store.clear_participants_for_session("sess-1")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=upserter),
            threading.Thread(target=clearer),
            threading.Thread(target=upserter),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # No deadlock, no corruption — final state is unsurprising
        # (could be empty or partial, but must be a clean dict)
        snap = store.snapshot_participants("k1")
        assert isinstance(snap, dict)
        for uid, info in snap.items():
            assert {"name", "email", "message_count", "first_seen"} <= set(info.keys())
