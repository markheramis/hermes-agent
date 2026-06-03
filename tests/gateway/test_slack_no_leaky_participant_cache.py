"""Regression guard: Slack's per-channel participant cache stays removed.

The original in-branch implementation stored participants in
``SlackAdapter._channel_participants`` — a per-instance dict that was
never cleared, leaking PII across sessions for the process lifetime.

Phase 1.4 moved tracking to ``SessionStore.upsert_participant`` (session-
scoped, cleared on reset/expiry/shutdown).  This test pins the removal:
any reintroduction of an adapter-local participant cache fails CI.

Approach: AST inspection of ``gateway/platforms/slack.py``.  Cheaper and
less brittle than running the adapter against a real Slack workspace.
"""

import ast
from pathlib import Path


SLACK_ADAPTER = Path(__file__).parent.parent.parent / "gateway" / "platforms" / "slack.py"


def _parse_slack():
    return ast.parse(SLACK_ADAPTER.read_text(encoding="utf-8"))


class TestNoLocalParticipantCache:
    def test_channel_participants_attr_not_assigned(self):
        """``self._channel_participants = ...`` must not appear anywhere."""
        tree = _parse_slack()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == "_channel_participants"
                ):
                    raise AssertionError(
                        "SlackAdapter._channel_participants reintroduced.  "
                        "Per-channel participant tracking belongs in "
                        "SessionStore (session-scoped + cleaned up), not "
                        "on the adapter (process-scoped + leaky)."
                    )
        # AnnAssign too (type-annotated form)
        for node in ast.walk(tree):
            if not isinstance(node, ast.AnnAssign):
                continue
            target = node.target
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "_channel_participants"
            ):
                raise AssertionError(
                    "SlackAdapter._channel_participants reintroduced "
                    "(annotated form)."
                )

    def test_upsert_participant_method_removed(self):
        """The local upsert helper must not return — that's the store's job."""
        tree = _parse_slack()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_upsert_participant":
                raise AssertionError(
                    "SlackAdapter._upsert_participant reintroduced.  "
                    "Use SessionStore.upsert_participant instead — it is "
                    "session-scoped and lock-protected."
                )

    def test_snapshot_participants_method_removed(self):
        tree = _parse_slack()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_snapshot_participants":
                raise AssertionError(
                    "SlackAdapter._snapshot_participants reintroduced.  "
                    "Use SessionStore.snapshot_participants(session_key)."
                )

    def test_build_source_calls_do_not_pass_participants(self):
        """No call to ``self.build_source(..., participants=...)`` in slack.py.
        The roster is populated by the gateway from the store snapshot.
        Slack just needs to provide user_name / user_email; the rest is
        the gateway's job."""
        tree = _parse_slack()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "build_source"
            ):
                continue
            for kw in node.keywords:
                if kw.arg == "participants":
                    raise AssertionError(
                        "build_source() call in slack.py passes "
                        "participants= — this should come from the "
                        "SessionStore, not from the Slack adapter."
                    )


class TestUserCachesPreserved:
    """The per-user (not per-channel) name/email caches stay — they're
    optimization caches scoped to the workspace, not session data."""

    def test_user_name_cache_still_present(self):
        src = SLACK_ADAPTER.read_text(encoding="utf-8")
        assert "_user_name_cache" in src, (
            "Per-user name cache should be preserved — it's an "
            "API-call optimization, not session-scoped PII."
        )

    def test_user_email_cache_still_present(self):
        src = SLACK_ADAPTER.read_text(encoding="utf-8")
        assert "_user_email_cache" in src
