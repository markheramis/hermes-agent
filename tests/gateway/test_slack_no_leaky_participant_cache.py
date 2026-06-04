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


class TestEmailCacheWriteIsGated:
    """``self._user_email_cache[user_id] = email`` writes must be gated
    on ``SLACK_CAPTURE_USER_EMAIL`` (via the ``_slack_capture_user_email_enabled``
    helper).  Otherwise the email opt-in flag only controls the read path
    while emails accumulate in memory regardless of the user's setting —
    a privacy bug we already shipped once.
    """

    @staticmethod
    def _if_test_mentions_gate(test_node: ast.AST) -> bool:
        """Return True if an ``if`` test ultimately checks the opt-in gate."""
        for sub in ast.walk(test_node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                if sub.func.id == "_slack_capture_user_email_enabled":
                    return True
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                if sub.value == "SLACK_CAPTURE_USER_EMAIL":
                    return True
        return False

    def _email_cache_writes(self, func: ast.FunctionDef):
        """Yield (assign_node, ancestors) for every ``self._user_email_cache[...] = ...``
        inside *func*.  ``ancestors`` is the path from func to assign,
        exclusive of both ends."""
        # Build child→parent map for the subtree
        parents = {}
        for node in ast.walk(func):
            for child in ast.iter_child_nodes(node):
                parents[id(child)] = node

        for node in ast.walk(func):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Subscript):
                    continue
                val = target.value
                if not (
                    isinstance(val, ast.Attribute)
                    and isinstance(val.value, ast.Name)
                    and val.value.id == "self"
                    and val.attr == "_user_email_cache"
                ):
                    continue
                # Walk up to func collecting ancestors
                ancestors = []
                cur = parents.get(id(node))
                while cur is not None and cur is not func:
                    ancestors.append(cur)
                    cur = parents.get(id(cur))
                yield node, ancestors

    def test_email_cache_write_in_resolve_user_name_is_gated(self):
        tree = _parse_slack()
        target_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_resolve_user_name":
                target_func = node
                break
            if isinstance(node, ast.FunctionDef) and node.name == "_resolve_user_name":
                target_func = node
                break
        assert target_func is not None, "_resolve_user_name not found in slack.py"

        # Pretend FunctionDef so ast.walk + iter_child_nodes work uniformly
        for assign, ancestors in self._email_cache_writes(target_func):
            gated = any(
                isinstance(a, ast.If) and self._if_test_mentions_gate(a.test)
                for a in ancestors
            )
            assert gated, (
                f"_user_email_cache write at line {assign.lineno} in "
                f"_resolve_user_name is NOT inside a "
                f"SLACK_CAPTURE_USER_EMAIL / _slack_capture_user_email_enabled "
                f"guard.  Email collection must be opt-in at BOTH read AND "
                f"write — otherwise flipping the env var off leaves cached "
                f"PII in memory for the process lifetime."
            )

    def test_gate_helper_exists(self):
        src = SLACK_ADAPTER.read_text(encoding="utf-8")
        assert "_slack_capture_user_email_enabled" in src, (
            "Expected a module-level _slack_capture_user_email_enabled() "
            "helper that mirrors Feishu's opt-in pattern."
        )
