"""Tests for the participant rendering in the system prompt.

After Phase 2 the prompt should:
  - Keep the single-line ``Current session: <name> [<email>]`` block
    (always useful when the LLM addresses the user)
  - When >1 participants are tracked, emit a one-line *hint* pointing
    to the ``get_session_participants`` tool — NOT the full roster
  - When 0 or 1 participants, emit no hint at all
  - Never embed per-user `<email>` lines in the prompt (Phase 2
    regression guard against the in-branch behaviour)
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent.system_prompt import build_system_prompt_parts


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
        _user_name=None,
        _user_email=None,
        _participants={},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _prompt_text(agent):
    """Render the prompt and concat all parts into one string for grepping."""
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        parts = build_system_prompt_parts(agent)
    return "\n".join(parts.values())


class TestCurrentSessionLine:
    """One-line `Current session:` block is preserved."""

    def test_name_only(self):
        text = _prompt_text(_make_agent(_user_name="Mark"))
        assert "Current session: Mark" in text
        assert "<" not in text.split("Current session:", 1)[1].split("\n", 1)[0]

    def test_name_and_email(self):
        text = _prompt_text(_make_agent(_user_name="Mark", _user_email="m@x.com"))
        assert "Current session: Mark <m@x.com>" in text

    def test_no_line_without_name(self):
        text = _prompt_text(_make_agent())
        assert "Current session:" not in text


class TestParticipantHint:
    """The hint replaces the old per-user roster block."""

    def test_no_hint_when_zero_participants(self):
        text = _prompt_text(_make_agent(_participants={}))
        assert "get_session_participants" not in text
        assert "Session has" not in text

    def test_no_hint_when_one_participant(self):
        text = _prompt_text(_make_agent(
            _participants={"u1": {"name": "Mark", "email": "", "message_count": 1, "first_seen": ""}}
        ))
        assert "get_session_participants" not in text
        assert "Session has" not in text

    def test_hint_when_two_participants(self):
        text = _prompt_text(_make_agent(_participants={
            "u1": {"name": "Mark", "email": "", "message_count": 5, "first_seen": ""},
            "u2": {"name": "Alice", "email": "", "message_count": 3, "first_seen": ""},
        }))
        assert "Session has 2 participants" in text
        assert "get_session_participants" in text

    def test_hint_when_many_participants(self):
        roster = {
            f"u{i}": {"name": f"User{i}", "email": "", "message_count": i + 1, "first_seen": ""}
            for i in range(20)
        }
        text = _prompt_text(_make_agent(_participants=roster))
        assert "Session has 20 participants" in text

    def test_hint_is_a_single_line(self):
        text = _prompt_text(_make_agent(_participants={
            "u1": {"name": "Mark", "email": "", "message_count": 5, "first_seen": ""},
            "u2": {"name": "Alice", "email": "", "message_count": 3, "first_seen": ""},
        }))
        # Find the hint line and confirm it doesn't bleed into multi-line.
        for line in text.splitlines():
            if "Session has" in line:
                assert "get_session_participants" in line
                return
        raise AssertionError("hint line not found")


class TestNoRosterLeak:
    """Regression guard: the old per-user roster block must not return."""

    def test_no_per_user_lines_with_email(self):
        text = _prompt_text(_make_agent(_participants={
            "u1": {"name": "Mark", "email": "mark@x.com", "message_count": 5, "first_seen": ""},
            "u2": {"name": "Alice", "email": "alice@x.com", "message_count": 3, "first_seen": ""},
        }))
        # PII must not be embedded — that's exactly what we moved to the tool
        assert "mark@x.com" not in text
        assert "alice@x.com" not in text
        # The old roster pattern was "  {name} <{email}> — {count} msgs"
        assert "msgs" not in text
        assert "<mark@x.com>" not in text

    def test_no_per_user_lines_without_email(self):
        text = _prompt_text(_make_agent(_participants={
            "u1": {"name": "Mark", "email": "", "message_count": 5, "first_seen": ""},
            "u2": {"name": "Alice", "email": "", "message_count": 3, "first_seen": ""},
        }))
        # Old format was "  {name} — {count} msgs"
        assert "msgs" not in text
        assert "Session participants" not in text  # old section header
