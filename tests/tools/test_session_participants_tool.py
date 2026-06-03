"""Tests for ``get_session_participants`` tool.

Covers:
  - Tool is registered in the global registry with the right shape
  - Empty roster path (ContextVar / env var unset)
  - Populated path (ContextVar set as dict)
  - Bridge path (ContextVar set as JSON string — subprocess transport)
  - Malformed JSON degrades gracefully (returns empty, not error)
  - Sort order is message_count descending
  - Sanitization is preserved through the boundary (we render whatever
    the store gave us — sanitization happens at write time, not read)
"""

import json
import os

import pytest

from tools.session_participants_tool import (
    GET_SESSION_PARTICIPANTS_SCHEMA,
    get_session_participants_tool,
)


@pytest.fixture(autouse=True)
def _clean_env():
    """Ensure HERMES_SESSION_PARTICIPANTS is unset before/after each test."""
    os.environ.pop("HERMES_SESSION_PARTICIPANTS", None)
    yield
    os.environ.pop("HERMES_SESSION_PARTICIPANTS", None)


class TestRegistration:
    def test_tool_registered(self):
        from tools.registry import registry
        entry = registry.get_entry("get_session_participants")
        assert entry is not None
        assert entry.schema["name"] == "get_session_participants"
        assert entry.toolset == "session"

    def test_schema_takes_no_args(self):
        params = GET_SESSION_PARTICIPANTS_SCHEMA["parameters"]
        assert params["type"] == "object"
        assert params["properties"] == {}
        assert params.get("additionalProperties") is False

    def test_check_fn_returns_true(self):
        from tools.session_participants_tool import (
            check_session_participants_requirements,
        )
        assert check_session_participants_requirements() is True


class TestEmptyPath:
    def test_no_env_var_returns_empty(self):
        out = json.loads(get_session_participants_tool({}))
        assert out["count"] == 0
        assert out["participants"] == []
        assert "note" in out  # explanation present

    def test_empty_string_env_var_returns_empty(self):
        os.environ["HERMES_SESSION_PARTICIPANTS"] = ""
        out = json.loads(get_session_participants_tool({}))
        assert out["count"] == 0

    def test_empty_json_object_returns_empty(self):
        os.environ["HERMES_SESSION_PARTICIPANTS"] = "{}"
        out = json.loads(get_session_participants_tool({}))
        assert out["count"] == 0


class TestPopulatedPath:
    def test_decodes_json_string(self):
        os.environ["HERMES_SESSION_PARTICIPANTS"] = json.dumps({
            "u1": {"name": "Alice", "email": "alice@x.com",
                   "message_count": 5, "first_seen": "2026-06-03T12:00:00"},
        })
        out = json.loads(get_session_participants_tool({}))
        assert out["count"] == 1
        assert out["participants"][0]["name"] == "Alice"
        assert out["participants"][0]["email"] == "alice@x.com"
        assert out["participants"][0]["message_count"] == 5

    def test_sort_order_message_count_desc(self):
        os.environ["HERMES_SESSION_PARTICIPANTS"] = json.dumps({
            "u1": {"name": "Alice", "email": "", "message_count": 1, "first_seen": ""},
            "u2": {"name": "Bob",   "email": "", "message_count": 5, "first_seen": ""},
            "u3": {"name": "Carol", "email": "", "message_count": 3, "first_seen": ""},
        })
        out = json.loads(get_session_participants_tool({}))
        names = [p["name"] for p in out["participants"]]
        assert names == ["Bob", "Carol", "Alice"]

    def test_user_id_falls_back_to_name_when_missing(self):
        os.environ["HERMES_SESSION_PARTICIPANTS"] = json.dumps({
            "u-anon": {"email": "", "message_count": 1, "first_seen": ""},
        })
        out = json.loads(get_session_participants_tool({}))
        assert out["participants"][0]["name"] == "u-anon"

    def test_no_note_when_populated(self):
        os.environ["HERMES_SESSION_PARTICIPANTS"] = json.dumps({
            "u1": {"name": "Alice", "email": "", "message_count": 1, "first_seen": ""},
        })
        out = json.loads(get_session_participants_tool({}))
        assert "note" not in out


class TestRobustness:
    def test_malformed_json_returns_empty(self):
        os.environ["HERMES_SESSION_PARTICIPANTS"] = "{this is not json"
        out = json.loads(get_session_participants_tool({}))
        assert out["count"] == 0
        # Doesn't surface a confusing error to the LLM
        assert "participants" in out

    def test_non_dict_json_returns_empty(self):
        os.environ["HERMES_SESSION_PARTICIPANTS"] = "[1, 2, 3]"
        out = json.loads(get_session_participants_tool({}))
        assert out["count"] == 0

    def test_skips_non_dict_entries(self):
        # If a value is malformed (not a dict), it's filtered out, not surfaced
        os.environ["HERMES_SESSION_PARTICIPANTS"] = json.dumps({
            "u1": {"name": "Alice", "email": "", "message_count": 1, "first_seen": ""},
            "u2": "not a dict",
            "u3": None,
        })
        out = json.loads(get_session_participants_tool({}))
        assert out["count"] == 1
        assert out["participants"][0]["name"] == "Alice"

    def test_handler_accepts_extra_kwargs(self):
        """Defensive — registry dispatch may pass extra kwargs we don't use."""
        out = json.loads(get_session_participants_tool({}, callback=None, extra="ignored"))
        assert out["count"] == 0


class TestSanitizationPreserved:
    """The tool itself does not sanitize — that happens at upsert time.
    But we should not RE-introduce hazardous chars on the rendering boundary."""

    def test_output_does_not_introduce_unsafe_chars(self):
        # Input is already sanitized (e.g., 'Alice' not 'Alice\x00').
        # We pin that the output JSON round-trips it cleanly.
        os.environ["HERMES_SESSION_PARTICIPANTS"] = json.dumps({
            "u1": {"name": "Alice", "email": "alice@x.com",
                   "message_count": 1, "first_seen": "2026-06-03T12:00:00"},
        })
        out = json.loads(get_session_participants_tool({}))
        for p in out["participants"]:
            assert "\x00" not in p["name"]
            assert "\x00" not in p["email"]
