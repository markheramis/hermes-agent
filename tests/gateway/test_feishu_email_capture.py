"""Tests for the Feishu adapter's opt-in user_email capture.

Phase 4.2 adds email extraction to ``_resolve_sender_profile``, gated
behind ``FEISHU_CAPTURE_USER_EMAIL`` (mirrors Slack's
``SLACK_CAPTURE_USER_EMAIL`` pattern).

Coverage:
  - Env-var helper parses common truthy/falsy values
  - Email cache reads are gated by the env var even when the cache is populated
  - sender_profile contains user_email when the gate is on
  - sender_profile contains None for user_email when the gate is off
  - build_source call sites pass user_email through
"""

import ast
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.platforms.feishu import _feishu_capture_user_email_enabled


FEISHU_ADAPTER = Path(__file__).parent.parent.parent / "gateway" / "platforms" / "feishu.py"


@pytest.fixture(autouse=True)
def _clean_feishu_email_env():
    os.environ.pop("FEISHU_CAPTURE_USER_EMAIL", None)
    yield
    os.environ.pop("FEISHU_CAPTURE_USER_EMAIL", None)


class TestEnvVarGate:
    def test_default_off(self):
        assert _feishu_capture_user_email_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "True"])
    def test_truthy_values(self, value):
        os.environ["FEISHU_CAPTURE_USER_EMAIL"] = value
        assert _feishu_capture_user_email_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "junk"])
    def test_falsy_values(self, value):
        os.environ["FEISHU_CAPTURE_USER_EMAIL"] = value
        assert _feishu_capture_user_email_enabled() is False


class TestEmailCacheGated:
    """The cache returns email ONLY when the env-gate is on, even if the
    cache holds a value (defends against operator toggling the gate off
    after data was already collected)."""

    def _make_adapter_with_email_in_cache(self):
        from gateway.platforms.feishu import FeishuAdapter
        adapter = object.__new__(FeishuAdapter)
        adapter._sender_name_cache = {}
        adapter._sender_email_cache = {"open-id-1": ("user@example.com", 9_999_999_999.0)}
        return adapter

    def test_get_cached_returns_email_when_cache_hit(self):
        adapter = self._make_adapter_with_email_in_cache()
        # The _get_cached_sender_email helper itself does NOT gate (caller
        # gates).  This pins the contract.
        assert adapter._get_cached_sender_email("open-id-1") == "user@example.com"

    def test_get_cached_returns_none_for_unknown(self):
        adapter = self._make_adapter_with_email_in_cache()
        assert adapter._get_cached_sender_email("never-seen") is None

    def test_get_cached_returns_none_for_expired(self):
        from gateway.platforms.feishu import FeishuAdapter
        adapter = object.__new__(FeishuAdapter)
        adapter._sender_name_cache = {}
        # expire_at = 0 -> always expired
        adapter._sender_email_cache = {"open-id-1": ("user@example.com", 0.0)}
        assert adapter._get_cached_sender_email("open-id-1") is None
        # Expired entry should also be popped
        assert "open-id-1" not in adapter._sender_email_cache


class TestSenderProfileGatedExposure:
    """``_resolve_sender_profile`` adds user_email only when env var is on,
    even if the cache contains the email."""

    @pytest.mark.asyncio
    async def test_email_suppressed_when_env_off(self):
        from gateway.platforms.feishu import FeishuAdapter
        adapter = object.__new__(FeishuAdapter)
        adapter._sender_name_cache = {"ou_test": ("Alice", 9_999_999_999.0)}
        adapter._sender_email_cache = {"ou_test": ("alice@example.com", 9_999_999_999.0)}
        adapter._client = None

        sender_id = MagicMock(
            open_id="ou_test", user_id=None, union_id=None,
        )

        # Env var is unset (autouse fixture)
        with patch.object(adapter, "_resolve_sender_name_from_api", return_value="Alice"):
            profile = await adapter._resolve_sender_profile(sender_id)

        assert profile["user_email"] is None, (
            "user_email leaked when FEISHU_CAPTURE_USER_EMAIL is off"
        )

    @pytest.mark.asyncio
    async def test_email_surfaced_when_env_on(self):
        from gateway.platforms.feishu import FeishuAdapter
        adapter = object.__new__(FeishuAdapter)
        adapter._sender_name_cache = {"ou_test": ("Alice", 9_999_999_999.0)}
        adapter._sender_email_cache = {"ou_test": ("alice@example.com", 9_999_999_999.0)}
        adapter._client = None

        sender_id = MagicMock(
            open_id="ou_test", user_id=None, union_id=None,
        )

        os.environ["FEISHU_CAPTURE_USER_EMAIL"] = "1"
        with patch.object(adapter, "_resolve_sender_name_from_api", return_value="Alice"):
            profile = await adapter._resolve_sender_profile(sender_id)

        assert profile["user_email"] == "alice@example.com"

    @pytest.mark.asyncio
    async def test_email_none_for_bots_even_when_env_on(self):
        from gateway.platforms.feishu import FeishuAdapter
        adapter = object.__new__(FeishuAdapter)
        adapter._sender_name_cache = {}
        adapter._sender_email_cache = {}
        adapter._client = None

        sender_id = MagicMock(
            open_id="ou_bot", user_id=None, union_id=None,
        )

        os.environ["FEISHU_CAPTURE_USER_EMAIL"] = "1"
        with patch.object(adapter, "_resolve_sender_name_from_api", return_value="BotName"):
            profile = await adapter._resolve_sender_profile(sender_id, is_bot=True)

        assert profile["user_email"] is None, (
            "Bot messages should never carry user_email; bots have no human identity"
        )


class TestBuildSourceCallsPassEmail:
    """All three build_source call sites in feishu.py must pass user_email."""

    def test_all_call_sites_pass_user_email(self):
        tree = ast.parse(FEISHU_ADAPTER.read_text(encoding="utf-8"))
        build_source_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "build_source"
            ):
                build_source_calls.append(node)

        assert len(build_source_calls) >= 3, (
            f"Expected ≥3 build_source() calls in feishu.py; found {len(build_source_calls)}"
        )
        for i, call in enumerate(build_source_calls):
            kw_names = {kw.arg for kw in call.keywords}
            assert "user_email" in kw_names, (
                f"build_source() call #{i+1} in feishu.py does not pass "
                f"user_email — the Feishu email capture is broken at this site"
            )
