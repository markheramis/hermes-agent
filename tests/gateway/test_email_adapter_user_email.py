"""Tests for the email adapter's user_email plumbing.

The email adapter is the only platform where the sender's email IS
the user identifier — no API call, no env-var gating.  Phase 4.1
wires that one-liner into the existing build_source call.

This test pins the wiring at the source level (cheaper than spinning
up an IMAP connection in CI).
"""

import ast
from pathlib import Path


EMAIL_ADAPTER = Path(__file__).parent.parent.parent / "gateway" / "platforms" / "email.py"


class TestEmailAdapterPipesSenderToUserEmail:
    def test_build_source_call_passes_sender_as_user_email(self):
        tree = ast.parse(EMAIL_ADAPTER.read_text(encoding="utf-8"))
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

        assert build_source_calls, "no build_source() call found"
        # At least one call must pass user_email pointing at the sender
        # address (variable name is sender_addr in the handler).
        found = False
        for call in build_source_calls:
            for kw in call.keywords:
                if kw.arg == "user_email":
                    if isinstance(kw.value, ast.Name) and kw.value.id == "sender_addr":
                        found = True
                        break
            if found:
                break
        assert found, (
            "Email adapter's build_source() must pass user_email=sender_addr.  "
            "The sender's address IS the email — losing this wiring means "
            "the agent can't attribute outbound replies or commits."
        )
