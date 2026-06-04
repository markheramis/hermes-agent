"""On-demand session participant roster for the agent.

Reads ``HERMES_SESSION_PARTICIPANTS`` (populated by the gateway from
``SessionStore.snapshot_participants``); returns an empty roster in
CLI/DM/cron paths where the var isn't set.
"""

import json
import logging
from typing import Any, Dict, Optional

from gateway.session_context import get_session_env

logger = logging.getLogger(__name__)


GET_SESSION_PARTICIPANTS_SCHEMA = {
    "name": "get_session_participants",
    "description": (
        "Return the roster of users who have spoken in the current "
        "conversation session.  Use this when you need to credit "
        "contributors (commit messages, acknowledgments), look up a "
        "participant's email for outreach, or otherwise see who else "
        "is part of this conversation.\n\n"
        "Returns a JSON object with:\n"
        "  - ``participants``: list of {name, email, message_count, first_seen} "
        "objects, sorted by message_count descending\n"
        "  - ``count``: total participants tracked\n"
        "  - ``note`` (optional): present only when the roster is empty, "
        "explaining why (DM, CLI session, or first-message-in-channel)\n\n"
        "Reactive-only: a participant appears here only after they have "
        "sent at least one message in this session.  Silent observers in "
        "the channel are NOT tracked.  Roster lifetime equals session "
        "lifetime — data is wiped on session reset, idle expiry, and "
        "shutdown."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


def _parse_participants_envvar() -> Dict[str, Dict[str, Any]]:
    """Read the roster from the ContextVar / env-bridge value.

    May arrive as a dict (in-process) or JSON string (subprocess bridge
    via ``tools/environments/local.py``). Decode errors degrade silently.
    """
    raw = get_session_env("HERMES_SESSION_PARTICIPANTS", "")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        decoded = json.loads(raw)
        return decoded if isinstance(decoded, dict) else {}
    except (ValueError, TypeError) as exc:
        logger.debug("Failed to decode HERMES_SESSION_PARTICIPANTS: %s", exc)
        return {}


def get_session_participants_tool(
    args: Optional[Dict[str, Any]] = None,
    **_kwargs: Any,
) -> str:
    """Handler — returns roster as JSON string."""
    try:
        roster = _parse_participants_envvar()
    except Exception as exc:
        logger.debug("get_session_participants: read failed: %s", exc)
        return json.dumps({
            "participants": [],
            "count": 0,
            "error": "Failed to read session roster.",
        }, ensure_ascii=False)

    if not roster:
        return json.dumps({
            "participants": [],
            "count": 0,
            "note": (
                "No participants tracked — this is likely a DM, a CLI "
                "session, or no one has spoken yet."
            ),
        }, ensure_ascii=False)

    items = []
    for user_id, info in roster.items():
        if not isinstance(info, dict):
            continue
        # Don't fall back to the raw platform user_id (e.g. Slack ``Uxxxxxxx``)
        # — surfacing it as ``name`` would let the agent quote opaque IDs as
        # if they were human-readable names. Mark it explicitly instead.
        raw_name = info.get("name", "")
        name = str(raw_name).strip() if raw_name else ""
        items.append({
            "name": name or "(unknown)",
            "email": str(info.get("email", "") or ""),
            "message_count": int(info.get("message_count", 0) or 0),
            "first_seen": str(info.get("first_seen", "") or ""),
        })
    # Most-active contributors lead.
    items.sort(key=lambda p: -p["message_count"])

    return json.dumps({
        "participants": items,
        "count": len(items),
    }, ensure_ascii=False)


def check_session_participants_requirements() -> bool:
    """Always available; returns an empty roster when the ContextVar is unset."""
    return True


# --- Registry ---
from tools.registry import registry

registry.register(
    name="get_session_participants",
    toolset="session",
    schema=GET_SESSION_PARTICIPANTS_SCHEMA,
    handler=lambda args, **kw: get_session_participants_tool(args, **kw),
    check_fn=check_session_participants_requirements,
    emoji="👥",
)
