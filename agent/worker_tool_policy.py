"""In-process, fail-closed tool policy for delegated workers.

This policy narrows accidental model/tool routing. It is not Host authority or
OS isolation; deployments that need containment must use the configured
whole-process or terminal-backend isolation posture described in SECURITY.md.
"""

from __future__ import annotations

import threading
import time
import weakref
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from math import isfinite
from secrets import token_urlsafe
from typing import Iterable


_AUTHORITY = "hermes-in-process-dev-only"
_MAX_TTL_SECONDS = 24 * 60 * 60
_MAX_TASK_ID_LENGTH = 512
_MAX_PRINCIPAL_ID_LENGTH = 512
_MAX_TOOL_NAME_LENGTH = 256
_MAX_RESOURCE_LENGTH = 4096
_MAX_TOOL_NAMES = 256
_MAX_RESOURCES = 128
_pending_lock = threading.Lock()
_pending_nonces: dict[str, float] = {}
_active_owners: dict[str, tuple[weakref.ReferenceType, float]] = {}
_current_policy: ContextVar[object | None] = ContextVar("hermes_worker_tool_policy", default=None)


@dataclass(frozen=True, slots=True)
class WorkerToolAuthorization:
    """Immutable claim minted for one delegated task inside this process."""

    task_id: str
    principal_id: str
    permitted_tools: frozenset[str]
    permitted_resources: frozenset[str]
    expires_at: float
    nonce: str
    authority: str = _AUTHORITY


def _valid_names(values, *, limit: int, max_length: int) -> frozenset[str] | None:
    # Require a finite, sized container. Arbitrary iterables can be unbounded
    # and mappings are not permission lists.
    if type(values) not in (list, tuple, set, frozenset) or len(values) > limit:
        return None
    items = tuple(values)
    if any(
        not isinstance(value, str) or not value.strip() or len(value) > max_length
        for value in items
    ):
        return None
    return frozenset(items)


def issue_worker_tool_authorization(
    *, task_id: str, principal_id: str, permitted_tools: Iterable[str],
    permitted_resources: Iterable[str], ttl_seconds: float,
) -> WorkerToolAuthorization | None:
    """Mint a one-use in-process claim; invalid or unbounded inputs fail closed."""
    tools = _valid_names(permitted_tools, limit=_MAX_TOOL_NAMES, max_length=_MAX_TOOL_NAME_LENGTH)
    resources = _valid_names(permitted_resources, limit=_MAX_RESOURCES, max_length=_MAX_RESOURCE_LENGTH)
    if (
        not isinstance(task_id, str) or not task_id.strip() or len(task_id) > _MAX_TASK_ID_LENGTH
        or not isinstance(principal_id, str) or not principal_id.strip()
        or len(principal_id) > _MAX_PRINCIPAL_ID_LENGTH
        or tools is None or not tools or resources is None
        or isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float))
        or not 0 < ttl_seconds <= _MAX_TTL_SECONDS
    ):
        return None
    nonce = token_urlsafe(24)
    claim = WorkerToolAuthorization(
        task_id=task_id,
        principal_id=principal_id,
        permitted_tools=tools,
        permitted_resources=resources,
        expires_at=time.monotonic() + float(ttl_seconds),
        nonce=nonce,
    )
    with _pending_lock:
        now = time.monotonic()
        for expired_nonce in [key for key, expiry in _pending_nonces.items() if expiry <= now]:
            _pending_nonces.pop(expired_nonce, None)
        for expired_nonce, (_owner_ref, expiry) in list(_active_owners.items()):
            if expiry <= now:
                _active_owners.pop(expired_nonce, None)
        _pending_nonces[nonce] = claim.expires_at
    return claim


def _discard_active_owner(nonce: str, reference: weakref.ReferenceType) -> None:
    with _pending_lock:
        current = _active_owners.get(nonce)
        if current is not None and current[0] is reference:
            _active_owners.pop(nonce, None)


def attach_worker_tool_authorization(
    agent, claim: object, *, task_id: object, principal_id: object, permitted_resources: object,
) -> bool:
    """Consume a valid claim once and bind it to exactly one live agent object."""
    if type(claim) is not WorkerToolAuthorization:
        return False
    if (
        claim.authority != _AUTHORITY
        or claim.task_id != task_id
        or claim.principal_id != principal_id
        or type(permitted_resources) is not frozenset
        or claim.permitted_resources != permitted_resources
        or not isinstance(claim.task_id, str) or not claim.task_id.strip()
        or len(claim.task_id) > _MAX_TASK_ID_LENGTH
        or not isinstance(claim.principal_id, str) or not claim.principal_id.strip()
        or len(claim.principal_id) > _MAX_PRINCIPAL_ID_LENGTH
        or type(claim.permitted_tools) is not frozenset
        or type(claim.permitted_resources) is not frozenset
        or not claim.permitted_tools
        or len(claim.permitted_tools) > _MAX_TOOL_NAMES
        or len(claim.permitted_resources) > _MAX_RESOURCES
        or any(
            not isinstance(name, str) or not name.strip() or len(name) > _MAX_TOOL_NAME_LENGTH
            for name in claim.permitted_tools
        )
        or any(
            not isinstance(name, str) or not name.strip() or len(name) > _MAX_RESOURCE_LENGTH
            for name in claim.permitted_resources
        )
        or not isinstance(claim.expires_at, (int, float))
        or isinstance(claim.expires_at, bool)
        or not isfinite(claim.expires_at)
        or claim.expires_at <= time.monotonic()
        or not isinstance(claim.nonce, str) or not claim.nonce or len(claim.nonce) > 256
    ):
        return False
    try:
        owner_ref = weakref.ref(agent, lambda reference, nonce=claim.nonce: _discard_active_owner(nonce, reference))
    except TypeError:
        return False
    with _pending_lock:
        pending_expiry = _pending_nonces.get(claim.nonce)
        if pending_expiry is None or pending_expiry <= time.monotonic():
            _pending_nonces.pop(claim.nonce, None)
            return False
        _pending_nonces.pop(claim.nonce, None)
        _active_owners[claim.nonce] = (owner_ref, claim.expires_at)
    agent.worker_tool_authorization = claim
    agent.worker_tool_allowlist = claim.permitted_tools
    return True


def active_worker_tool_allowlist(agent) -> frozenset[str]:
    """Return the live worker grant or an empty set (DEFAULT DENY)."""
    claim = getattr(agent, "worker_tool_authorization", None)
    if type(claim) is not WorkerToolAuthorization:
        return frozenset()
    with _pending_lock:
        owner = _active_owners.get(claim.nonce)
        if owner is None:
            return frozenset()
        if owner[1] <= time.monotonic():
            _active_owners.pop(claim.nonce, None)
            return frozenset()
        if owner[0]() is not agent:
            return frozenset()
    if (
        claim.authority != _AUTHORITY
        or not isinstance(claim.task_id, str) or not claim.task_id.strip()
        or len(claim.task_id) > _MAX_TASK_ID_LENGTH
        or not isinstance(claim.principal_id, str) or not claim.principal_id.strip()
        or len(claim.principal_id) > _MAX_PRINCIPAL_ID_LENGTH
        or not isinstance(claim.expires_at, (int, float))
        or isinstance(claim.expires_at, bool)
        or not isfinite(claim.expires_at)
        or claim.expires_at <= time.monotonic()
        or getattr(agent, "worker_task_id", None) != claim.task_id
        or getattr(agent, "worker_principal_id", None) != claim.principal_id
        or getattr(agent, "worker_resource_scope", None) != claim.permitted_resources
        or getattr(agent, "worker_tool_allowlist", None) != claim.permitted_tools
    ):
        return frozenset()
    return claim.permitted_tools


def filter_worker_tool_definitions(agent, definitions: list[dict]) -> tuple[list[dict], set[str]]:
    """Filter one tool-schema snapshot through the immutable worker grant."""
    if getattr(agent, "platform", None) != "subagent":
        filtered = list(definitions)
    else:
        allowed = active_worker_tool_allowlist(agent)
        filtered = [
            item for item in definitions
            if isinstance(item, dict)
            and isinstance(item.get("function"), dict)
            and item["function"].get("name") in allowed
        ]
    names = {
        item["function"]["name"] for item in filtered
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
        and isinstance(item["function"].get("name"), str)
    }
    return filtered, names


def worker_tool_is_allowed(tool_name: object, claim: object = None) -> bool:
    """Check a final dispatch against the active context policy; no policy means normal agent."""
    current = _current_policy.get() if claim is None else claim
    if current is None:
        return True
    return (
        type(current) is WorkerToolAuthorization
        and current.authority == _AUTHORITY
        and isinstance(current.expires_at, (int, float))
        and not isinstance(current.expires_at, bool)
        and isfinite(current.expires_at)
        and current.expires_at > time.monotonic()
        and isinstance(tool_name, str)
        and tool_name in current.permitted_tools
    )


@contextmanager
def worker_tool_policy_scope(agent):
    """Bind the worker grant across registry, deferred, connector and nested calls."""
    claim = getattr(agent, "worker_tool_authorization", None)
    if getattr(agent, "platform", None) != "subagent":
        yield
        return
    if not active_worker_tool_allowlist(agent) and getattr(claim, "permitted_tools", None):
        # Keep an expired, malformed, or replayed claim from inheriting an outer policy.
        claim = object()
    token: Token = _current_policy.set(claim if type(claim) is WorkerToolAuthorization else object())
    try:
        yield
    finally:
        _current_policy.reset(token)


def current_worker_tool_policy() -> object | None:
    """Current task's policy for nested dispatch sites."""
    return _current_policy.get()
