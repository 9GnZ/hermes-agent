"""Behavior contracts for delegated Worker tool permissions.

The policy is in-process hardening; it is not a substitute for OS isolation.
"""

from dataclasses import replace
from itertools import count
from types import SimpleNamespace
import time

import pytest

from agent.worker_tool_policy import (
    WorkerToolAuthorization,
    active_worker_tool_allowlist,
    attach_worker_tool_authorization,
    filter_worker_tool_definitions,
    issue_worker_tool_authorization,
    worker_tool_policy_scope,
)
from tools.delegate_tool_toolsets import _resolve_child_tool_permissions


class _Agent:
    pass


def _issue(*, tools=("read_file",), ttl=60):
    return issue_worker_tool_authorization(
        task_id="task-1", principal_id="parent-1", permitted_tools=tools,
        permitted_resources=("workspace:C:/dev",), ttl_seconds=ttl,
    )


def _worker(claim=None):
    agent = _Agent()
    agent.platform = "subagent"
    agent.worker_task_id = "task-1"
    agent.worker_principal_id = "parent-1"
    agent.worker_tool_authorization = None
    agent.worker_tool_allowlist = frozenset()
    agent.worker_resource_scope = claim.permitted_resources if claim is not None else frozenset()
    if claim is not None:
        assert attach_worker_tool_authorization(
            agent, claim, task_id=agent.worker_task_id, principal_id=agent.worker_principal_id,
            permitted_resources=claim.permitted_resources,
        )
    return agent


def _tool(name):
    return {"type": "function", "function": {"name": name, "parameters": {}}}


@pytest.mark.parametrize("parent", [
    SimpleNamespace(enabled_toolsets=None),
    SimpleNamespace(enabled_toolsets=["terminal"]),
    SimpleNamespace(enabled_toolsets=["terminal"], valid_tool_names=[]),
    SimpleNamespace(enabled_toolsets=["terminal"], valid_tool_names={"terminal", None}),
    SimpleNamespace(enabled_toolsets=[], valid_tool_names={"terminal"}),
])
def test_missing_empty_or_malformed_parent_allowlist_never_uses_profile_defaults(parent):
    enabled, _disabled, allowed = _resolve_child_tool_permissions(parent, None, "leaf")

    assert enabled == []
    assert allowed == frozenset()


def test_child_explicit_tools_are_intersected_with_parent_and_empty_is_deny_all():
    import model_tools

    parent = SimpleNamespace(
        enabled_toolsets=["file", "web"], disabled_toolsets=None,
        valid_tool_names=model_tools._select_tool_names(["file", "web"], None, quiet_mode=True),
    )
    _enabled, _disabled, allowed = _resolve_child_tool_permissions(parent, ["file"], "leaf")
    assert "read_file" in allowed
    assert "web_search" not in allowed
    assert "delegate_task" not in allowed

    _enabled, _disabled, empty = _resolve_child_tool_permissions(parent, [], "orchestrator")
    assert empty == frozenset()


def test_child_cannot_inherit_delegation_role_permission_without_parent_grant():
    parent = SimpleNamespace(
        enabled_toolsets=["file"], disabled_toolsets=None,
        valid_tool_names={"read_file"},
    )

    enabled, _disabled, allowed = _resolve_child_tool_permissions(parent, None, "orchestrator")

    assert "delegation" not in enabled
    assert "delegate_task" not in allowed


def test_orchestrator_role_does_not_widen_an_explicit_child_request():
    parent = SimpleNamespace(
        enabled_toolsets=["file", "delegation"], disabled_toolsets=None,
        valid_tool_names={"read_file", "delegate_task"},
    )

    enabled, _disabled, allowed = _resolve_child_tool_permissions(parent, ["file"], "orchestrator")

    assert "delegation" not in enabled
    assert "delegate_task" not in allowed


def test_child_grant_excludes_toolsets_removed_by_child_policy():
    import model_tools

    parent = SimpleNamespace(
        enabled_toolsets=["hermes-cli"], disabled_toolsets=None,
        valid_tool_names=model_tools._select_tool_names(["hermes-cli"], None, quiet_mode=True),
    )

    _enabled, disabled, allowed = _resolve_child_tool_permissions(parent, None, "leaf")

    assert "kanban" in disabled
    assert "kanban_show" not in allowed
    assert "delegate_task" not in allowed


@pytest.mark.parametrize("kwargs", [
    {
        "task_id": "", "principal_id": "parent", "permitted_tools": ["read_file"],
        "permitted_resources": [], "ttl_seconds": 60,
    },
    {
        "task_id": "task", "principal_id": "", "permitted_tools": ["read_file"],
        "permitted_resources": [], "ttl_seconds": 60,
    },
    {
        "task_id": "task", "principal_id": "parent", "permitted_tools": [""],
        "permitted_resources": [], "ttl_seconds": 60,
    },
    {
        "task_id": "task", "principal_id": "parent", "permitted_tools": ["read_file"],
        "permitted_resources": [], "ttl_seconds": 0,
    },
    {
        "task_id": "task", "principal_id": "parent", "permitted_tools": ["read_file"],
        "permitted_resources": [], "ttl_seconds": 86401,
    },
    {
        "task_id": "task", "principal_id": "parent", "permitted_tools": [],
        "permitted_resources": [], "ttl_seconds": 60,
    },
    {
        "task_id": "task", "principal_id": "parent", "permitted_tools": {"read_file": True},
        "permitted_resources": [], "ttl_seconds": 60,
    },
    {
        "task_id": "task", "principal_id": "parent", "permitted_tools": count(),
        "permitted_resources": [], "ttl_seconds": 60,
    },
    {
        "task_id": "task", "principal_id": "parent", "permitted_tools": [f"tool-{i}" for i in range(257)],
        "permitted_resources": [], "ttl_seconds": 60,
    },
    {
        "task_id": "task", "principal_id": "parent", "permitted_tools": ["read_file"],
        "permitted_resources": [f"workspace:{i}" for i in range(129)], "ttl_seconds": 60,
    },
    {
        "task_id": "t" * 513, "principal_id": "parent", "permitted_tools": ["read_file"],
        "permitted_resources": [], "ttl_seconds": 60,
    },
])
def test_invalid_or_unbounded_task_claim_is_not_minted(kwargs):
    assert issue_worker_tool_authorization(**kwargs) is None


def test_local_claim_is_immutable_identity_bound_and_single_use():
    claim = _issue()
    assert claim is not None
    assert claim.authority == "hermes-in-process-dev-only"
    with pytest.raises(AttributeError):
        claim.task_id = "other"

    worker = _worker(claim)
    assert active_worker_tool_allowlist(worker) == frozenset({"read_file"})

    replay = _Agent()
    replay.platform = "subagent"
    replay.worker_task_id = "task-1"
    replay.worker_principal_id = "parent-1"
    replay.worker_tool_authorization = None
    replay.worker_tool_allowlist = frozenset()
    replay.worker_resource_scope = claim.permitted_resources
    assert not attach_worker_tool_authorization(
        replay, claim, task_id="task-1", principal_id="parent-1",
        permitted_resources=claim.permitted_resources,
    )
    assert active_worker_tool_allowlist(replay) == frozenset()


@pytest.mark.parametrize("task_id,principal_id", [("other-task", "parent-1"), ("task-1", "other-parent")])
def test_untrusted_or_identity_mismatched_claim_fails_closed(task_id, principal_id):
    claim = _issue()
    assert claim is not None
    worker = _Agent()
    worker.platform = "subagent"
    worker.worker_task_id = task_id
    worker.worker_principal_id = principal_id
    worker.worker_tool_authorization = None
    worker.worker_tool_allowlist = frozenset()
    worker.worker_resource_scope = claim.permitted_resources

    assert not attach_worker_tool_authorization(
        worker, claim, task_id=task_id, principal_id=principal_id,
        permitted_resources=claim.permitted_resources,
    )
    assert active_worker_tool_allowlist(worker) == frozenset()


def test_synthetic_dataclass_and_expired_claim_are_not_accepted():
    locally_fabricated = WorkerToolAuthorization(
        task_id="task-1", principal_id="parent-1", permitted_tools=frozenset({"read_file"}),
        permitted_resources=frozenset(), expires_at=time.monotonic() + 60,
        nonce="not-issued-by-policy", authority="hermes-in-process-dev-only",
    )
    worker = _worker()
    assert not attach_worker_tool_authorization(
        worker, locally_fabricated, task_id="task-1", principal_id="parent-1",
        permitted_resources=locally_fabricated.permitted_resources,
    )

    expired = replace(_issue(), expires_at=time.monotonic() - 1)
    assert not attach_worker_tool_authorization(
        _worker(), expired, task_id="task-1", principal_id="parent-1",
        permitted_resources=expired.permitted_resources,
    )


def test_expiring_an_attached_claim_revokes_dispatch_immediately(monkeypatch):
    from tools.registry import registry

    worker = _worker(_issue())
    worker.worker_tool_authorization = replace(
        worker.worker_tool_authorization, expires_at=time.monotonic() - 1,
    )
    entry = SimpleNamespace(handler=lambda args, **kwargs: '{"ok":true}', is_async=False)
    monkeypatch.setattr(registry, "get_entry", lambda name, scope=None: entry)

    with worker_tool_policy_scope(worker):
        result = registry.dispatch("read_file", {})

    assert "allowlist" in result


def test_subagent_without_a_claim_is_default_deny(monkeypatch):
    from tools.registry import registry

    worker = _worker()
    entry = SimpleNamespace(handler=lambda args, **kwargs: '{"ok":true}', is_async=False)
    monkeypatch.setattr(registry, "get_entry", lambda name, scope=None: entry)

    with worker_tool_policy_scope(worker):
        result = registry.dispatch("read_file", {})

    assert "allowlist" in result


def test_resource_scope_must_match_the_task_claim():
    claim = _issue()
    worker = _Agent()
    worker.platform = "subagent"
    worker.worker_task_id = "task-1"
    worker.worker_principal_id = "parent-1"
    worker.worker_tool_authorization = None
    worker.worker_tool_allowlist = frozenset()
    worker.worker_resource_scope = frozenset({"workspace:C:/other"})

    assert not attach_worker_tool_authorization(
        worker, claim, task_id="task-1", principal_id="parent-1",
        permitted_resources=worker.worker_resource_scope,
    )
    assert active_worker_tool_allowlist(worker) == frozenset()


def test_dynamic_tool_snapshot_is_filtered_to_the_immutable_allowlist():
    worker = _worker(_issue())

    definitions, names = filter_worker_tool_definitions(
        worker, [_tool("read_file"), _tool("mcp_late_tool"), _tool("lcm_grep")],
    )

    assert names == {"read_file"}
    assert [item["function"]["name"] for item in definitions] == ["read_file"]


def test_registry_final_dispatch_denies_direct_nested_tool_calls(monkeypatch):
    from tools.registry import registry

    worker = _worker(_issue())
    called = []

    def handler(args, **kwargs):
        called.append(args)
        return '{"ok":true}'

    entry = SimpleNamespace(handler=handler, is_async=False)
    monkeypatch.setattr(registry, "get_entry", lambda name, scope=None: entry)
    with worker_tool_policy_scope(worker):
        allowed = registry.dispatch("read_file", {"path": "C:/dev/a.txt"})
        denied = registry.dispatch("write_file", {"path": "C:/dev/a.txt", "content": "x"})

    assert allowed == '{"ok":true}'
    assert len(called) == 1
    assert "allowlist" in denied


def test_execute_code_registry_fallback_uses_worker_permissions(monkeypatch):
    from tools.registry import registry

    claim = _issue(tools=("execute_code", "read_file"))
    worker = _worker(claim)
    observed = {}

    def handler(args, **kwargs):
        observed.update(kwargs)
        return '{"ok":true}'

    entry = SimpleNamespace(handler=handler, is_async=False)
    monkeypatch.setattr(registry, "get_entry", lambda name, scope=None: entry)
    with worker_tool_policy_scope(worker):
        registry.dispatch("execute_code", {})

    assert set(observed["enabled_tools"]) == {"execute_code", "read_file"}


def test_agent_level_dispatch_fails_closed_before_inline_execution():
    from agent.agent_runtime_helpers import invoke_tool

    worker = _worker(_issue())

    result = invoke_tool(worker, "write_file", {"path": "C:/dev/a.txt", "content": "x"}, "task-1")

    assert "allowlist" in result


def test_deferred_recursive_dispatch_cannot_escape_worker_allowlist(monkeypatch):
    import model_tools

    claim = issue_worker_tool_authorization(
        task_id="task-1", principal_id="parent-1", permitted_tools=("tool_call",),
        permitted_resources=(), ttl_seconds=60,
    )
    worker = _worker(claim)
    monkeypatch.setattr(
        model_tools, "_dispatch_bridge_tool",
        lambda *args, **kwargs: (None, ("write_file", {"path": "C:/dev/a.txt", "content": "x"})),
    )

    with worker_tool_policy_scope(worker):
        result = model_tools.handle_function_call("tool_call", {})

    assert "allowlist" in result
