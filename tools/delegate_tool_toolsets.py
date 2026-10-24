"""Child toolset resolution for delegate_task: what a child may and may never use."""

from __future__ import annotations

import logging
from typing import Optional

from toolsets import TOOLSETS, resolve_toolset
logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module

# Tools that children must never have access to
DELEGATE_BLOCKED_TOOLS = frozenset(
    [
        "delegate_task",  # no recursive delegation
        "clarify",  # no user interaction
        "memory",  # no writes to shared MEMORY.md
        "send_message",  # no cross-platform side effects
        "cronjob_manage",  # no scheduling more work in the parent's name
    ]
)
def _is_mcp_toolset_name(name: str) -> bool:
    """Return True for canonical MCP toolsets and their registered aliases."""
    if not name:
        return False
    if str(name).startswith("mcp-"):
        return True
    try:
        from tools.registry import registry
        target = registry.get_toolset_alias_target(str(name))
    except Exception:
        target = None
    return bool(target and str(target).startswith("mcp-"))

def _expand_parent_toolsets(parent_toolsets: set) -> set:
    """Add every toolset whose tools are a subset of the parent's tools: a parent on a composite like ``hermes-cli``
    must still let a child request ``web``/``terminal``; bare name intersection would reject them. Both sides use
    the RESOLVED static surface: a composite's ``includes`` (``debugging`` -> ``web``/``file``, ``safe``) are tools
    the parent genuinely holds, and the child never gains a tool the parent lacks."""
    parent_tool_names = {
        t for ts_name in parent_toolsets if ts_name in TOOLSETS for t in resolve_toolset(ts_name, include_registry=False)
    }
    expanded = set(parent_toolsets)
    if parent_tool_names:
        expanded.update(
            ts_name for ts_name in TOOLSETS
            if ts_name not in expanded
            and (resolved := resolve_toolset(ts_name, include_registry=False))
            and set(resolved).issubset(parent_tool_names)
        )
    return expanded

def _strip_blocked_tools(toolsets: list[str]) -> list[str]:
    """Remove toolsets whose tools are ALL blocked (derived from DELEGATE_BLOCKED_TOOLS so the two can't drift) plus
    composite toolsets children must never get (``delegation``, ``kanban``)."""
    blocked_toolset_names = {"delegation", "kanban"} | {
        name
        for name in TOOLSETS
        if (resolved_tools := resolve_toolset(name, include_registry=False))
        and all(tool in DELEGATE_BLOCKED_TOOLS for tool in resolved_tools)
    }
    return [t for t in toolsets if t not in blocked_toolset_names]

def _blocked_toolsets_for_role(role: str) -> list[str]:
    """One-tool deny toolsets for the role; passed as ``disabled_toolsets`` so
    blocked names inside mixed bundles are subtracted AFTER composite expansion."""
    blocked_names = set(DELEGATE_BLOCKED_TOOLS)
    if role == "orchestrator":
        blocked_names.discard("delegate_task")
    return sorted(
        name for name, defn in TOOLSETS.items() if defn.get("tools") and set(defn.get("tools", ())).issubset(blocked_names)
    )

def _resolve_child_tool_permissions(
    parent_agent, toolsets: Optional[list[str]], effective_role: str
) -> tuple[list[str], list[str], frozenset[str]]:
    """``(enabled_toolsets, disabled_toolsets)`` for a child. Children never gain tools the parent lacks: explicit
    ``toolsets`` are intersected with the parent's (composite-expanded) set, else the parent's enabled set is
    inherited. Blocked tools are stripped twice — whole blocked toolsets here, and exact one-tool deny toolsets via
    ``disabled_toolsets`` so blocked names inside mixed bundles (hermes-cli) are subtracted AFTER composite
    expansion and survive registry refreshes. Orchestrators retain ``delegation`` only when the parent grant allows it.
    Exact permitted tool names remain the final authority."""
    raw_parent_names = getattr(parent_agent, "valid_tool_names", None)
    if not isinstance(raw_parent_names, (set, frozenset, list, tuple)) or any(
        not isinstance(name, str) or not name.strip() for name in raw_parent_names
    ):
        parent_names = set()
    else:
        parent_names = set(raw_parent_names)
    if getattr(parent_agent, "platform", None) == "subagent":
        from agent.worker_tool_policy import active_worker_tool_allowlist
        parent_names.intersection_update(active_worker_tool_allowlist(parent_agent))

    parent_enabled = getattr(parent_agent, "enabled_toolsets", None)
    if parent_enabled is None:
        import model_tools
        parent_toolsets = {
            ts for name in parent_names if (ts := model_tools.get_toolset_for_tool(name)) is not None
        }
    else:
        if not isinstance(parent_enabled, (list, tuple, set, frozenset)) or any(
            not isinstance(name, str) or not name.strip() for name in parent_enabled
        ):
            parent_toolsets = set()
            parent_names = set()
        else:
            parent_toolsets = set(parent_enabled)
            if not parent_toolsets:
                parent_names = set()
            elif parent_names:
                try:
                    import model_tools
                    parent_selected_names = model_tools._select_tool_names(
                        sorted(parent_toolsets), getattr(parent_agent, "disabled_toolsets", None),
                        quiet_mode=True,
                    )
                    parent_names.intersection_update(parent_selected_names)
                except Exception:
                    parent_names = set()

    selection_valid = toolsets is None or (
        isinstance(toolsets, (list, tuple))
        and all(isinstance(name, str) and name.strip() for name in toolsets)
    )
    if selection_valid and toolsets is not None:
        expanded_parent = _expand_parent_toolsets(parent_toolsets)
        unknown = [name for name in toolsets if name not in TOOLSETS and not _is_mcp_toolset_name(name)]
        child_toolsets = [name for name in toolsets if name in expanded_parent] if not unknown else []
    elif selection_valid and parent_enabled is not None:
        child_toolsets = list(parent_enabled) if isinstance(parent_enabled, (list, tuple)) else sorted(parent_toolsets)
    elif selection_valid:
        child_toolsets = sorted(parent_toolsets)
    else:
        child_toolsets = []
    child_toolsets = _strip_blocked_tools(child_toolsets)

    raw_parent_disabled = getattr(parent_agent, "disabled_toolsets", None)
    inherited_disabled = (
        [str(name) for name in raw_parent_disabled] if isinstance(raw_parent_disabled, (list, tuple, set)) else []
    )
    if effective_role == "orchestrator":
        inherited_disabled = [name for name in inherited_disabled if name != "delegation"]
    child_disabled_toolsets = list(
        dict.fromkeys(inherited_disabled + _blocked_toolsets_for_role(effective_role) + ["kanban"])
    )
    if selection_valid and parent_names:
        candidate_toolsets = list(child_toolsets)
        requested_toolsets = list(toolsets) if toolsets is not None else sorted(parent_toolsets)
        import model_tools
        requested_names = model_tools._select_tool_names(
            requested_toolsets, child_disabled_toolsets, quiet_mode=True,
        )
        if effective_role == "orchestrator" and "delegate_task" in (parent_names & requested_names):
            # The orchestration role keeps delegation only when the parent's
            # exact effective grant and this child request both include it.
            candidate_toolsets.append("delegation")
        effective_names = model_tools._select_tool_names(
            candidate_toolsets, child_disabled_toolsets, quiet_mode=True,
        )
        allowed_names = parent_names.intersection(effective_names)
        blocked_names = set(DELEGATE_BLOCKED_TOOLS)
        if effective_role == "orchestrator" and "delegate_task" in allowed_names:
            blocked_names.discard("delegate_task")
        allowed_names.difference_update(blocked_names)
        if effective_role == "orchestrator" and "delegate_task" in allowed_names:
            child_toolsets.append("delegation")
    else:
        allowed_names = set()
    if not allowed_names:
        child_toolsets = []
    return child_toolsets, child_disabled_toolsets, frozenset(allowed_names)


def _resolve_child_toolsets(
    parent_agent, toolsets: Optional[list[str]], effective_role: str
) -> tuple[list[str], list[str]]:
    """Compatibility view; child construction also consumes the exact tool allowlist."""
    enabled, disabled, _allowed = _resolve_child_tool_permissions(parent_agent, toolsets, effective_role)
    return enabled, disabled
