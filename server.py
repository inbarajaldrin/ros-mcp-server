"""server.py — thin entry for the consolidated ros-mcp-server.

Builds a ModeConfig from env at import (the existing ROS_MCP_MODE / MCP_CHECKPOINT_GATE
pattern — NO new injection mechanism), installs it into server_core, runs the startup
fail-closed config check (same import-time fail-closed behavior as before), and
runs the server.

All tool definitions, the harness (gates / commit_object / set_task_phase / floor /
predicates / signals), the 14 shared backend fns, and the real-world state-store live in
server_core.py. server_quat.py is UNTOUCHED (separate file + config).

Mode selection (one env knob, base substrate = remap; ROS_DOMAIN_ID=0 always):
  ROS_MCP_MODE unset / "sim"   -> sim/canonical route-mode (real->sim redirect, full harness,
                                  ground-truth pose). This is the proven remap-sim behavior.
  ROS_MCP_MODE = "real"        -> real-direct route-mode (identity, order-only pre-gates,
                                  state-store pose injection + --use-default-base-position).

Importing this module re-exports server_core's public surface (and the state-store helpers
the real-mode harness reaches), so existing in-process importers keep working.
"""
import os

import server_core
from server_core import (  # re-export the full surface so `import server` behaves like before
    mcp,
    ModeConfig,
    _remap_mode,
    _identity_mode,
    logger,
    ws_manager,
)

# Re-export the state-store + gate helpers reached in-process by tests / triggers
# (ablations/test_real_checkall_gate.py loads this module and calls these directly).
from server_core import (
    _state_path,
    _lock_path,
    _load_state,
    _save_state,
    _get_object_state,
    _set_object_state,
    clear_primitive_state,
    _run_certified_seated_set,
    _assert_assembly_order,
    _assert_assembly_order_real,
)


def _build_mode_config() -> ModeConfig:
    """Build the active ModeConfig from env (import-time, one knob).

    The base substrate is the canonical remap server. ROS_MCP_MODE='real' flips the
    route-mode to identity (real-direct) and turns on the state-store seams; anything
    else (unset / 'sim') keeps the proven sim/remap behavior.
    """
    runtime_mode = os.getenv("ROS_MCP_MODE", "").strip().lower()
    if runtime_mode == "real":
        # Real-direct route-mode (folded from the frozen server.py): identity routing,
        # order-only pre-gates, state-store pose injection, default-base extras.
        return ModeConfig(
            route_mode=_identity_mode,
            pose_store=True,
            real_default_base=True,
            full_harness=False,
            quat=False,
        )
    # Default: sim / canonical remap route-mode (the proven remap-sim behavior).
    return ModeConfig(
        route_mode=_remap_mode,
        pose_store=False,
        real_default_base=False,
        full_harness=True,
        quat=False,
    )


# Install the ModeConfig + run the import-time fail-closed config check NOW, so a
# launched server fails closed exactly as the prior server did at import.
server_core.configure(_build_mode_config())
server_core.run_startup_checks()


if __name__ == "__main__":
    server_core.serve()
