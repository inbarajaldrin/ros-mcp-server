# ROS-MCP Server Modes

Three server variants, one per assembly-collection mode. Each is launched by its own
config in `mcp-client-example/`. The modes differ on two orthogonal axes:
**execution routing** (where primitives run) and **pose contract** (who supplies the
object orientation to the primitives).

| Mode | Server file | Config (mcp-client-example) | Execution routing | Pose contract |
|------|-------------|------------------------------|-------------------|---------------|
| **Split** — true sim/real (default) | `server.py` | `mcp_config.json` | real→real, sim→sim | **backend** resolves from `primitive_state.json`; agent passes object name only |
| **Remap** — ablation | `server_remap.py` | `mcp_config_remap.json` | real→**sim** (Phase 3 labelled `real`, executes sim) | none (sim-style params, no quat) |
| **Quat** — quat mandated | `server_quat.py` | `mcp_config_quat.json` | real→sim | **agent** passes `current_object_orientation [x,y,z,w]` in sim AND real |

**Lineage:** `server_quat` (original ablation server) → `server_remap` (= quat server
with the agent quat requirement stripped). `server.py` (the Split / true-real server,
formerly `server_p3.py`) auto-injects the quat from the state store, so the agent never
passes it — hence it is the canonical default server.

## Which mode each experiment uses
- `mode2_escalation`, `mode1_isolated`, `ablation_context`, `ablation_orchestrator` → **Remap** (`mcp_config_remap.json`)
- `ablation_quat` → **Quat** (`mcp_config_quat.json`)
- `mode3_real_execution/{sim,real}` → **Split** (`mcp_config.json`)

## Split-mode real runs — state store contract
The Split server keeps a per-run `primitive_state.json` (a FIXED shared file under
`MCP_CLIENT_OUTPUT_DIR`). Occlusion-tolerant real `verify_assembly --check-all` and the
assembly-order gate trust it ONLY when a `run_id` has been stamped this run. Therefore
every `mode3_real_execution/real/*.yaml` calls `@tool-exec:ros-mcp-server__clear_primitive_state()`
in `onStart` (atomic reset to `{run_id, objects:{}}`). See
`.planning/track-b-realmode-checkall-design.md` (in the paper1 repo) for the full design.

## Cleanup history (2026-05-30)
- Renamed servers (git mv, history preserved): `server_p3.py`→`server.py` (Split, default),
  `server_mode2.py`→`server_remap.py`, `server_ablation_quat.py`→`server_quat.py`.
- Deleted the original baseline `server.py` (no config launched it); the name now belongs
  to the Split server.
- Configs renamed to match modes: Split → `mcp_config.json` (was `mcp_config_real.json`),
  Remap → `mcp_config_remap.json` (was `mcp_config.json`), Quat → `mcp_config_quat.json`
  (was `mcp_config_ablation_quat.json`). Removed the duplicate `mcp_config_mode2.json`
  (preserved as `*.deleted-dup-20260530`).
- Fixed mis-wired refs: `mode3_real_execution/real/*` (was Remap) and `/sim/*` (was a
  non-existent `mcp_config_p3.json`) → now Split (`mcp_config.json`).
- KNOWN GAP: `mode3_real_execution/real/fmb2.yaml` does not exist (only fmb1, fmb3).
