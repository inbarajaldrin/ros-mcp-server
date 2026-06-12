# Server Consolidation Plan

**Status:** scoped, not started · **Date:** 2026-06-12 · **Owner:** ros-runner

Reconciles the ros-mcp-server entrypoints into one canonical server. Doc-first by design —
this is the spec to implement against; do NOT start editing ~5k lines without it (the prior
scoping session flagged "this can break all three experiment arms if rushed").

---

## 1. Premise — why now

Three server files exist, but they are no longer three equals:

| File | Lines | State |
|---|---|---|
| `server_remap.py` | 2268 | **Canonical truth** — remap substrate + the entire harness (gates, `commit_object`, `set_task_phase`, floor, predicates). Edited continuously; P1/P2/P3 all validated on it. |
| `server.py` | 1691 | Frozen Jun 8. Pre-harness. Unique payload = the **real-world state-store** + real-direct routing. |
| `server_quat.py` | 1454 | Frozen Jun 8. Strict subset of remap + the quat pose-param contract. **LEFT ALONE this pass.** |

Because everything runs through `server_remap` today and all three phases pass, **the merge
target already exists and is proven.** This is a refactor of a working file into a cleaner
shape, not a reconciliation of divergent logic. That is what makes it safe to do now.

**Scope decision (operator, 2026-06-12):** the **two-server merge** (`server.py` real-world
logic → `server_remap` → new `server.py`) is **mandatory**. `server_quat.py` and
`mcp_config_quat.json` are **left untouched** and remain usable. `MCP_AGENT_QUAT` folds into the
core **only if the signature shim proves clean** (§7); otherwise quat stays on its own file.

---

## 2. Target structure

```
server_core.py   # each tool defined ONCE; tool bodies + the shared backend + ModeConfig
server.py        # thin entry: build ModeConfig from env, register the surface, mcp.run()
server_quat.py   # UNTOUCHED (separate file + config; folded later if ever)
```

The unified tool skeleton — every tool routes through one body delegating to four named seams:

```python
def TOOL(object_name, ..., mode):
    mode = cfg.route_mode(mode)                       # seam 1
    if g := cfg.apply_pre_gates(TOOL, object_name, mode, ...): return g   # seam 2
    pose = cfg.resolve_pose(object_name, grasp_id, orientation)          # seam 3
    cmd  = build_cmd(..., pose, cfg.real_extras(mode))                   # seam 4
    return _parse_result(_run_with_retry(_run_primitive, "TOOL.py", cmd))
```

`ModeConfig` is built from env at import (continues the existing `ROS_MCP_MODE` /
`MCP_CHECKPOINT_GATE` pattern — no new injection mechanism). Fields = the four seams.

---

## 3. The 4 seams (where ALL divergence lives — never robot logic)

14 backend functions are byte-identical across all files (zero work): `execute_python_code,
get_topics, read_topic, _run_primitive, _run_query, _run_with_retry, _parse_result,
_start_services, _ensure_services_healthy, _wait_for_rosbridge, _handle_elicitation,
_invoke_scene_setup, _np_to_mcp_image, _is_connection_error`.

| Seam | `server.py` (real) | `server_remap.py` (sim, canonical) | quat (untouched) |
|---|---|---|---|
| **1 route-mode** | `--mode {mode}` (identity, real→real) | `--mode sim` via `_remap_mode` (real→sim) | `_remap_mode` |
| **2 pre-gates** | order only | full harness (init/order/hold/home/phase#/ckpt) | none |
| **3 pose-resolution** | inject from **state-store** | omit — sim primitive reads ground-truth pose | agent passes required quat params |
| **4 real-extras** | `--use-default-base-position` when real | (never real) | same as split |

Canonical-server per-tool gate matrix (from `server_remap`, the body we keep):

| Tool | route | pre-gates | state-store (real seam 3/4) |
|---|---|---|---|
| `move_home` | remap | init, home | — |
| `control_gripper` | remap | init | — |
| `move_to_grasp` | remap | init, order | writes (real) |
| `translate_object` | remap | init, order | reads + default-base (real) |
| `rotate_object` | remap | init, order | reads + writes (real) |
| `move_to_safe_height` | remap | init, hold | — |
| `verify_grasp` | remap | — | reads orientation (real) |
| `verify_assembly` | remap | — | writes certified seated-set (real) |
| `verify_disassembly` | remap | — | — |
| `get_scene_info` | remap | — | — |
| `commit_object` | remap | order, phase#, ckpt | — |
| `signal_phase_complete` | remap | phase# | — |
| `signal_operator` | remap | — | — |
| `set_task_phase` | (host) | — | — — close the host-only leak (§6) |

Tools to **retire** (not in `paper-tool-manifest.json`): `verify_clearance`, `scan_workspace`
(may return later as a recovery-only primitive, never a phase), `signal_verify_results`.

---

## 4. The real-world state-store (the substantive fold)

This is the unique payload `server.py` carries. In **sim** the server queries the twin for
"what is assembled"; in **real** the camera is occluded by the gripper (the "held-object pose
chains, not reads" rule), so the server maintains a certified store chained across tool calls.

- File-backed: `$MCP_CLIENT_OUTPUT_DIR/primitive_state.json` + `.lock` (file-locked R/W).
- `run_id`-scoped: `clear_primitive_state` stamps a fresh `run_id` at phase start; trust requires
  a `run_id` stamped THIS run — `_run_certified_seated_set` **fails closed** without it.
- Functions to fold into `server_core`: `_state_path`, `_lock_path`, `_load_state[_unlocked]`,
  `_save_state[_unlocked]`, `_get_object_state`, `_set_object_state`, `clear_primitive_state`,
  `_run_certified_seated_set`.
- Registered as the `resolve_pose` (seam 3) implementation for the real route-mode; the sim
  route-mode's `resolve_pose` is "omit — read ground-truth." `verify_assembly`'s real path uses
  `_run_certified_seated_set` as its seated-set oracle.

---

## 5. ModeConfig

```python
@dataclass
class ModeConfig:
    route_mode:    Callable        # identity (real-direct) | real→sim redirect
    apply_pre_gates: Callable      # which pre-gates fire (full harness vs order-only)
    resolve_pose:  Callable        # state-store(real) | ground-truth(sim) | agent-supplied(quat)
    real_extras:   Callable        # --use-default-base-position when real, else none
    quat: bool = False             # only if §7 lands; else server_quat handles it
```

Base substrate = remap: **`ROS_DOMAIN_ID=7` always, no sim/real domain switch built.** The
real route-mode flips `route_mode` to identity + `resolve_pose` to the state-store; it does NOT
change the domain. Selected by one env knob at import.

---

## 6. Drift fixes the merge folds in

- **`_run_primitive` env-injection:** `server.py`/`server_quat` are MISSING the `ROS_MCP_MODE`
  env injection + telemetry that remap's `_run_primitive` gained. The merge adopts remap's
  version (the correct one) — incidentally fixes the split drift.
- **`set_task_phase` host-only leak:** it's framework-required (arms the order gate) and
  host-only by design, but agent-callable today (not in the 18-tool manifest, not deny-listed).
  Close it in the merge: category `config` + a host-only guard (`_is_host_call`).

---

## 7. Quat fold (CONDITIONAL — only if clean)

Quat differs from remap in exactly ONE seam: pose-resolution — the 3 pose tools
(`translate_object`, `rotate_object`, `verify_grasp`) take **required** orientation params.
Mechanism = thin per-mode signature shims (NOT isaac's dynamic `_build_tool_function` — judged a
forced fit): when `MCP_AGENT_QUAT=1`, register the 3 tools with the quat signature; off → without.

Decision: structure `resolve_pose` so quat is a third value (`agent-supplied`), making the
architecture quat-ready. **Build the `MCP_AGENT_QUAT` shim only if the signature-conditional
registration is clean in FastMCP.** If it is fiddly, STOP — leave `server_quat.py` on its own
file/config (the operator's default). Do not create two drifting quat implementations.

---

## 8. Acceptance / parity (the gate before deleting `server_remap.py`)

1. **Tool-surface parity:** the unified server exposes the identical canonical tool *names* as
   today's `server_remap` — checked against `paper-tool-manifest.json` (the client preflight
   refuses to run on drift; it checks names, not params, so quat shims don't break it).
2. **`tests/probe_gates.py`** passes on `home` / `commit` / `signal` (no-LLM, live sim).
3. **Live acceptance (operator-specified):** the **gpt-5-mini keyed P2 with gpt-5 as the ablation
   model, runs=1**, reproduces the results it already gives.

Only when 1–3 are green: delete `server_remap.py`. (`server_quat.py` stays.)

---

## 9. Build order

1. **This doc** (done).
2. `server_core.py` first — the `_meta` table + ModeConfig + the 14 shared backend fns + tool
   bodies live here; everything else bolts onto it.
3. Thin `server.py` — build ModeConfig from env, register surface, `mcp.run()`.
4. Fold state-store (§4) + drift fixes (§6); retire dead tools (§3).
5. Quat (§7) — only if clean.
6. Parity gate (§8.1–8.2) → live acceptance (§8.3).
7. Delete `server_remap.py`.
8. **Config squash (cross-repo, with client-runner):** 3 configs → 1 `mcp_config.json` +
   per-study `serverMode`/`MCP_AGENT_QUAT` → env (the `resolveMcpConfigPath` change). Server-first;
   the 3 configs temporarily point at the one `server.py` via env before the client squash lands.

---

## 10. Risks

- **Cross-repo coupling (config squash):** needs an `mcp-client-runner` edit. Coordinate (same as
  bug-2). Do it AFTER the server merge is proven, not during.
- **Signature shims (quat):** the one place modes differ in agent-visible schema. Conditional
  FastMCP registration is the only fiddly bit — gated behind §7's "only if clean."
- **The 5k-line edit:** mitigated by doc-first + the live P2 acceptance reproducing a known-good
  result. Keep `server_remap.py` on disk until §8 is fully green — it is the rollback.
