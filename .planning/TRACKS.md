# Execution Tracks — Away vs At Robot

Live list of what's actionable on each track. **Update whenever a task transitions states.**

This complements ROADMAP.md (which is phase-structured) and REQUIREMENTS.md (which is requirement-tagged). TRACKS.md answers the question "what should I work on right now given my current physical access to the robot?"

See `.planning/codebase/CONVENTIONS.md` for the two-track principle.

---

## Track legend

- `[N]` — **No-robot**: can be done from anywhere with the codebase.
- `[R]` — **Robot-required**: needs the physical UR5e + ROS bringup.
- `[N→R]` — **Hybrid**: scaffolding done away, final verification at-robot.

---

## Active session: 2026-05-04 — Phase 4 CLOSED, operator back at-robot, Phase 5 entry

### Done this session (2026-05-03 long at-robot day)

- ✅ **Phase 3 COMPLETE — 60 demos collected** across all 4 FMB1 objects (u_brown 10 + u_orange 10 + line_green 20 + inverted_u_yellow 20)
  - 7 autonomous + 53 assisted; all schema_v1.1; all `assist_level` + `user_notes` populated
- ✅ **Schema bumped to v1.1** (additive): `assist_level`, `current_object_orientation_input`, `tcp_pose_at_active_start`, `hover_pose_world` (promoted), `tcp_to_object_transform`; CSV: `wrench_frame_id` + `obj_x..obj_qw` per row
- ✅ **Wrapper wired to populate v1.1 fields**: derives tcp_to_object_quat at HOVER end, per-row obj_q* = R_tcp × R_t2o
- ✅ **--setup-only flag** added to `run_assembly_step.py` (lets agent run wrapper as background for SIGTERM control)
- ✅ **--skip-home-on-done** flag added to compliant_insert.py (saves ~5s per cleanup)
- ✅ **Force-mode tuning locked** for operator-in-the-loop demos: fz=9N (override 5N cap) + lin-speed=0.54 + gain=1.0
- ✅ **Two new anti-patterns added to CLAUDE.md** (kill -0 not pgrep -f; one Bash call per workflow step)

### Phase status

- **Phase 1, Phase 2, Phase 3, Phase 4, Phase 7 ✅ DONE** (5/7 = 71%)
- **Phase 5 ➡️ ACTIVE** (hybrid — at-robot for live testing, at-away for YAML+code)

---

## Ready to work right now

### At-away track (`[N]`) — RECOMMENDED NEXT

1. **Phase 4 — Dashboard scaffolding** (DASH-01..09, ~4-6 hours)
   - **Now has REAL data**: 60 CSVs in `compliant_insertion_studio/logs/` (10/10/20/20 across the 4 FMB1 objects)
   - Single static HTML using Plotly + PapaParse from CDN, no build step, no backend
   - File to create: `compliant_insertion_studio/analyzer/analyze_inserts.html`
   - Per DASH-04: per-object signature cards auto-compute median Fz at success, |Tx|/|Ty| peak distribution, lateral travel during ACTIVE, descent duration — using only hands-off-window-restricted samples
   - Schema reference: `compliant_insertion_studio/docs/SCHEMA.md` + `compliant_insertion_studio/wrapper/schema_v1.py` (v1.1 = 41 columns)

2. **Phase 6 — Dispatcher code stub** (DISP-01..03, can start in parallel)
   - Config resolution + MANUAL_GUIDED fallback
   - Doesn't need Phase 3 data or Phase 5 configs to scaffold

3. **Commit the pending changes** (10 min, requires operator approval)
   - See HANDOFF.json `uncommitted_files` list (60 demo CSVs + 7 modified/new code files)
   - Suggested grouping in HANDOFF.json `remaining_tasks` "Commit cleanup"

### At-robot track (`[R]`) — only if continuing collection or re-running

1. **Release inverted_u_yellow** (currently held in gripper from last session) + close stack — see `.continue-here.md` Infrastructure State for cleanup commands
2. **Re-run Phase 3** (if restarting from clean state) — use `run_assembly_step.py` per object with the locked tuning params: `--fz 9.0 --override-fz-cap --lin-speed 0.54 --gain 1.0 --step-back auto --auto-step-back-seconds 5 --no-prompt-notes --skip-home-on-done`

### Blocked

- **Phase 5 — Algorithm + per-object configs** — blocked until Phase 4 dashboard surfaces the signature cards (now unblocked from data side; only Phase 4 in the way)
- **Phase 6 — Generalization validation** (VAL-01..05) — blocked until Phase 5 configs exist
- **Phase 6 — Integration test at translate_object.py:1085** — blocked on Phase 5 algorithm

---

## Bug Fix Inventory (this session) — pending commit approval

| # | File | Fix | Severity |
|---|---|---|---|
| 1 | `compliant_insertion_studio/wrapper/compliant_insert.py` | Strip ANSI escapes in `_list_active_controllers` (was always returning empty set → every controller switch silently aborted) | **CRITICAL — silent show-stopper** |
| 2 | `compliant_insertion_studio/wrapper/compliant_insert.py` | Bumped `_await_controller_active` timeout 2s → 5s | High |
| 3 | `compliant_insertion_studio/wrapper/compliant_insert.py` | Added `import re` for the ANSI strip | (part of #1) |
| 4 | `compliant_insertion_studio/wrapper/_run_hover.py` | `TranslateObject(mode="real")` (was missing required mode arg) | High |
| 5 | `primitives/move_to_grasp.py:888` | Tolerance `expected-2.0` → `expected-5.0` (matches RG2 mode-16 ±3-5mm grip overshoot) | Medium |
| 6 | `compliant_insertion_studio/scripts/launch_camera.sh` | NEW helper for aruco camera node | New file |
| 7 | `compliant_insertion_studio/scripts/run_assembly_step.py` | NEW reusable orchestrator script (Phase 3 entry point) | New file |
| 8 | `~/Desktop/ros2_ws/src/onrobot_ros/onrobot_ros/rg_gripper.py` | `/gripper_status` formatter now includes `Circuit1`/`Circuit2` (separate repo, NOT in this commit set) | Medium |

---

## When the operator next leaves the robot

Suggested at-away sequence:

1. **Phase 4 dashboard scaffolding** (DASH-01..09): single static HTML against synthetic CSVs
2. **Phase 6 dispatcher code** (DISP-01..03): config resolution + MANUAL_GUIDED fallback
3. **Review and commit the bug-fix inventory above** if not already done

---

*Updated: 2026-05-03 17:31 UTC — clean session pause AFTER Phase 3 complete. Phase 1 + 2 + 3 + 7 done (4/7 = 57%); Phase 4 ready at-away. 60 CSVs collected across all 4 FMB1 objects with full schema_v1.1 metadata. Force-mode tuning locked at fz=9N + lin-speed=0.54 + gain=1.0. Two new anti-patterns added to CLAUDE.md (kill -0 not pgrep -f; one Bash call per workflow step).*
*Update mechanism: edit this file when transitioning a task between Ready / In-Progress / Done / Blocked, or when a phase changes which track items it owns.*
