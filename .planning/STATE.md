---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: planning
stopped_at: Phase 1 [R]-half ✅ DONE this session (CAL-03/04/06..08); Phase 2 [R]-half (WRAP-VERIFY) outstanding — needs operator to set up u_brown
last_updated: "2026-05-03T11:55:00.000Z"
last_activity: 2026-05-03 at-robot session — F/T payload calibration ran successfully (mass 2.11 kg, CoG ≈ origin), smoke test passed, paused before WRAP-VERIFY (needs physical part setup).
progress:
  total_phases: 7
  completed_phases: 2
  total_plans: 0
  completed_plans: 0
  percent: 28
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-01, gripper-name fix 2026-05-02)

**Core value:** Replace the failing `--insert` real-mode path with a force-compliant insert primitive that works reliably across every FMB1 part and is a one-config-file extension to any new part.
**Current focus:** WRAP-VERIFY (the only remaining at-robot task before Phase 3 data collection). Operator needs to physically place u_brown on FMB1 base + grasp it before next session. See `.planning/TRACKS.md` for the live checklist.

## Current Position

Phase: Phase 1 ✅ DONE 2026-05-03 (foundational F/T payload calibration recovered + smoke test passed); Phase 7 ✅ shipped 2026-05-02 (RG2 URDF + dual-RobotModel RViz). Phase 2 [N]-half shipped, [R]-half (WRAP-VERIFY) is the next at-robot task. 2 of 7 phases formally complete.
Most recent work: 2026-05-03 — CAL-03 (mass 2.11 kg, CoG ≈ [-0.003, +0.003, -0.032]), CAL-04 (all 8 poses reachable), CAL-06..08 (smoke test PASS), HOME_JOINTS introduced, wrapper frame-conversion bug fixed, workspace convention +X = robot's RIGHT empirically verified.
Next at-robot session goal: WRAP-VERIFY end-to-end on u_brown (~15 min including induced-failure tests), then Phase 3 data collection.

Progress: [██░░░░░░░░] 28% (2 of 7 phases formally complete; Phase 2 [N]-half done — only [R]-half WRAP-VERIFY remains)

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: —
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Foundation Setup + F/T Calibration | 0 | — | — |
| 2. Episode Wrapper + Telemetry Schema | 0 | — | — |
| 3. 20-Episode FMB1 Data Collection | 0 | — | — |
| 4. Analyzer Dashboard | 0 | — | — |
| 5. Algorithm Derivation + Per-Object Configs | 0 | — | — |
| 6. Dispatcher Integration + Generalization Validation | 0 | — | — |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Initialization: Termination criterion treated as a Phase 5 deliverable, not an input — operator honestly cannot pre-answer which signal (force-absorbed / z-reached / motion-stopped) is most reliable; data will reveal it.
- Initialization: Parametric universal algorithm + per-object YAML, not per-object scripts — all FMB1 objects are peg-in-hole; differences are tolerances, force levels, axis choices.
- Initialization: Single static HTML dashboard (Plotly.js + PapaParse from CDN) — sole user is the operator; functional plots beat polish.
- Initialization: Existing `_real_mode_stash/` is reference-only, not foundation — stale import paths, untracked, overcomplicated; re-derive cleanly using stash's force-mode RPC patterns as guidance.

### Pending Todos

None yet.

### Blockers/Concerns

None yet. Build-order interlocks (per research SUMMARY.md) are baked into phase ordering — no parallel-phase risks at this point.

### Roadmap Evolution

- 2026-05-02: Phase 7 added: Gripper URDF + RViz Visualization (USD→URDF conversion + integrate OnRobot RG2 into UR5e URDF for accurate RViz preview and ros2_control collision-checking; replaces custom DH-based `GRIPPER_CENTER_TOOL_OFFSET`). Per CONVENTIONS §6, this phase is decoupled from Phases 1–6 and recommended to be pulled forward to BEFORE the first real-hardware Phase 1 calibration run, so operator can verify calibration poses with full gripper geometry visible in RViz.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none — initial milestone)* | | | |

## Session Continuity

Last session: 2026-05-03T11:55:00.000Z
Stopped at: Clean session-pause after Phase 1 [R]-half completion. All ROS processes shut down via close_robot.sh. Operator declined skinny WRAP-VERIFY in favor of resuming next at-robot session with u_brown physically set up.
Resume file: .planning/TRACKS.md (live at-robot checklist) + .planning/HANDOFF.json (decisions + next_action block)
