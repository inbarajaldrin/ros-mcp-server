---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: planning
stopped_at: AT ROBOT — Phase 1 + Phase 2 + Phase 7 [N]-halves shipped; ready to do Phase 1 + Phase 2 + Phase 7 [R]-halves
last_updated: "2026-05-03T00:00:00.000Z"
last_activity: 2026-05-03 — at-robot session start. Phase 2 wrapper + telemetry schema shipped 2026-05-02; gripper-name fix sweep + dx/dy/dz warning + planning state refresh committed.
progress:
  total_phases: 7
  completed_phases: 1
  total_plans: 0
  completed_plans: 0
  percent: 14
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-01, gripper-name fix 2026-05-02)

**Core value:** Replace the failing `--insert` real-mode path with a force-compliant insert primitive that works reliably across every FMB1 part and is a one-config-file extension to any new part.
**Current focus:** AT ROBOT — Phase 1 [R]-half (SETUP-01, CAL-03/04, CAL-06..08), Phase 7 visual sign-off, Phase 2 wrapper end-to-end verification. See `.planning/TRACKS.md` for the ordered checklist.

## Current Position

Phase: Phase 7 of 7 ✅ shipped 2026-05-02 (Gripper URDF + RViz Visualization, real-bringup sign-off pending)
Most recent work: Phase 2 [N]-half ✅ shipped 2026-05-02 (episode wrapper + locked v1 telemetry schema). Phase 2 [R]-half (real-robot end-to-end + induced-failure tests) is the at-robot to-do.
Next at-robot session goal: Close Phase 1 [R]-half (drives Phase 2 [R]-half once calibration is good).
Status: Phase 7 done; Phase 1 [N]-half (11/17 reqs) done; Phase 2 [N]-half (17/17 reqs) done. 1 of 7 phases formally complete in roadmap.
Last activity: 2026-05-03 — at-robot session start. Pre-flight commits landed (4 commits + dx/dy/dz warning + this STATE.md refresh).

Progress: [█░░░░░░░░░] 14% (1 of 7 phases formally complete; significant code shipped in Phases 1+2 awaiting at-robot signoff to mark them complete)

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

Last session: 2026-05-03T00:00:00.000Z
Stopped at: AT ROBOT — pre-flight commits landed; ready to start [R]-track per .planning/TRACKS.md priority order (SETUP-01 first)
Resume file: .planning/TRACKS.md (live at-robot checklist)
