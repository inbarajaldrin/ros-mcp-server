# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-01)

**Core value:** Replace the failing `--insert` real-mode path with a force-compliant insert primitive that works reliably across every FMB1 part and is a one-config-file extension to any new part.
**Current focus:** Phase 1 — Foundation Setup + F/T Calibration

## Current Position

Phase: 1 of 6 (Foundation Setup + F/T Calibration)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-05-01 — Roadmap created, 6 phases mapped to 62 v1 requirements with 100% coverage

Progress: [░░░░░░░░░░] 0%

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

Last session: 2026-05-01
Stopped at: Roadmap created (6 phases, 62 requirements mapped, 100% coverage). Ready for `/gsd-plan-phase 1`.
Resume file: None
