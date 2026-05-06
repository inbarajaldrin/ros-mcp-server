---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: planning
stopped_at: Phase 4 COMPLETE (analyzer dashboard shipped — 5 segmentation methods + cross-episode tab). Phase 1 + 2 + 3 + 4 + 7 = 5 of 7 phases (71%). Phase 5 (Algorithm Derivation) is next; operator now at-robot.
last_updated: "2026-05-04T19:00:00.000Z"
last_activity: 2026-05-04 at-away dashboard session — built two-stage analyzer (preprocess.py + analyze_inserts.html) with 5 segmentation methods (M1 force, M2 kinematic, M3 energetic, M5 torque-motion, M7 object slip), per-shape baselines via iterative quiet-window pooling (10-25× larger sample base), Cross-Episode tab with feature scatter (18 axes) + contact-aligned compare. GPT adversarial review caught + fixed 6 issues including critical M3 frame mismatch (wrench tool0 × velocity base) and baseline pollution from "autonomous" traces self-triggering M1.
progress:
  total_phases: 7
  completed_phases: 5
  total_plans: 0
  completed_plans: 0
  percent: 71
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-01)

**Core value:** Replace the failing `--insert` real-mode path with a force-compliant insert primitive that works reliably across every FMB1 part and is a one-config-file extension to any new part.
**Current focus:** Phase 4 — Analyzer Dashboard (at-away). Build static HTML with Plotly + PapaParse against the 60 collected CSVs in `compliant_insertion_studio/logs/`. See `.planning/.continue-here.md` for the live handoff or `.planning/HANDOFF.json` for structured state.

## Current Position

**Phase 1, Phase 2, Phase 3, Phase 4, Phase 7 ✅ DONE** (5 of 7 phases complete = 71%).

- **Phase 1 (Foundation Setup + F/T Calibration)** — done 2026-05-03 (foundational F/T payload calibration recovered, smoke test passed: mass=2.11 kg, CoG ≈ origin-aligned, 32 mm into tool).
- **Phase 2 (Episode Wrapper + Telemetry Schema)** — done 2026-05-03 (WRAP-VERIFY end-to-end on u_brown: 15,177 telemetry samples, FSM walks PRE→HOVER→ZERO→ACTIVE→DONE).
- **Phase 3 (20-Episode FMB1 Data Collection)** — done 2026-05-03 (60 demos: u_brown 10 + u_orange 10 + line_green 20 + inverted_u_yellow 20; 7 autonomous + 53 assisted; all schema_v1.1).
- **Phase 4 (Analyzer Dashboard)** — done 2026-05-04 (5 segmentation methods, cross-episode scatter+compare, see `.planning/phases/04-analyzer-dashboard/04-SUMMARY.md`).
- **Phase 7 (Gripper URDF + RViz)** — done 2026-05-02.

Most recent work: 2026-05-04 — analyzer dashboard delivered with 5 segmentation methods (M1 force, M2 kinematic, M3 energetic with proper base-frame transform, M5 torque-motion, M7 object slip), iterative quiet-window baseline (95% of samples pass through), Cross-Episode tab with 18-feature scatter + 2-episode contact-aligned compare. GPT adversarial review caught the critical M3 frame mismatch + baseline pollution.

Next session goal: **Phase 5 — Algorithm Derivation + Per-Object Configs**. Operator is at-robot. Highest-leverage at-robot work for Phase 5: (a) review dashboard signatures to derive per-shape termination predicates, (b) write `configs/defaults.yaml` + per-shape YAMLs, (c) wire wrapper to read configs and auto-terminate, (d) test ≥5 consecutive autonomous successes per shape.

Progress: [██████████░░░░] 71% (5 of 7 phases formally complete)

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
| 7. Gripper URDF + RViz Visualization | 0 | — | — |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. **Decisions added 2026-05-03:**

- **Termination criterion stays Phase 5 work** — wrapper's only ACTIVE exit today is `timeout` (line 869 of `compliant_insert.py`). The 15,177 samples per WRAP-VERIFY run are the dataset Phase 5 will mine for force-saturation / Z-target / motion-stopped signals.
- **Tolerance widened from 2mm to 5mm** in `move_to_grasp.py:888` to match OnRobot RG2's inherent ~3-5mm grip-mode-16 overshoot. Researched 2 reference repos (takuya-ki/onrobot-rg, tonydle/OnRobot_ROS2_Driver) + official docs — confirmed firmware has NO precise positioning mode (only mode 1 grip, 8 stop, 16 grip_w_offset; both 1 and 16 are GRIP modes with overshoot).
- **OnRobot bridge fix kept**: `Circuit1`/`Circuit2` fields now visible in `/gripper_status` (was hidden, made the long-standing safety-latch bug invisible). Source: `~/Desktop/ros2_ws/src/onrobot_ros/onrobot_ros/rg_gripper.py`.
- **Discovered Modbus power-cycle for safety latch**: Write `unit=63 addr=0 value=2` triggers tool-power cycle that clears latched safety circuits. Documented in upstream Osaka-University-Harada-Laboratory/onrobot lib but missing from local custom_libraries/onrobot.py. NOT yet wired in (requires URCap re-attach via pendant STOP+PLAY after the reboot).
- **Reusable orchestrator script committed**: `compliant_insertion_studio/scripts/run_assembly_step.py` implements the canonical pick→rotate→place→regrasp→rotate→insert sequence per `ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json`. Step 13 of the canonical (the legacy `translate_object --insert`) is replaced by the new compliant_insert wrapper. Two entry modes: full sequence (object on table) and `--already-held` (skip pick/place/regrasp).

### Pending Todos

None.

### Blockers/Concerns

- None blocking Phase 3 entry. All upstream verifications complete.
- **Soft concern**: F/T payload calibration mild residual warnings (0.86 N / 0.062 Nm vs 0.5 N / 0.05 Nm "optimal"). Probably sensor warm-up. Operator can re-run after 10-30 min idle warmup if Phase 3 data shows orientation-dependent bias.
- **Known firmware quirk**: RG2 mode 16 has ±3-5mm grip overshoot. Width-based grasp checks must tolerate this (already widened in `move_to_grasp.py:888`). Mode 1 has tighter ±2mm but targets raw mechanism width — inconvenient caller semantics, no functional benefit at current tolerances.

### Roadmap Evolution

- 2026-05-02: Phase 7 added (Gripper URDF + RViz Visualization).
- 2026-05-03: WRAP-VERIFY validation extended to include the FULL canonical pick-rotate-place-regrasp-rotate-insert sequence (not just hover-zero-active). Driven by the discovery that the ground-truth `Assembly_fmb_assembly_1_results.json` requires this longer chain. New orchestrator script institutionalizes it for Phase 3 collection.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Wrapper validation | **Induced-failure tests** (SIGTERM/SIGABRT/SIGKILL mid-ACTIVE on 3 fresh wrapper runs) | Deferred to early Phase 3 — operator preferred to advance to data collection rather than burn another hour on signal-handling tests this session | 2026-05-03 |
| Bridge fix | Wire `restartPowerCycle` Modbus command into `onrobot_ros` (auto-clears safety latch when stuck-gripper detected) | Documented in HANDOFF, deferred until next time the safety latch trips | 2026-05-03 |
| Bug fix | The 6 wrapper-side bug fixes from this session (move_to_grasp tolerance, _run_hover mode arg, ANSI strip, controller-await timeout, re import, run_assembly_step.py, launch_camera.sh) are NOT yet committed — operator approval needed for the commit | Pending operator review | 2026-05-03 |

## Session Continuity

Last session: 2026-05-04T19:00:00.000Z
Stopped at: Phase 4 closed; operator back at-robot for Phase 5 entry.
Resume action: open dashboard at `http://localhost:8766/analyzer/analyze_inserts.html` (Cross-Episode tab) → derive per-shape termination predicates from feature scatter → write `configs/defaults.yaml` + per-shape YAMLs → wire wrapper → test on robot.
