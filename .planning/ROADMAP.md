# Roadmap: Compliant Insertion Studio

## Overview

Replace the failing `--insert` real-mode path on UR5e + Robotiq 2F-85 with a force-compliant insert primitive that works reliably on every FMB1 part and is a one-config-file extension to any new part. The journey follows a non-negotiable build-order interlock derived from three converging research threads: lock the data contract (schema + wrapper) and pre-collection hardware/F-T calibration first → collect 20 real demos → build the dashboard against those real shapes → derive per-object YAML configs and the termination criterion *from* the dashboard signatures → and only then flip the integration switch at `translate_object.py:1085` and validate generalization with ≥5 consecutive autonomous successes per object plus a second-assembly proof. The termination criterion is itself a project deliverable, not an input.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Foundation Setup + F/T Calibration** - Hardware/driver upgrade, folder scaffold, F/T smoke-test primitive, SOP docs
- [ ] **Phase 2: Episode Wrapper + Telemetry Schema** - Lock the CSV+meta data contract; wrapper owns full lifecycle
- [ ] **Phase 3: 20-Episode FMB1 Data Collection** - 5 demos x 4 FMB1 objects with mandatory failure-mode quota
- [ ] **Phase 4: Analyzer Dashboard** - Static HTML dashboard built against real Phase 3 traces
- [ ] **Phase 5: Algorithm Derivation + Per-Object Configs** - Universal algorithm + per-object YAMLs + termination criterion derived from data
- [ ] **Phase 6: Dispatcher Integration + Generalization Validation** - Single-line integration at translate_object.py:1085 + ≥5 consecutive autonomous successes per object + second-assembly proof
- [ ] **Phase 7: Gripper URDF + RViz Visualization** - Import OnRobot RG2 gripper geometry into UR5e URDF (USD→URDF), replace custom DH-based collision setup with URDF-driven RViz visualization + ros2_control collision checking

## Phase Details

### Phase 1: Foundation Setup + F/T Calibration
**Goal**: Hardware, drivers, folder scaffold, **foundational F/T payload calibration**, and F/T runtime smoke-test are locked-in so that no episode is collected against a contaminated baseline, with a wrong gripper payload (causing orientation-dependent bias), or into a wrong directory layout.
**Depends on**: Nothing (first phase)
**Requirements**: SETUP-01..07, CAL-01..10 (17 reqs total)
**Success Criteria** (what must be TRUE):
  1. **Foundational payload calibration**: `python3 compliant_insertion_studio/shared/ft_calibration.py --gripper-id <name>` runs ≥ 8 poses, recovers payload mass + CoG + 6-axis bias, writes versioned YAML to `configs/`, prints `set_target_payload(mass, cog)` line for paste into bringup. Operator pastes + commits + restarts bringup. After this, force mode behaves correctly across orientations (no per-pose re-zeroing needed).
  2. **Session-level smoke test**: `python3 compliant_insertion_studio/shared/ft_smoke_test.py` standalone reports pass/fail with per-axis residual bias < 2 N, |T| < 0.3 Nm, drift < 0.5 N/s.
  3. **Folder scaffold**: `compliant_insertion_studio/` tree exists with all subdirectories (wrapper, dispatcher, shared, configs, analyzer, logs, docs) and a top-level README describing standalone use.
  4. **Driver state**: `ur_robot_driver` reports version ≥ 2.13.0, `pyproject.toml` numpy pin reconciled with runtime numpy 2.2.6, `set_target_payload` documented as bringup-only single source of truth.
  5. **Migration done**: existing `primitives/compliant_insert.py` migrated to `compliant_insertion_studio/wrapper/compliant_insert.py` with import paths working.
  6. **SOP doc**: `docs/ft_calibration_sop.md` covers all three calibration layers (foundational / session / per-pose), warm-up window, when-to-re-run, what-to-do-on-failure, and the URCap Measure wizard alternative.
  7. **Gitignore**: excludes `compliant_insertion_studio/logs/insert_*.csv`, `*.meta.json`, and `_references/` so binary-ish telemetry and reference repos never enter the repo.
**If this phase fails**: Every demo collected afterward is suspect — contaminated F/T baseline, wrong driver version, wrong payload (causing orientation-dependent bias the wrapper's per-pose zero only partially masks), or missing folder layout means re-collection (the most expensive failure mode per build-order interlocks #1 and #2).
**Plans**: TBD

### Phase 2: Episode Wrapper + Telemetry Schema
**Goal**: The episode wrapper owns the full PRE → HOVER → ZERO → ACTIVE → DONE/ABORT lifecycle and writes the locked CSV + sidecar JSON schema. The wrapper is the only schema writer; getting it right now prevents re-collection later.
**Depends on**: Phase 1
**Requirements**: WRAP-01, WRAP-02, WRAP-03, WRAP-04, WRAP-05, WRAP-06, WRAP-07, WRAP-08, WRAP-09, WRAP-10, WRAP-11, TELE-01, TELE-02, TELE-03, TELE-04, TELE-05, TELE-06
**Success Criteria** (what must be TRUE):
  1. Operator can run `compliant_insertion_studio/wrapper/compliant_insert.py` standalone and see the robot move PRE → HOVER → ZERO → ACTIVE → safe-height → home with phase tags written to CSV every step
  2. SIGUSR1 toggles the `event_marker` column, SIGUSR2 logs a `zero_event` row and re-zeroes F/T mid-episode, SIGTERM exits cleanly as success, and an abort signal exits cleanly tagging outcome=abort — even if force_mode is already stopped or controller switch fails partway (idempotent cleanup verified by induced failure)
  3. Each episode produces `logs/insert_<object>_<YYYYMMDD_HHMMSS>.csv` (100 Hz, 7-phase tagged, full pose/target/wrench/gripper/commanded-Fz columns) AND a matching `.meta.json` with object/base/grasp_id, outcome, post-zero bias, hands-off window timestamps, schema_version=1, and free-text user_notes prompted at end
  4. ZERO phase enforces the operator-confirmed STEP-BACK gate before zeroing and logs a +1s post-zero drift sample; ZERO warns (not aborts) if any axis bias > 2 N
  5. Aborted episodes are preserved equally to successes (never deleted), ACTIVE phase verifies switch_controller transition completed via list_controllers poll (not just RPC success), and HOVER pose is rejected if IK lands at a joint-limit edge
**If this phase fails**: Schema is the data contract for every downstream consumer (dashboard, signature card, termination derivation). A schema bug discovered in Phase 4 means re-collecting all 20 episodes — the project's most expensive failure mode.
**Plans**: TBD

### Phase 3: 20-Episode FMB1 Data Collection
**Goal**: A 20-episode FMB1 demo dataset (5 episodes per object across u_brown, u_orange, line_green, inverted_u_yellow) with mandatory failure-mode coverage exists on disk in the locked schema, ready to feed the dashboard.
**Depends on**: Phase 2
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05
**Success Criteria** (what must be TRUE):
  1. `compliant_insertion_studio/logs/` contains >= 20 CSV+meta pairs covering all 4 FMB1 objects with >= 5 episodes per object
  2. Each object has at least one episode tagged as abort or intentional misalignment in its meta JSON (failure-mode quota)
  3. Every episode's meta JSON has a non-empty `user_notes` field describing what the operator did to guide the part in
  4. All episodes were collected on the physical UR5e + Robotiq + FMB1 base in real mode (no sim, no synthetic) and the hands-off window timestamps are populated in every meta JSON
**If this phase fails**: Dashboard built against an incomplete or biased dataset visualizes a fiction. Without a failure-mode library, the algorithm phase cannot derive failure-detection thresholds — interlock #4 violation.
**Plans**: TBD

### Phase 4: Analyzer Dashboard
**Goal**: A single static HTML dashboard auto-loads all CSVs + meta JSONs from `logs/` and surfaces the per-object signatures (median Fz, |T| peaks, lateral travel, descent duration) that the operator will type into per-object YAMLs in Phase 5.
**Depends on**: Phase 3
**Requirements**: DASH-01, DASH-02, DASH-03, DASH-04, DASH-05, DASH-06, DASH-07, DASH-08, DASH-09
**Success Criteria** (what must be TRUE):
  1. Operator opens `compliant_insertion_studio/analyzer/analyze_inserts.html` directly in a browser (no server, no build), drops the 20 CSVs + meta JSONs from Phase 3 via the file picker, and sees them auto-pair by basename
  2. Single-episode view renders F vs t (3 traces) + T vs t (3 traces) + Z vs t with synced cursors, F-vs-Z phase plot, 3D trajectory with target marker, event-marker vertical lines, phase-band background coloring, and the metadata panel
  3. Cross-episode overlay view filters by object + outcome and overlays traces time-aligned on the first-contact event (not absolute t=0)
  4. Per-object signature card auto-computes median Fz at success, |Tx|/|Ty| peak distributions, lateral travel during ACTIVE, and descent duration — using only the hands-off-window-restricted samples
  5. Dashboard remains responsive when loaded with synthetic >= 50-episode dataset (uses scattergl + decimation, not SVG scatter)
**If this phase fails**: Without the dashboard signature cards, parameter derivation in Phase 5 has no visual cross-episode overlay — operators would be reading raw CSVs in a Jupyter scratchpad. Interlock #5 violation: dashboard must precede algorithm derivation because termination criterion is derived BY LOOKING AT signatures.
**Plans**: TBD
**UI hint**: yes

### Phase 5: Algorithm Derivation + Per-Object Configs
**Goal**: One universal `compliant_insert.py` algorithm parameterized by per-object YAML configs in `configs/`. The termination criterion for each FMB1 object is derived FROM the Phase 4 dashboard signatures (not assumed) and recorded with rationale in the YAML — this is the project's research deliverable.
**Depends on**: Phase 4
**Requirements**: ALGO-01, ALGO-02, ALGO-03, ALGO-04, ALGO-05, ALGO-06, ALGO-07, ALGO-08, ALGO-09
**Success Criteria** (what must be TRUE):
  1. `configs/defaults.yaml` exists with universal values (gain_scaling=0.5, damping_factor=0.7, default termination = combined predicate force-absorbed AND motion-stopped with must_agree=true) and `configs/<object>.yaml` per FMB1 object overrides only the differences via deep-merge
  2. Each per-object YAML records its termination predicate (drawn from the schema combinator: force-absorbed, motion-stopped, z-reached, snap-fit Fz_peak_then_drop, multi-peg torque-band — combinable with AND/OR) AND a rationale comment naming which dashboard signature drove the choice
  3. The schema supports snap-fit `Fz_peak_then_drop` and multi-peg torque-band termination so future-FMB-2 parts that need them are a config diff, not a code change
  4. Operator can run `python wrapper/compliant_insert.py --config configs/u_brown.yaml` and watch the wrapper auto-terminate on the YAML-specified predicate (not on operator SIGTERM) on a known-good u_brown setup
  5. Mid-insert re-classification is structurally disallowed — once a config is loaded, parameters are fixed for the episode (verified by code review of the wrapper)
**If this phase fails**: Configs and termination criteria are the BRAIN of the system. Without them, integration in Phase 6 ships an empty-config dispatcher that silently breaks every `--insert` ablation — interlock #6 violation. Termination criterion is itself a Phase deliverable, not an input.
**Plans**: TBD

### Phase 6: Dispatcher Integration + Generalization Validation
**Goal**: The dispatcher routes `translate_object.py:1085` to the new compliant-insert subsystem, and the system passes its validation gate: >= 5 consecutive autonomous successes per FMB1 object + at least one part from a *second* assembly inserted using only its YAML config.
**Depends on**: Phase 5
**Requirements**: DISP-01, DISP-02, DISP-03, DISP-04, DISP-05, DISP-06, DISP-07, VAL-01, VAL-02, VAL-03, VAL-04, VAL-05
**Success Criteria** (what must be TRUE):
  1. The single-line edit at `primitives/translate_object.py:1085` points at `compliant_insertion_studio/dispatcher/compliant_insert_episode.py`, the existing `--insert` real-mode flow works end-to-end via `server_p3.py` MCP tool surface (translate_object signature unchanged), and the existing `recording_dryrun_real_u_brown.yaml` and `recording_dryrun_real.yaml` ablation YAMLs complete the insert leg autonomously
  2. Each FMB1 object passes >= 5 consecutive autonomous insert successes with no manual interventions and no parameter tweaks between attempts (logged to `logs/validation_*`)
  3. At least one part from a second assembly is successfully inserted using only its YAML config — no algorithm changes — and passes >= 3 consecutive autonomous successes
  4. Calling the dispatcher with an unknown object name drops into MANUAL_GUIDED fallback (full 6-DOF compliance, operator-terminated only) and the JSON output includes a `hint` field with the exact `--collect` command for the LLM to surface to the operator
  5. `compliant_insertion_studio/README.md` and `docs/tune_a_new_part.md` document standalone install (copy folder, edit one path in host's translate_object equivalent) and the step-by-step "tune a new part in 30 min" workflow with a timing budget
**If this phase fails**: This is the integration step that proves the project's core value claim. Failing here means either the integrated path silently breaks ablations (DISP failure) or the system only "works once" (VAL failure — Pitfall #20, the project's documented top deployment failure mode).
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation Setup + F/T Calibration | 0/TBD | Not started | - |
| 2. Episode Wrapper + Telemetry Schema | 0/TBD | Not started | - |
| 3. 20-Episode FMB1 Data Collection | 0/TBD | Not started | - |
| 4. Analyzer Dashboard | 0/TBD | Not started | - |
| 5. Algorithm Derivation + Per-Object Configs | 0/TBD | Not started | - |
| 6. Dispatcher Integration + Generalization Validation | 0/TBD | Not started | - |
| 7. Gripper URDF + RViz Visualization (decoupled — can pull forward) | 0/TBD | Not started | - |

### Phase 7: Gripper URDF + RViz Visualization
**Goal**: Import the OnRobot RG2 gripper geometry into the UR5e URDF so RViz shows the actual gripper mounted on the flange, and ros2_control collision-checks the gripper natively. Replaces the custom DH-based `GRIPPER_CENTER_TOOL_OFFSET` (229 mm hardcoded) with a proper URDF-driven approach. Source geometry: existing USD asset in the Isaac Sim extension (`isaac-sim-mcp/exts/ur5e-dt/`) — convert USD → URDF, link to the UR5e URDF as a fixed-joint child of `tool0`, validate visually in RViz.
**Depends on**: None — independent of the calibration/wrapper/data pipeline. **Per CONVENTIONS §6 phase boundaries are guidance not gates: this phase is decoupled from Phases 1–6 and can be pulled forward to immediately benefit RViz preview during calibration pose verification.** Recommended sequencing: do this BEFORE the first real-hardware Phase 1 calibration run so collision visualization is live during pose previews.
**Requirements**: TBD (will be added when phase is planned via `/gsd-plan-phase 7`). Anticipated coverage:
  - USD → URDF conversion script for the OnRobot RG2 (likely via `usd_to_urdf` tooling or hand-written from USD geometry)
  - Combined URDF that adds the RG2 as a fixed-joint child of UR5e's `tool0` link
  - RViz config with gripper rendered correctly
  - `ros2_control` / MoveIt collision-check integration so the gripper is part of the planning scene
  - Removal (or deprecation) of `GRIPPER_CENTER_TOOL_OFFSET` from `primitives/shared/config.py` and downstream usages, replaced by URDF FK lookups
  - Documentation in `compliant_insertion_studio/docs/` for swapping in different grippers in the future
**Success Criteria** (what must be TRUE — to be refined during planning):
  1. RViz preview of any robot pose shows the gripper geometry attached at `tool0`
  2. The fake-hardware bringup includes the gripper as a part of the URDF (no manual TF publish)
  3. Calibration / wrapper code that previously used `GRIPPER_CENTER_TOOL_OFFSET` reads gripper-tip TF from URDF instead
  4. A new gripper assembly can be swapped in with a single URDF/xacro edit (no Python changes)
**If this phase fails**: We continue with the custom DH offset approach. The compliance pipeline still works but: (a) RViz previews don't show gripper collisions, requiring custom DH-based preflight scripts (annoying to maintain — operator's stated complaint), (b) swapping grippers means editing constants in multiple Python files instead of one URDF.
**Plans**: TBD
