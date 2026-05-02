# Execution Tracks — Away vs At Robot

Live list of what's actionable on each track. **Update whenever a task transitions states.**

This complements ROADMAP.md (which is phase-structured) and REQUIREMENTS.md (which is requirement-tagged). TRACKS.md answers the question "what should I work on right now given my current physical access to the robot?"

See `.planning/codebase/CONVENTIONS.md` for the two-track principle.

---

## Track legend

- `[N]` — **No-robot**: can be done from anywhere with the codebase. Code authoring, doc writing, RViz fake-hardware previews, USD→URDF conversion, dashboard scaffolding, dispatcher logic.
- `[R]` — **Robot-required**: needs the physical UR5e + ROS bringup connected to it. Calibration runs, demo collection, validation, smoke tests in the actual session pose.
- `[N→R]` — **Hybrid**: scaffolding done away (no-robot), final verification at-robot. Most code-heavy phases fall into this.

---

## Active session: AWAY FROM ROBOT (2026-05-02)

### Ready to work right now (no-robot track)

Priority order — top first:

1. **Phase 7 — Gripper URDF + RViz Visualization (entire phase, no-robot)**
   - USD → URDF conversion of the OnRobot RG2 (USD source: `~/Documents/isaac-sim-mcp/exts/ur5e-dt/`)
   - Integrate RG2 as fixed-joint child of `tool0` in UR5e URDF
   - RViz config update so gripper renders correctly
   - Replace `GRIPPER_CENTER_TOOL_OFFSET` usages with URDF FK lookups (Python primitives that consume the offset)
   - Validation against fake-hardware bringup (no real robot needed)
   - Documentation for swapping a different gripper in the future
   - **Why prioritize**: directly improves the calibration-pose preview workflow when operator returns to the robot — every Phase-1 calibration pose check from now on benefits.

2. **Phase 2 — Wrapper extension (no-robot half: WRAP-01..11, TELE-01..06 code)**
   - Extend the migrated `compliant_insertion_studio/wrapper/compliant_insert.py` from current ~250 LOC scaffold to the full PRE → HOVER → ZERO → ACTIVE → DONE/ABORT lifecycle
   - Implement enriched CSV schema (TELE-01..06): phase tag, event_marker, full TCP pose + target pose + per-axis errors, wrench, gripper_width, commanded_Fz, zero_event flag
   - Sidecar JSON metadata writer (per-episode, schema_version=1)
   - SIGUSR1/USR2/TERM/abort signal handlers with idempotent cleanup
   - Hands-off STEP-BACK gate logic + +1s post-zero drift check
   - HOVER pose joint-limit pre-check
   - Integration of `ft_smoke_test` as PRE-phase precondition (CAL-08 finalization)
   - **Verification deferred to at-robot session.**

3. **Phase 4 — Dashboard scaffolding (no-robot half: DASH-01..09 code, no real-data validation)**
   - Single static `compliant_insertion_studio/analyzer/analyze_inserts.html`
   - Plotly.js + PapaParse from CDN (cache to `analyzer/assets/` for offline)
   - File picker + auto-pair CSV/meta JSON loader
   - Single-episode view (F-vs-t, T-vs-t, Z-vs-t, F-vs-Z phase, 3D trajectory, event markers, phase bands, metadata panel)
   - Cross-episode overlay with first-contact time alignment
   - Per-object signature card (auto-stats from hands-off-window-restricted samples)
   - Decimation + scattergl for >50 episodes
   - **Test against synthetic CSVs generated locally** (not real Phase 3 data — we don't have any yet; this is "build the UI, validate with fake data, refine later with real data")

4. **Phase 6 — Dispatcher code (no-robot half: DISP-01..03 code)**
   - `compliant_insertion_studio/dispatcher/compliant_insert_episode.py`: config resolution, lazy YAML loading, MANUAL_GUIDED fallback for unknown objects
   - `shared/insert_config.py`: deep-merge helper
   - `configs/defaults.yaml` skeleton
   - Unit tests for the dispatcher (no ROS deps)
   - **Defer DISP-04 (`translate_object.py:1085` swap) and DISP-06 (ablation YAML completion) until Phase 5 is done with real data** — per CONVENTIONS §6, these are coupled to having actual configs to dispatch to.

### Blocked on robot — do NOT work these now

These require physical robot access. Tracked here so we know what's pending when operator returns.

- **CAL-03**: actually run `ft_calibration.py` on physical robot, recover real mass/CoG, paste `set_target_payload` into bringup, restart bringup once
- **CAL-04**: final pose set verification on physical hardware (RViz fake-hardware previews are still no-robot — but final "robot reaches all 8 poses without joint singularity at the actual current bringup config" is at-robot)
- **CAL-06..08 (smoke test)**: standalone code is done; running smoke test on real robot to confirm pass criteria still pending
- **SETUP-01**: `apt upgrade ros-humble-ur-robot-driver ros-humble-ur-msgs` (needs robot bringup restart window)
- **SETUP-02**: paste `set_target_payload(...)` into bringup launch (file edit is no-robot if we can locate the launch file remotely; the **restart + verification** is at-robot)
- **SETUP-03**: confirm new force-mode defaults (`gain_scaling=0.5`, `damping_factor=0.7`) work on real robot in test force-mode call
- **WRAP verification**: full wrapper end-to-end run on real robot, induced-failure cleanup tests
- **DATA collection**: 20 demo episodes (Phase 3 entirely)
- **DASH validation against real data**: drop the 20 real CSVs into the dashboard, eyeball each view
- **ALGO config tuning per object**: read dashboard signatures with operator + write each YAML + test wrapper auto-termination on known-good real setup
- **DISP-04..06 (integration tests)**: `translate_object.py:1085` swap + ablation YAMLs end-to-end through MCP
- **VAL-01..05 (validation runs)**: ≥5 consecutive autonomous successes per object, second-assembly part proof

---

## When the operator returns to the robot

Suggested at-robot session sequence (assuming Phase 7 + Phase 2 code + Phase 4 dashboard + Phase 6 dispatcher all done away-from-robot):

1. SETUP-01: `apt upgrade` + restart bringup (~5 min)
2. CAL-03 + CAL-04: run `ft_calibration.py` → paste `set_target_payload` → restart bringup → run smoke test (~15 min)
3. SETUP-02 + SETUP-03 verification on real robot (~5 min)
4. WRAP verification: run wrapper end-to-end on a known-good u_brown setup (~15 min including induced-failure tests)
5. **Phase 1 done.** ✅
6. Begin Phase 3 — DATA collection (~60–90 min for 20 episodes)
7. Return at-away to do Phase 4 dashboard validation against the real CSVs and Phase 5 algorithm derivation

Total at-robot time: ~2.5 hours for Phase 1 verification + Phase 3 collection. The rest is hybrid (away + at).

---

*Updated: 2026-05-02 after Phase 7 added and two-track pattern established.*
*Update mechanism: edit this file when transitioning a task between Ready / In-Progress / Done / Blocked, or when a phase changes which track items it owns.*
