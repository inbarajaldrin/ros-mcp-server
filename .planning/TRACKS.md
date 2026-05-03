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

## Active session: AT ROBOT (2026-05-03)

### Done since last TRACKS.md update

- ✅ **Phase 7 — Gripper URDF + RViz Visualization** shipped 2026-05-02 (USD→OBJ+MTL pipeline + dual-URDF + static TF + dual-RobotModel RViz). Visual sign-off on real bringup still outstanding (item 4 below).
- ✅ **Phase 2 — Wrapper extension + telemetry schema** shipped 2026-05-02 (~700 LOC wrapper + 269-line SCHEMA.md + 130-LOC schema_v1 + 235-LOC telemetry + 116-LOC HOVER subprocess + 338-LOC synthetic test, 34/34 synthetic checks pass). End-to-end real-robot verification still outstanding (item 5 below).
- ✅ Gripper-name fix sweep (Robotiq 2F-85 → OnRobot RG2) across 10 docs/code files.
- ✅ Pre-existing dirty files committed: `primitives/shared/config.py` refactor, `utils/ursim_cli.py` bootup/shutdown_seq.

### Ready to work right now (at-robot track)

Priority order — top first.

1. **SETUP-01: apt upgrade UR driver** (~5 min, needs bringup restart window)
   - `sudo apt upgrade ros-humble-ur-robot-driver ros-humble-ur-msgs`
   - Reason: 2.13.0 ships F/T frame bugfix needed before calibration data collection

2. **CAL-03 + CAL-04: foundational F/T payload calibration** (~15 min)
   - Run `python3 compliant_insertion_studio/shared/ft_calibration.py --gripper-id onrobot_rg2_with_camera`
   - Paste resulting `set_target_payload(mass, cog)` line into bringup launch
   - Restart bringup once
   - Re-verify across 8 poses (CAL-04)

3. **CAL-06..08: ft_smoke_test on real robot** (~5 min)
   - Run `python3 compliant_insertion_studio/shared/ft_smoke_test.py` once after CAL-03 paste
   - Confirms session-level bias < 2 N, |T| < 0.3 Nm, drift < 0.5 N/s
   - Pass = Phase 1 [R]-half done

4. **Phase 7 visual sign-off** (~5 min)
   - Open RViz with the new launch (`compliant_insertion_studio/launch/ur5e_with_rg2.launch.py`)
   - Confirm RG2 renders at `tool0` correctly with the calibration pose set on real bringup
   - Verifies the URDF-side work was right against fake-hardware (only at-robot validation outstanding)

5. **WRAP verification: full wrapper end-to-end on a known-good u_brown setup** (~15 min including induced-failure tests)
   - `python3 -m compliant_insertion_studio.wrapper.compliant_insert --object-name u_brown --base-name fmb1_base --grasp-id 0 --current-object-orientation 0 0 0 1 --use-default-base-position --fz 3.0`
   - Verify FSM walks PRE → HOVER → ZERO → ACTIVE → DONE
   - Heed the new "TARGET-POSE LIMITATION" WARN line — `dx/dy/dz` columns are not meaningful until Phase 5
   - Induced-failure tests: SIGTERM mid-ACTIVE, SIGABRT mid-ZERO, SIGKILL mid-ACTIVE → verify cleanup reaches safe-state every time
   - Sign-convention spot check: push gripper down by hand → confirm `fz` goes positive in CSV
   - Pass = Phase 2 [R]-half done

After items 1–5: **Phase 1 + Phase 2 + Phase 7 all fully complete.** Ready to start Phase 3 (DATA collection) the next at-robot session, or pivot back to [N] tracks (Phase 4 dashboard, Phase 6 dispatcher) for at-away sessions.

### Pending — needs robot but blocked on items 1–5 above

- **DATA collection** (Phase 3 entire): 20 demo episodes, 5 per FMB1 object × 4 objects (~60–90 min)
- **DISP-04..06 (integration tests)**: `translate_object.py:1085` swap + ablation YAMLs end-to-end through MCP — also blocked on Phase 5 configs existing
- **VAL-01..05 (validation runs)**: ≥5 consecutive autonomous successes per object, second-assembly part proof

### Don't waste at-robot time on these (do at-away)

These are pure code, no robot needed — would burn at-robot time unnecessarily.

- **Phase 4 — Dashboard scaffolding** (DASH-01..09): build away from robot against synthetic CSVs first; real-data validation comes after Phase 3 collection
- **Phase 6 — Dispatcher code** (DISP-01..03): standalone code, no robot needed
- **Phase 5 — Algorithm derivation**: needs Phase 3 collected data first, so blocked at-away anyway

---

## When the operator next leaves the robot

Suggested at-away sequence (after items 1–5 above are all green):

1. **Phase 4 dashboard scaffolding** (DASH-01..09): single static HTML against synthetic CSVs (~half a session)
2. **Phase 6 dispatcher code** (DISP-01..03): config resolution + MANUAL_GUIDED fallback (~half a session)

Both unblock the at-robot DATA collection + algorithm derivation work that follows.

---

*Updated: 2026-05-03 — at-robot session start, Phase 7 + Phase 2 [N]-halves shipped 2026-05-02.*
*Update mechanism: edit this file when transitioning a task between Ready / In-Progress / Done / Blocked, or when a phase changes which track items it owns.*
