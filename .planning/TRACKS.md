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

## Active session: PAUSED 2026-05-03 (clean handoff)

### Done this session (2026-05-03)

- ✅ **All 4 horizontal axes empirically verified** via verify_baselink_motion.py (commands base_link direction → robot moves in that direction). Wrapper's tool0_controller → base_link wrench transform confirmed correct.
- ✅ **Workspace convention corrected**: empirically `+X = robot's RIGHT` (operator's verbal `+X = LEFT` was wrong). +Y = forward, +Z = up unchanged.
- ✅ **Wrapper frame-conversion bug fixed**: `_start_force_mode()` now uses `task_frame.header.frame_id="base"` + explicit base_link→base sign flip. (Previously `frame_id="base_link"` silently auto-transformed to `base` which is `R_z(180°)` from base_link, flipping commanded X/Y.) For Z-only insertion this was a no-op; for any future X/Y commanded force it would have been a latent bug. Now correct end-to-end.
- ✅ **HOME_JOINTS** introduced: `[+90°, -90°, +90°, -90°, -90°, 0°]` joint-space "tidy home" matching workspace orientation. `move_home.py --joint-space` flag wires it. Calibration uses it for clean entry/exit.
- ✅ **CALIBRATION_POSES_RAD rotated +π/2 in shoulder_pan** to match workspace; LSQ math invariant under rotations about base Z.
- ✅ **CAL-03 done**: F/T payload calibration ran successfully on real robot. **mass = 2.1109 kg, CoG = [-0.0032, +0.0031, -0.0318] m.** Residuals 0.86 N / 0.062 Nm (mild warn but rank 10/10, condition number 16.3). Operator pasted `set_target_payload(2.1109, [-0.0032, 0.0031, -0.0318])` into bringup.
- ✅ **CAL-04 done**: All 8 calibration poses (workspace-rotated) reachable on real hardware end-to-end with `--move-duration-s 10` (no protective stops).
- ✅ **CAL-06..08 done**: ft_smoke_test PASS post-payload. Bias F max 0.33 N (threshold 2.0), torque max 0.006 Nm (threshold 0.3), drift max 0.007 N/s (threshold 0.5).
- ✅ **launch_robot.sh + close_robot.sh** helpers committed; documented stale-play bringup-restart trap and Local-mode pendant requirement.
- ✅ **Phase 7 launch supports real hardware** (`use_fake_hardware:=false robot_ip:=...`); RG2 visualization in RViz works in BOTH modes.
- ✅ **ursim_cli** surfaces clear hint when Local-mode blocks an action command (play/stop/load/power_on/etc.).

### Ready to work right now (at-robot track) — single remaining task before Phase 3

1. **WRAP-VERIFY** — full wrapper end-to-end on a known-good u_brown setup (~15 min including induced-failure tests)
   - **Operator must physically place u_brown on FMB1 base + grasp it before this can run.** That's the only thing blocking it.
   - Then: `bash compliant_insertion_studio/scripts/launch_robot.sh real --rviz`
   - Pendant: STOP+PLAY (URCap re-link after bringup restart — see launch_robot.sh "NEXT STEPS" block for why)
   - Wrapper: `python3 -m compliant_insertion_studio.wrapper.compliant_insert --object-name u_brown --base-name fmb1_base --grasp-id 0 --current-object-orientation <qx qy qz qw> --use-default-base-position --fz 3.0 --step-back prompt --no-prompt-notes`
   - Verify FSM walks PRE → HOVER → ZERO → ACTIVE → DONE; eyeball CSV at `compliant_insertion_studio/logs/insert_u_brown_<ts>.csv`
   - Heed the "TARGET-POSE LIMITATION" WARN line — `dx/dy/dz` columns are not meaningful until Phase 5
   - Induced-failure tests: SIGTERM mid-ACTIVE, SIGABRT mid-ACTIVE, SIGKILL mid-ACTIVE → verify cleanup reaches safe-state every time
   - F/T sign-convention spot check: push gripper down by hand → confirm `fz` goes negative in CSV (gravity in tool frame)
   - Pass = Phase 2 [R]-half done. Phase 1 + Phase 2 + Phase 7 then all fully complete.

2. **Phase 7 visual sign-off** (~2 min, can be done anytime alongside any at-robot work)
   - RViz already starts with `launch_robot.sh real --rviz` — just confirm gripper renders at `tool0` in current pose. Already validated against fake-hardware.

After WRAP-VERIFY: **Phase 1 + Phase 2 + Phase 7 all fully complete.** Ready to start Phase 3 (DATA collection) — same at-robot session if energy permits, or next session.

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

*Updated: 2026-05-03 11:55 UTC — clean session pause after Phase 1 [R]-half completion. WRAP-VERIFY is the only at-robot task remaining before Phase 3 data collection.*
*Update mechanism: edit this file when transitioning a task between Ready / In-Progress / Done / Blocked, or when a phase changes which track items it owns.*
