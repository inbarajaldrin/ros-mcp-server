# Detailed Rationale + Historical Architecture Notes

> Moved from `CLAUDE.md` 2026-05-06 to keep the project memory file lean. Most of the "Architecture" content here describes the **Phase 5 iterative-loop workflow** which is now superseded by the autonomous SEARCH director (see `AUTONOMOUS_INSERTION_METHODOLOGY.md`). Kept for historical context — useful if you're reading CSVs from May 2026 or earlier where the legacy spiral / Mode B was active.
>
> Read this only if you need WHY for a specific decision; the binding rules live in CLAUDE.md and the methodology lives in `AUTONOMOUS_INSERTION_METHODOLOGY.md` + the SKILL.md.

## Detailed rationale (humans only — auto-summarizer drops most of what follows)
### Research before code — procedure
- Search 2–4 query variations (vendor docs, GitHub, papers, community forums)
- Identify candidate repos / articles
- Clone repos into `_references/repos/`, save articles to `_references/articles/`
- Read enough to make an honest copy/modify/write-fresh decision per piece
- Only then write code
### Phase boundaries — finish coupled work together (rationale)
- Project-defined deviation from conventional GSD
- If requirements from later phases are tightly coupled to current work, finish them together and mark them complete now, regardless of which phase they nominally belong to
- Update REQUIREMENTS.md traceability when this happens to record where each requirement actually completed
- ROADMAP.md phase-completion still requires *all* of that phase's owned requirements to be done — but those done early count
- Do **not** intentionally pull future work forward to "save time" — only collapse when the coupling is real and finishing-now is materially cheaper than finishing-later
- This avoids the conventional-GSD waste of revisiting touched files in a later phase to make small additions that could have been done in one pass
### Folder structure rationale
- All Compliant Insertion Studio code lives under `compliant_insertion_studio/` (see REQUIREMENTS.md → "Project Layout" for full tree)
- The entire subsystem is droppable into other robotics projects with a one-line edit at the host's `translate_object` equivalent
- Implication: when adding new code for this project, default to placing it under `compliant_insertion_studio/`. Only put code in `primitives/` (or elsewhere) if you're modifying *existing* primitives or fixing a bug in the host repo unrelated to this project's deliverables
### Calibration hierarchy detail
- **Foundational** (per-mount): payload mass + CoG via `set_target_payload()` after running calibration script (or URCap Measure wizard). If this is wrong, nothing downstream is right — orientation-dependent bias persists no matter how often you `zero_ftsensor`
- **Session-level**: F/T smoke test confirms the sensor's bias is close to zero in a known pose. If this fails, foundational is suspect
- **Per-pose**: `zero_ftsensor` before force mode subtracts residual bias. Fast, cheap, but does NOT substitute for foundational
### Inline default rationale
- The GSD `workflow.research / plan_check / verifier / code_review / pattern_mapper` toggles are all `false` in `.planning/config.json` to enforce inline-default
- If you find yourself wanting to spawn an agent for routine work, ask first
- The 4-researcher init at project start was right because the 4 surveys were genuinely independent (each researcher knew nothing about the others), parallel (5 min instead of 20), and produced artifacts (the .md files) rather than intermediate chatter that would have polluted the main context
### Safety conventions detail
- Operator's pendant preference is Local mode (manual control retained)
- Don't call `dashboard_client/recover`, `dashboard_client/play`, or other Remote-mode services
- Recovery from protective stops is manual on the pendant
- After protective stop, operator clears it; only then can your code resume
- Force-mode wrench limits exist because gear/part/fixture damage is the binding constraint, not benchmark performance
- SIGTERM cleanup is operator trust. If the wrapper's cleanup is unreliable, operators will hesitate to use SIGTERM and instead try to interrupt other ways (which may leave the robot in force mode)
### Commit discipline detail
- Pre-existing-codebase fixes (bug fixes outside the project's scope) get their own commit, separate from project deliverables
- Project commits should reference requirement IDs they complete (e.g., `feat(WRAP-01..05): episode wrapper PRE/HOVER/ZERO phases`)
- The `_references/` folder is gitignored — never committed
- Telemetry logs are gitignored — never committed
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

### Phase 5 iterative-loop workflow (live as of 2026-05-04)

Three-script chain for tuning the insert primitive on one object:

```
loop_iterate.py  →  iterate_insert.py  →  compliant_insert.py (wrapper)
   harness            one-attempt              FSM: PRE→HOVER→ZERO→ACTIVE→DONE/ABORT
   N consecutive       delegates setup to
   successes           run_assembly_step --setup-only
                       then launches wrapper
```

Canonical command (one object, ≥5 consecutive successes):

```
python3 -m compliant_insertion_studio.scripts.loop_iterate \
  --object-name u_orange --base-name base1 --grasp-id 1
```

Tighter target / specific first held quat:

```
python3 -m compliant_insertion_studio.scripts.loop_iterate \
  --object-name u_orange --base-name base1 --grasp-id 1 \
  --target-success-count 3 \
  --first-held-quat 0.0062 -0.6494 0.7604 0.0055
```

`iterate_insert.py` (one attempt) flow:
1. Capture base1 world pose from `/objects_poses_real` BEFORE grasp (camera unobstructed by gripper)
2. Run `run_assembly_step.py --setup-only` → executes pick → rotate → place → regrasp → rotate, prints `PARSED HELD_QUAT [...]` on stdout
3. Launch the wrapper with `--base-world-pose X Y Z QX QY QZ QW` (CAD-derived predicted target) and `--use-default-base-position` and the held quat
4. Wrapper FSM runs to DONE (predicate met) or ABORT (timeout / max-corrections / safety)

`loop_iterate.py` chains across attempts: parses `PARSED HELD_QUAT` from attempt 1's stdout and passes it to attempts 2+ as `--already-held --held-quat <captured>`. Attempt 1 does the full canonical sequence (pick → rotate → place → regrasp → rotate). Attempts 2..N do **rotate-only** via `run_assembly_step --already-held --setup-only` (jumps straight to step 12 = rotate_object). **The gripper stays closed across every iteration** — no release, no regrasp. But **rotate_object IS called every attempt**, re-snapping EE orientation to canonical face-down.

Why the every-attempt rotate is critical: the wrapper's cleanup retracts holding whatever EE orientation the prior insert left, which is typically tilted by a few degrees because peg/slot tolerance lets the held part rotate slightly during force-mode wedging. Without re-rotating, the next attempt starts tilted (we measured a 4.5° baked-in tilt on 2026-05-04 iter 7 because we wrongly skipped rotate) and burns 20+ corrections fighting the angle instead of just inserting. Calling `rotate_object` is cheap (~3 s, one trajectory) and idempotent (snaps to nearest fold-equivalent of canonical).

Why every release+regrasp cycle is bad: (a) wastes ~30 s, (b) the place-down disturbs the part on the table, (c) `run_assembly_step` opens the gripper as its first action when NOT in `--already-held` mode, dropping any held part wherever the robot currently is. We hit this on 2026-05-04 (iter 5 + iter 6 setups failed at `move_to_grasp` because the part had drifted). Keeping the part held end-to-end + re-rotating each attempt is the canonical pattern.

### Wrapper key APIs

- `cad_lookup.predict_tcp_at_seat(base, object, grasp_id, base_world_xyz, base_world_quat_xyzw, flange_offset_m=0.2286)` — full chain `T_world_base ∘ T_base_object_seat ∘ T_object_grasp_point ∘ T_grasp_point_tcp`. Reads `~/Documents/aruco-grasp-annotator/data/fmb_assembly1.json` + `grasp_points/<obj>_grasp_points.json`.
- `--config <yaml>` — defaults to `compliant_insertion_studio/configs/defaults.yaml` (universal, shape-agnostic). Per-shape YAMLs are deleted; the termination predicate is derived from CAD chain, not shape geometry.
- `--use-default-base-position` — required when `--base-position` is not passed; computes hover/insert targets from CAD chain.

### Cancel-safety chain (REQUIRES idempotent process-group cleanup)

```
Operator Ctrl+C
  → loop_iterate SIGTERM handler
    → os.killpg(iterate_insert pgid, SIGTERM)
      → iterate_insert SIGTERM handler
        → os.killpg(wrapper pgid, SIGTERM)
          → wrapper run_done():
              stop_force_mode → switch to scaled_joint_trajectory_controller
              → _await_controller_active (gates the next step)
              → python3 -m primitives.move_to_safe_height
              → exit
```

Each layer wraps the child in `os.setsid` (`preexec_fn=os.setsid` in `subprocess.Popen`) so SIGTERM hits the whole pgroup. The wrapper's `_await_controller_active` is what makes the safe-height subprocess actually run — without it, the script-mode invocation can fail silently (ModuleNotFoundError swallowed) leaving force_mode_controller active.

### Engagement gate must allow z-drop dominance (2026-05-04 PM)

The CAD-derived predicted xy can be 15–20 mm off the actual seat (observed on iter v70: peg seated at xy=(0.0301,−0.3598), CAD-predicted (0.0257,−0.3560), error ~17 mm even after spiral recentering). A strict 6 mm `engagement_dist_thresh_m` then wrongly rejects a peg that's clearly inside the slot. Add a z-drop dominance shortcut in ENTRY_SETTLE: if `surface_z − tcp_z ≥ engaged_z_drop_dominant_m` (default 20 mm), accept engagement regardless of CAD-xy distance. Operator-demo full descent is 25–30 mm, so 20 mm is a safe "definitely inside" threshold. The dist gate still catches metastable low-force states where peg sits on top of rim (z_drop ≪ 20 mm).

### Contact xy ≠ Seat xy (2026-05-04 evening)

Critical late-session finding (likely root cause of an entire 12-hour failed iteration session): the **first-contact xy** (where peg first hits Fz>5N) is NOT the **seat xy** (where peg ends up after descending through chamfer). For u_orange today: contact at (+0.0308, -0.3554), seat at (+0.0341, -0.3635). 5.7mm offset in X, 8mm in Y. The first contact lands on the **rim**, the seat is the **slot center**. **`hole_xy_prior` should be the SEAT xy**, not the contact xy. Using contact xy as the spiral target points the search at the rim, where the peg can't drop in.

For each operator demo, extract BOTH values; use SEAT xy as the prior:
- contact_xy = first ACTIVE row with `fz > 5N`
- seat_xy = LAST ACTIVE row (or row where descent rate drops to 0 with z near predicted seat z)

### State-independent global seat detector (2026-05-04 evening)

The FSM's local `_motion_stopped_first_t` resets on every state transition (ENTRY_SETTLE↔FIND_HOLE↔INSERT). With our gate flapping under marginal conditions, this means a SEATED peg can sit motionless for 60+ seconds without ever accumulating the 0.75s sustain needed for the local seat predicate to fire. **Add a state-independent seat detector that runs every tick regardless of FSM state** (in `update()` before state dispatch): if `surface_z - tcp_z >= 20mm AND |dz/dt| < 0.5mm/s AND tilt < 5°` for 1.0s sustained, declare exit_done. Verified offline on FAIL_INSERT CSV: would have correctly fired at t=74.75s when peg had been seated for 60s.

### Cross-run signal analysis findings (2026-05-04 evening)

Comparing 6 successful vs 122 failed u_orange attempts at each timepoint post-contact:
- **t=1.0s: success has F_lat=3.5N + xy_excursion=0.9mm; failure has F_lat=0.6N + xy=0.3mm.**
- **t=1.5s: success F_lat=4.7N, xy=2.1mm; failure 0.8N, 0.4mm.**
- t=2.0s: success z_drop=2.2mm; failure 0.16mm.
- Successes have ~5N sustained F_lat for 1-1.5s causing 2mm xy excursion, then peg drops at t=2s.
- Failures don't accumulate this lateral motion → never engage chamfer.

Implication: an `INITIAL_PRESS` phase commanding 5N sustained for 1.5s in a fixed direction is the missing mechanism. The spiral PD only generates 0.5-2N sustained because PD is reactive — error stays small. Operator's hand applies sustained directed push that algorithm doesn't replicate. Implemented in FSM as `find_hole_press_*` knobs.

### Anti-pattern: tuning parameters without first analyzing data

12-hour session in 2026-05-04 wasted hours on parameter iteration (16+ versions v82-v97) before doing the analytical work that should have come first. **Rule for next sessions: write a CSV cross-run analyzer FIRST. Iterate parameters only after the data has revealed the actual failure mechanism.** Monitor commands are for state-transition events, not for diagnosing why peg isn't moving.

### Mode A vs Mode B (Phase 5)

- **Mode A** = pure compliance with universal termination predicate:
  `motion_stopped AND tcp_z_reached_predicted (CAD-derived) AND descended_post_contact ≥ 25 mm`, sustained 1 s.
- **Mode B** = active correction triggered by stuck-detection:
  `net z-descent over 2 s window < 0.5 mm/s AND smoothed Fz > 6 N`, sustained 2 s. State machine: NORMAL → CORRECTING → COOLDOWN → NORMAL. Up to `max_corrections=12` per episode, then ABORT.

### Mode B action-type research (2026-05-04, /tmp/gpt-task-result-123444.txt)

We tested 4 iterations of "push harder" Mode B (downward Fz=-9N + lateral force pokes 6N + counter-torque 0.3Nm). All 4 ABORT at 12 corrections. The peg is geometrically wedged on the rim; pushing deeper deepens the wedge. **Action type matters more than action magnitude.**

Right action for peg-on-rim wedge:
1. Retract 0.5–1.5 mm to unload static friction
2. Drop Fz to -2 to -4 N during search (NOT -9 N)
3. Run continuous spiral search 1.5–2.0 s, radius 0.25 → 1.0 mm
4. Lower force-mode gain during search: `gain_scaling=0.4–0.7` (vs nominal 1.0), `damping_factor=0.15–0.30` (vs nominal 0.7)
5. Restore nominal Fz/gains after correction

Sources GPT cited: Chhatpar & Branicky (2001) — spiral pitch = clearance; FANUC force-control manual — switch to search when error > chamfer + clearance/2; Robotiq spiral-search practice; Tang et al. (2016) on three-point contact.

### Wrench-feature hierarchy (verified empirically on iter-4 dataset)

- **Wrench frame is `tool0_controller`**, NOT `base_link`. Telemetry CSV column `wrench_frame_id` confirms. Direction-aware features (r_cop, F_lat in operator's intuitive frame) MUST be computed in tool frame, then optionally transformed via TCP quaternion if needed for base-frame analysis.
- **r_cop = ‖(-Ty/Fz, Tx/Fz)‖** is the missing direction signal. On iter-4 ACTIVE phase: median r_cop = 5.5 mm, mean COP vector = (-1.0, -5.4) mm (consistent direction, not noise) — for a ~22 mm peg, that's rim-contact-scale lever arm. Raw `Tx, Ty` look "tiny" (~0.05 Nm) until you normalize by Fz.
- **Counter-residual is geometrically wrong for wedges**. When peg is wedged at (-X,-Y) corner, wrist sensor reads (+X,+Y) because the part is being pressed into the (+X,+Y) rim edge. Counter-residual direction = AWAY from target. **Use TOWARD-TARGET direction** (CAD-derived `target_xy − tcp_xy`) for force corrections.
- **Stuck-detection: net z-descent over 2 s window**, NOT instantaneous v_z. Smooth Fz over 0.5 s window before threshold-checking — force-mode oscillation makes instantaneous fz dip below threshold momentarily, breaking sustain timers.

### Subprocess invocation rule
## Architecture

### Phase 5 iterative-loop workflow (live as of 2026-05-04)

Three-script chain for tuning the insert primitive on one object:

```
loop_iterate.py  →  iterate_insert.py  →  compliant_insert.py (wrapper)
   harness            one-attempt              FSM: PRE→HOVER→ZERO→ACTIVE→DONE/ABORT
   N consecutive       delegates setup to
   successes           run_assembly_step --setup-only
                       then launches wrapper
```

Canonical command (one object, ≥5 consecutive successes):

```
python3 -m compliant_insertion_studio.scripts.loop_iterate \
  --object-name u_orange --base-name base1 --grasp-id 1
```

Tighter target / specific first held quat:

```
python3 -m compliant_insertion_studio.scripts.loop_iterate \
  --object-name u_orange --base-name base1 --grasp-id 1 \
  --target-success-count 3 \
  --first-held-quat 0.0062 -0.6494 0.7604 0.0055
```

`iterate_insert.py` (one attempt) flow:
1. Capture base1 world pose from `/objects_poses_real` BEFORE grasp (camera unobstructed by gripper)
2. Run `run_assembly_step.py --setup-only` → executes pick → rotate → place → regrasp → rotate, prints `PARSED HELD_QUAT [...]` on stdout
3. Launch the wrapper with `--base-world-pose X Y Z QX QY QZ QW` (CAD-derived predicted target) and `--use-default-base-position` and the held quat
4. Wrapper FSM runs to DONE (predicate met) or ABORT (timeout / max-corrections / safety)

`loop_iterate.py` chains across attempts: parses `PARSED HELD_QUAT` from attempt 1's stdout and passes it to attempts 2+ as `--already-held --held-quat <captured>`. Attempt 1 does the full canonical sequence (pick → rotate → place → regrasp → rotate). Attempts 2..N do **rotate-only** via `run_assembly_step --already-held --setup-only` (jumps straight to step 12 = rotate_object). **The gripper stays closed across every iteration** — no release, no regrasp. But **rotate_object IS called every attempt**, re-snapping EE orientation to canonical face-down.

Why the every-attempt rotate is critical: the wrapper's cleanup retracts holding whatever EE orientation the prior insert left, which is typically tilted by a few degrees because peg/slot tolerance lets the held part rotate slightly during force-mode wedging. Without re-rotating, the next attempt starts tilted (we measured a 4.5° baked-in tilt on 2026-05-04 iter 7 because we wrongly skipped rotate) and burns 20+ corrections fighting the angle instead of just inserting. Calling `rotate_object` is cheap (~3 s, one trajectory) and idempotent (snaps to nearest fold-equivalent of canonical).

Why every release+regrasp cycle is bad: (a) wastes ~30 s, (b) the place-down disturbs the part on the table, (c) `run_assembly_step` opens the gripper as its first action when NOT in `--already-held` mode, dropping any held part wherever the robot currently is. We hit this on 2026-05-04 (iter 5 + iter 6 setups failed at `move_to_grasp` because the part had drifted). Keeping the part held end-to-end + re-rotating each attempt is the canonical pattern.

### Wrapper key APIs

- `cad_lookup.predict_tcp_at_seat(base, object, grasp_id, base_world_xyz, base_world_quat_xyzw, flange_offset_m=0.2286)` — full chain `T_world_base ∘ T_base_object_seat ∘ T_object_grasp_point ∘ T_grasp_point_tcp`. Reads `~/Documents/aruco-grasp-annotator/data/fmb_assembly1.json` + `grasp_points/<obj>_grasp_points.json`.
- `--config <yaml>` — defaults to `compliant_insertion_studio/configs/defaults.yaml` (universal, shape-agnostic). Per-shape YAMLs are deleted; the termination predicate is derived from CAD chain, not shape geometry.
- `--use-default-base-position` — required when `--base-position` is not passed; computes hover/insert targets from CAD chain.

### Cancel-safety chain (REQUIRES idempotent process-group cleanup)

```
Operator Ctrl+C
  → loop_iterate SIGTERM handler
    → os.killpg(iterate_insert pgid, SIGTERM)
      → iterate_insert SIGTERM handler
        → os.killpg(wrapper pgid, SIGTERM)
          → wrapper run_done():
              stop_force_mode → switch to scaled_joint_trajectory_controller
              → _await_controller_active (gates the next step)
              → python3 -m primitives.move_to_safe_height
              → exit
```

Each layer wraps the child in `os.setsid` (`preexec_fn=os.setsid` in `subprocess.Popen`) so SIGTERM hits the whole pgroup. The wrapper's `_await_controller_active` is what makes the safe-height subprocess actually run — without it, the script-mode invocation can fail silently (ModuleNotFoundError swallowed) leaving force_mode_controller active.

### Engagement gate must allow z-drop dominance (2026-05-04 PM)

The CAD-derived predicted xy can be 15–20 mm off the actual seat (observed on iter v70: peg seated at xy=(0.0301,−0.3598), CAD-predicted (0.0257,−0.3560), error ~17 mm even after spiral recentering). A strict 6 mm `engagement_dist_thresh_m` then wrongly rejects a peg that's clearly inside the slot. Add a z-drop dominance shortcut in ENTRY_SETTLE: if `surface_z − tcp_z ≥ engaged_z_drop_dominant_m` (default 20 mm), accept engagement regardless of CAD-xy distance. Operator-demo full descent is 25–30 mm, so 20 mm is a safe "definitely inside" threshold. The dist gate still catches metastable low-force states where peg sits on top of rim (z_drop ≪ 20 mm).

### Contact xy ≠ Seat xy (2026-05-04 evening)

Critical late-session finding (likely root cause of an entire 12-hour failed iteration session): the **first-contact xy** (where peg first hits Fz>5N) is NOT the **seat xy** (where peg ends up after descending through chamfer). For u_orange today: contact at (+0.0308, -0.3554), seat at (+0.0341, -0.3635). 5.7mm offset in X, 8mm in Y. The first contact lands on the **rim**, the seat is the **slot center**. **`hole_xy_prior` should be the SEAT xy**, not the contact xy. Using contact xy as the spiral target points the search at the rim, where the peg can't drop in.

For each operator demo, extract BOTH values; use SEAT xy as the prior:
- contact_xy = first ACTIVE row with `fz > 5N`
- seat_xy = LAST ACTIVE row (or row where descent rate drops to 0 with z near predicted seat z)

### State-independent global seat detector (2026-05-04 evening)

The FSM's local `_motion_stopped_first_t` resets on every state transition (ENTRY_SETTLE↔FIND_HOLE↔INSERT). With our gate flapping under marginal conditions, this means a SEATED peg can sit motionless for 60+ seconds without ever accumulating the 0.75s sustain needed for the local seat predicate to fire. **Add a state-independent seat detector that runs every tick regardless of FSM state** (in `update()` before state dispatch): if `surface_z - tcp_z >= 20mm AND |dz/dt| < 0.5mm/s AND tilt < 5°` for 1.0s sustained, declare exit_done. Verified offline on FAIL_INSERT CSV: would have correctly fired at t=74.75s when peg had been seated for 60s.

### Cross-run signal analysis findings (2026-05-04 evening)

Comparing 6 successful vs 122 failed u_orange attempts at each timepoint post-contact:
- **t=1.0s: success has F_lat=3.5N + xy_excursion=0.9mm; failure has F_lat=0.6N + xy=0.3mm.**
- **t=1.5s: success F_lat=4.7N, xy=2.1mm; failure 0.8N, 0.4mm.**
- t=2.0s: success z_drop=2.2mm; failure 0.16mm.
- Successes have ~5N sustained F_lat for 1-1.5s causing 2mm xy excursion, then peg drops at t=2s.
- Failures don't accumulate this lateral motion → never engage chamfer.

Implication: an `INITIAL_PRESS` phase commanding 5N sustained for 1.5s in a fixed direction is the missing mechanism. The spiral PD only generates 0.5-2N sustained because PD is reactive — error stays small. Operator's hand applies sustained directed push that algorithm doesn't replicate. Implemented in FSM as `find_hole_press_*` knobs.

### Anti-pattern: tuning parameters without first analyzing data

12-hour session in 2026-05-04 wasted hours on parameter iteration (16+ versions v82-v97) before doing the analytical work that should have come first. **Rule for next sessions: write a CSV cross-run analyzer FIRST. Iterate parameters only after the data has revealed the actual failure mechanism.** Monitor commands are for state-transition events, not for diagnosing why peg isn't moving.

### Mode A vs Mode B (Phase 5)

- **Mode A** = pure compliance with universal termination predicate:
  `motion_stopped AND tcp_z_reached_predicted (CAD-derived) AND descended_post_contact ≥ 25 mm`, sustained 1 s.
- **Mode B** = active correction triggered by stuck-detection:
  `net z-descent over 2 s window < 0.5 mm/s AND smoothed Fz > 6 N`, sustained 2 s. State machine: NORMAL → CORRECTING → COOLDOWN → NORMAL. Up to `max_corrections=12` per episode, then ABORT.

### Mode B action-type research (2026-05-04, /tmp/gpt-task-result-123444.txt)

We tested 4 iterations of "push harder" Mode B (downward Fz=-9N + lateral force pokes 6N + counter-torque 0.3Nm). All 4 ABORT at 12 corrections. The peg is geometrically wedged on the rim; pushing deeper deepens the wedge. **Action type matters more than action magnitude.**

Right action for peg-on-rim wedge:
1. Retract 0.5–1.5 mm to unload static friction
2. Drop Fz to -2 to -4 N during search (NOT -9 N)
3. Run continuous spiral search 1.5–2.0 s, radius 0.25 → 1.0 mm
4. Lower force-mode gain during search: `gain_scaling=0.4–0.7` (vs nominal 1.0), `damping_factor=0.15–0.30` (vs nominal 0.7)
5. Restore nominal Fz/gains after correction

Sources GPT cited: Chhatpar & Branicky (2001) — spiral pitch = clearance; FANUC force-control manual — switch to search when error > chamfer + clearance/2; Robotiq spiral-search practice; Tang et al. (2016) on three-point contact.

### Wrench-feature hierarchy (verified empirically on iter-4 dataset)

- **Wrench frame is `tool0_controller`**, NOT `base_link`. Telemetry CSV column `wrench_frame_id` confirms. Direction-aware features (r_cop, F_lat in operator's intuitive frame) MUST be computed in tool frame, then optionally transformed via TCP quaternion if needed for base-frame analysis.
- **r_cop = ‖(-Ty/Fz, Tx/Fz)‖** is the missing direction signal. On iter-4 ACTIVE phase: median r_cop = 5.5 mm, mean COP vector = (-1.0, -5.4) mm (consistent direction, not noise) — for a ~22 mm peg, that's rim-contact-scale lever arm. Raw `Tx, Ty` look "tiny" (~0.05 Nm) until you normalize by Fz.
- **Counter-residual is geometrically wrong for wedges**. When peg is wedged at (-X,-Y) corner, wrist sensor reads (+X,+Y) because the part is being pressed into the (+X,+Y) rim edge. Counter-residual direction = AWAY from target. **Use TOWARD-TARGET direction** (CAD-derived `target_xy − tcp_xy`) for force corrections.
- **Stuck-detection: net z-descent over 2 s window**, NOT instantaneous v_z. Smooth Fz over 0.5 s window before threshold-checking — force-mode oscillation makes instantaneous fz dip below threshold momentarily, breaking sustain timers.

### Subprocess invocation rule
