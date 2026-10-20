# Handoff for the Next Agent — Compliant Insertion Studio

> Read this **first.** Then route to the right doc based on what you're being asked to do.
>
> Date of last update: **2026-10-19 evening** (OptiPlex cell). The 2026-05-07 architecture below still
> stands; 2026-08-16 changed the branch and bringup path; 2026-10-14 fixed the grasp depth and
> added robot self-disassembly, a hands-off loop and depth seat checking; 2026-10-15 made the loop
> robust to the 7th-axis slider, fixed line_green's slot aim and scripted every recovery;
> **2026-10-18: 3 consecutive hands-off cycles (disassemble + assemble) passed** — see
> "State of the system" and the runbook under "Run commands".

## ⚠️ Which tree can run the real assembly

**Branch `ur5e-fmb1-demo` (working tree `~/Documents/ros-mcp-server-verified`) is the tree that
runs the real assembly.** `main` **cannot**, and the failure only surfaces once the robot is
already moving:

- `main`'s `utils/grasp_points_publisher.py` emits **composite** marker ids
  (`grasp_point_id*100 + direction_id` → 101…303) after the June candidate-native migration.
- The assembly JSONs still carry **flat** ids (u_brown=1, u_orange=1, line_green=1,
  inverted_u_yellow=2), and `replay_real_assembly.py` still passes `--grasp-id`. Nothing matches
  → every pick fails with `"Grasp point 1 not found … Available: [101, 102, …]"`.
- The replacement flag `--grasp-candidate` **refuses `--mode real` by design**
  (`primitives/move_to_grasp.py`, "no unverified real-arm path").

Work from `ur5e-fmb1-demo`, or from a worktree at tag `real-world-verified-2026-05-07`. **Start
the grasp publisher from that same tree** — running `main`'s copy re-introduces the composite ids
regardless of which tree the assembly script runs from. Every run command in this document assumes
this. `main` is sim-only until the candidate-native migration is finished on hardware.

## State of the system

**2026-05-07** (tag `real-world-verified-2026-05-07`, commit 75a418d): all four FMB1 objects
insert end-to-end via the replay script. Two insertion code paths in production:

- `compliant_insert` wrapper (autonomous SEARCH spiral + v4 detector) — u_brown, u_orange
- `prismatic_peg_insertion` (stash) routed via `translate_object` line_green/yellow branch — line_green, inverted_u_yellow

Routing key: `primitives/translate_object.py` `if args.object_name in ('line_green', 'inverted_u_yellow')`. **Unification of the two paths is queued for later work** — the split was retained because each path's compliance configuration suits a different geometry (XY-locked descent vs XY-compliant settling with Rx/Ry compliance).

**2026-08-16** — two full assembly passes on the real arm from `ur5e-fmb1-demo`:

| Pass | Result |
|---|---|
| 1 | Genuine autonomous **4/4**. |
| 2 | **Not** a 4/4. u_brown and u_orange seated on touchdown. `line_green` was **false-accepted at a wedge** (depth_err −2.32 mm) and `inverted_u_yellow` then seated only because the operator pushed line_green in **by hand**. |

Do not cite pass 2 as a second 4/4. The seat-acceptance gap it exposed is open work — see
"What's still open".

**2026-10-14** — robot disassembles and reassembles itself; 2 consecutive hands-off 4/4 cycles.

| Run | Result |
|---|---|
| Loop cycle 1 | disassemble 4/4 + assemble **4/4 hands-off**, 15.4 min (yellow seated on attempt 2) |
| Loop cycle 2 | disassemble 4/4 + assemble **4/4 hands-off**, 13.7 min |
| Loop cycle 3 | u_brown, u_orange seated; **line_green tilted** (pitch −2.45°, one end +4.95 mm), top-down reseat did not fix it → run stopped (fail-closed) |
| Board reset by operator | u_brown, u_orange seated; line_green tilted again (+3.3°); pulled out to the clear area and regrasped: still +3.0°; the top-down reseat after that read flat (0.81°); yellow seated → 4/4 |

What changed, and why (commits on `ur5e-fmb1-demo`, 2026-10-14):

- **Grasp depth was ~14 mm shallow on every part** (`386e31e`). `GRIPPER_CENTER_TOOL_OFFSET`
  = 0.2286 m is the RG2 **fingertip end**, not the pad centre (closed tips touched base1's top at
  flange − 230.86 mm). `move_to_grasp --pad-center` puts the pad centre on the grasp point (RG2
  4-bar linkage + 29.8 mm pad from CAD, `config.rg2_tip_length_m/pad_center_offset_m`), limited so
  the fingertips stay 4 mm above base1/seated parts at the seat (CAD meshes:
  `ablations/grasp_depth.py` → `eval_resources/fmb1_grasp_depth.json`) and 2 mm above the table at
  a pick. The executed depth (`grasp_tool_offset_m`) chains to the insert via `GRASP_TOOL_OFFSET_M`
  (`config.held_tool_offset`) and is recorded per part for the disassembly. Depths: u_brown/u_orange
  216.3 mm, line_green 225.8 mm (base top 7 mm under its fingers at the seat), yellow 211.0 mm.
  Operator's rule: *the grasp point and the pad centre should meet* — not "hold the whole height".
- **Self-disassembly with the insert's own grasp** (`ablations/disassemble.py`, operator rule: same
  grasp points as assembly). Free table spots from a median-of-10 depth map, ≥40 mm between parts.
  `--relocate PART [--from-staging]`, `--only PART --to-staging` (pull a bad seat out to the clear area).
- **Loop driver** `ablations/fmb1_loop.py --cycles N --start-with {disassemble,assemble}` → per-cycle
  record in `ablations/logs/fmb1_loop.jsonl` (seat − prediction per insert from the raw CSV).
- **line_green depth seat check** (`ablations/seat_check.py`): plane fit to the bar's top face from
  0.56 m (D415 gives no depth closer than ~0.45 m at 1280×720); pitch ≤ 1.5°, height ≤ 2.5 mm;
  one top-down reseat with the insert grasp if not flat.
- **Insert recovery** in the replay: a failed insert places the part in the clear area, regrasps,
  reinserts (`INSERT_MAX_RECOVERIES=1`). `--part-at-staging` resumes a part waiting in the clear area.
- **Force guards** use a 51-sample (~100 ms) median (`97f0bd1`): the robot-side wrench glitch runs up
  to 14 consecutive samples; a 15-sample median hit 22.6 N at rest and false-stopped a descent.
- **`stop_run.sh`** restores a commandable arm unattended (`e5c9dbf`): prismatic leaves
  `passthrough_trajectory_controller` active, which blocks the trajectory controller. Also fixed:
  prismatic's controller-state parse matched "inactive" as active; its seat may end ≤ 2.5 mm above
  the prediction (was 1.0 — yellow sat seated at +1.0 mm through two 60 s timeouts).
- Camera freshness check before every grasp; subprocess watchdog; per-sticker marker calibration
  (`aruco_camera_localizer` commit `97df4d5`, `config/sticker_calibration.json`).

**2026-10-15** — the failures behind "line_green doesn't go in" found and fixed; recoveries scripted.
Runbook: "Run commands → The hands-off loop, and getting out of trouble" below.

What was wrong, and the fix (commits `848031f` → `557dd71`):

- **The UR5e rides a Vention 7th-axis slider.** Inserts, nudges and pull-outs creep the carriage, so
  base1 moves *in the robot frame* (+8 mm in an hour) though it never moves on the table. The base
  was read once per run. Now: re-read between parts (on the vertical exit, 0.32 → 0.56 m, no
  detour), inside every seat check, between nudge tries, before every disassembly seat grasp.
- **The marker read of base1 is not good enough** (55–61 mm in x by viewpoint/run). `base_depth.py`
  fits the base's top-face outline (CAD 161 mm square) in aligned depth from 0.56 m: repeats to ~1 mm.
  Marker read is the fallback. Depth needs ≥ ~0.45 m: **not from home** (0 points there).
- **Base yaw was never applied to the insert.** `rotate_object` snaps to cardinals in the assembly
  frame; the base sits at ~3°. `translate_object`'s hover now turns the wrist by the residual yaw.
- **line_green's slot is the line through where the two U parts really seated**, not the base
  (CAD: u_brown y −47.5, u_orange +48.5, line_green between). It went straight in only when the two
  U parts sat at the same x; a 3.6 mm x difference turns the slot 2.2° → bar ends ±2.7 mm off →
  it sat on the U tops 38 mm up. Now aimed at the U parts' mean x and the yaw of their line, from
  their own force-search seat offsets (`ablations/logs/seated_offsets.json`). Not their y.
- **prismatic_peg_insertion pulled seated parts out.** A part at seat depth 6.6 mm from the aim was
  called a failure and retracted *gripped* (dragging seated parts); a part that had dropped into
  the channel was aborted on 5 mm drift. Now: at seat depth, ≤ 15 mm off-aim is a found slot; once
  5 mm below first contact, drift ≤ 15 mm is allowed.
- **Nudge (tilted line_green):** one push until 4 N (operator's rule), capped 15 mm past the end;
  the fingers come down past the RAW depth end (depth smears a raised end ~7 mm outward, so the
  corrected end is only for the cap) and step further out on contact — never land on the part.
- **A part slipped out during a pull-out and lay 7 mm from base1;** every step said "ok" and the
  next pick put an open finger on the base (protective stop). Now: grip checked (RG2 grip flag) after
  close and after the pull-out; `spacing_check.py` (every table part ≥ 25 mm from base1 and the other
  parts, CAD outlines at observed poses) runs at assembly startup and after disassembly; a part
  against base1 is picked with a grasp point whose fingers clear it (`clear_grasp`, line_green has
  grasp points 2/3 at ±50 mm along its length — added to aruco-grasp-annotator's
  `data/grasp_points/line_green_grasp_points.json`, which is **not** under git).
- **Protective stop → the run used to wait 60 s and command the stopped robot.** Now the replay polls
  the dashboard at 1 Hz, interrupts the step and skips recovery/cleanup; `recover_pstop.sh` recovers.
- **Robot program dropped under CPU load** (no RT scheduling on this PC; load 10.7/6 cores). The
  camera stack and RViz run at nice 10, the run scripts at nice 5.
- **Spots:** inside the home image by 60 px and above its bottom 160 px (the fingers at pre-grasp
  width hide that band — yellow there went undetected).

**2026-10-18 — 3 consecutive hands-off cycles: the end condition is met.**
`ablations/fmb1_loop.py --cycles 3 --start-with disassemble` (run "1018b", record in `ablations/logs/fmb1_loop.jsonl`,
logs `ablations/logs/loop_20261018_{105236,110346,111547}_*.log`): every cycle disassembled 4/4 and assembled 4/4 with no operator
action, 11.2 / 12.0 / 12.1 min per cycle. Every part verified after its insert and again at the end
(`verify_assembly --xy-only`, marker xy 0.2–6.7 mm). line_green came out tilted ~3.5° after the insert in
2 of 3 cycles (and in ~every run on 10-17/18); **one nudge flattened it every time** (here pitch → 0.01° and 0.14°).
Videos of those assemblies: `~/Documents/fmb1_frames/mp4/fmb1_run1018b_cycle{1,2,3}_assembly_*.mp4`
(also on dual-a4500 `~/fmb1_videos/`, and copied to the operator's Mac `~/Downloads`).

What changed 10-15 → 10-18, and why (each fixed a stop seen on the cell):

- **Placing, not dropping** (`bfdc78d`): `move_down --to-table` stops with closed tips 1 mm above the
  table — the old 80 mm floor dropped parts. Hover yaw fix = **base yaw only** (the chained camera
  residual turned U parts 5–7°; with no yaw fix yellow failed 6/6). Insert recovery lifts to safe
  height before moving to the clear area (it dragged a part across the assembly).
- **Never lift a gripped seated part; retry a failed release; halt when the robot is unreachable**
  (`6df8f74`). Disassembly grasps where the insert actually seated the part (`7613ed0`).
- **Spot planning** (`cc58efa`, `397ddff`, `9536863`, `0ba0bf1`): bounded backtracking over a
  reachable-ring grid, farthest-first, 30 mm between CAD outlines, view margin 40 px, finger band 100 px.
- **Empty gripper passed the grasp check** (`96d6a42`): u_brown's pick closed on nothing (0.6 mm), the
  width-only check passed, the insert "seated on touchdown". Now fails at width ≤ grip width − 6 mm.
- **Verify the assembly after every part** (`7eedefe`, `da43fb9`, `a9d2811`, `8bcdf1e`):
  `verify_seated` runs `queries/verify_assembly --xy-only` (xy + yaw) on every seated part from the base
  view, between parts and at the end. Only readings that DETECT the part vote (majority); "not detected"
  extends the window (≤ 8 reads, the localizer drops frames for seconds); a part never seen from above
  gets **one second look from home** (seated u_brown lies on its side right under the camera there —
  edge-on markers; it stopped a good cycle 3 on 10-18). Occlusion ledger
  `ablations/logs/assembly_state.json`: line_green under yellow passes only if verified before yellow.
- **Insert aim from seat history** (`6c1494f`): median of each part's last 5 seat offsets
  (`seat_offset_history.json`) — u_orange's seat sat 7–9.5 mm off in y, outside the 8 mm spiral.
- **Yellow's marker misread** (`fb3df93`, `6113d72`, `7c9f84f`, `651c01b`): the localizer intermittently
  puts yellow 70 mm below the table and ~76 mm off in xy (z −113 mm, grasp point −133 mm), in streaks of
  8–15 s right after arm motion, never with the arm still (0/157). One such grasp point sent step 1 to
  hover off the part with the wrong wrist yaw and step 2 came down on it at −87 N. Now `move_to_grasp`
  rejects poses AND grasp points below the table, waits out a misread streak (≤ 60 attempts, 30 s), and
  never descends to a target > 20 mm (xy) from where it hovered (re-hovers once, then fails).
  `spacing_check` clusters samples and lets depth pick between disagreeing readings (`55fe615`, `c0464fc`).
- **Rare roll wedge of line_green** (`1e74fd5`): level but > 2.5 mm high → tap both edges at 3 N with
  closed fingers, press the higher at 5 N (`ablations/tap_check.py`). Depth roll is unusable (−4° bias).
- **Demo video** (`b741e35`): never live-encode on the OptiPlex — two live H.264 recorders took the load
  5 → 12 on 6 cores and the CPU-bound localizer dropped/misread parts (three loop stops on 10-17).
  Record a rosbag, render later elsewhere (runbook below).
- **Cell:** enp1s0 is static (DHCP cycling dropped the arm link); after a link event restart the driver
  with `close_robot.sh` + `stack_up.sh`.

**2026-10-19 — generated start layouts (operator: "tip the parts flat, leave one tippable upright").**
The disassembly used to leave every part standing as seated, so every assembly was the easy 1-pass
case. `--layout staging` (fmb1_loop / disassemble) follows the IDEA of the operator's sim task
`fmb1_assemble_ur5e_rg2_flat_gen` (dual-a4500 `pa-footprint/plan-anything/data/tasks/...`; not its numbers —
the sim robot sits at table height): parts flat on a large face, yaw/flip free, in the staging
rectangle, ≥ 25 mm apart; one tippable part per cycle stays upright (u_brown → u_orange → yellow), or
`--all-flat`.

- **Staging rectangle** (sim → base_link → UR base): x 0.19…0.51, y −0.54…−0.22 — the sim's "reorient"
  rectangle lands on our clear area (−0.32, −0.50), which confirms the mapping. Clear (depth), reachable
  top-down (IK), seen from `STAGING_VIEW` (0.35, −0.38, 0.56) — **not from home** (home sees x ≤ ~0.42).
  The disassembly looks there first (depth: nothing raised may sit in a part's footprint or tip sweep).
- **Tip** (`ablations/tip_part.py`, operator's method): closed fingertips strike the part's TOP at an angle,
  ≤ 5 N, like the line_green nudge — lowered beside the part (guarded), one diagonal stroke meeting the near
  face ~16 mm below the top, ending 45 mm above the table (never touches the fallen part). U parts are struck
  on a leg (they stand open-top), yellow on its solid top centre. `--observe` tips a part where it stands.
- **Bugs found on the arm, fixed:** the gripper holds the U parts by a LEG, 25.5 mm off centre — plan by
  the part's centre, send the flange to centre + grasp offset (aiming by the flange put the strike past the
  part); a stroke starting above the top only clipped yellow's edge (5.9 N, no tip); wrist turn + carry in
  one move, and even a pure 89° turn in place, exceed move_cartesian's 1.5 rad joint step — turns over 60°
  go in two halves, in place, and back the same way; the stroke's wrist turn is ≤ 45°.
- **Assembly:** parts not seen from home are planned (`--derive-sequence`) and picked from the staging view.
  Flat U parts / yellow take the 2-pass path (place in the clear area, regrasp, insert).
- **Results 10-19 — NOT yet 3 clean staging cycles in a row** (a fresh 3-cycle staging run started 17:40 with
  everything below in; its outcome is in `ablations/logs/fmb1_loop.jsonl`). Every staging ASSEMBLY that ran
  to its end passed 4/4 hands-off (flat u_orange/u_brown/yellow on the 2-pass path, parts out of the home view
  picked from above their last-seen position). Best run before the afternoon's fixes: 2 clean cycles in a row
  (13:43–14:25). Each stop since, its cause and fix (all committed):
  - the 89° wrist turn over `move_cartesian`'s 1.5 rad cap (`8ac1c91` two halves; then `e14e4d8`: wrist_3
    gets its own 3.2 rad cap, one turn per part, no turn-back — the next part's hover turns at safe height);
  - a tipped yellow sliding 13 mm on → `0a35c74` landing margins, `68f9dcb` the draw keeps the most
    spread-out of 30 (closest pair median 32 → 55 mm; the region is ~24 % used);
  - **set-downs landed at 17–44 N** (move_down's 100 ms median lags the s-curve descent; measured) — an upright
    u_brown fell and lay 16 mm from yellow. `a02c4da`: move_down to 15 mm above the CAD touch height, then a
    4 mm/s guarded move; every landing since stopped at 5.0–5.1 N;
  - **`--relocate` scored every grasp at grasp 1's 35 mm opening**; u_brown's 84 mm bridge grasp looked clear and
    the 35 mm-open fingers came down on it → PROTECTIVE_STOP. `c4ef229`: each grasp at its own
    `grasp_validity` width, clearest wins; `801ecd9`/`68f9dcb`: a part too close is relocated inside the staging
    region (`--spot-staging`, farthest spot ≥ 25 mm, stance kept) and the check re-run once;
  - the staging view at 0.62 m sees a standing yellow only intermittently → `d0e9d98` pre-pick view straight
    above the part's last-seen position (0.52 m), seen for 20 s before the pick; `636cca1` the pick's own
    "not detected" wait 5 → 20 s; the layout check waits out a misread streak (30 s), looks above the spot, then home;
  - the assembly's startup spacing check now relocates a too-close part instead of stopping (`e14e4d8`);
  - **a flat yellow with its bridge grasp 5 mm from base1's edge** (base read scatters ~8 mm) → the descent hit the
    base → PROTECTIVE_STOP. `95b2343`: before every pick the recorded grasp's open-finger outlines are checked
    (≥ 15 mm from base1 / other parts) and the part relocated first if not. Open: `move_to_grasp`'s descent guard
    did not fire (its baseline is re-taken after a 0.5 s grace, and the F/T glitch makes a tighter threshold false-stop).
- **Do not skip the pass-2 `rotate_object`** (tried `rotate_object skipped` when the chained orientation was
  already fold-equivalent to the target: two rim stalls, insert aimed at TCP x +94 instead of +42 = 51 mm = 2 ×
  the leg-grasp offset). The camera's raw quaternion and rotate_object's output are the two fold branches of
  one orientation; `predict_tcp_at_seat` rotates the (asymmetric) grasp offset by the branch it gets, and the
  chain was validated with rotate_object's. Reverted in `77c8bc9`; the 0.01 rad step stays (~10 s per 2-pass part).
- **Prismatic insert retries re-aim from data, bounded by CAD** (`b3c60fa`, the insert code opened by the
  operator for this). The ladder used to retract 1 cm and descend at the same xy (3 identical 5.0 mm stalls on
  yellow). Now it logs which CAD contact the stall matched (`Contact candidates (CAD-derived): SEAT / BASE_RIM /
  OBJ:line_green …` via `cad_geometry`, the compliant path's chain; yellow: rim = seat + 25.5 mm, line_green's
  top = seat + 45.5) and attempt 2 aims where the part slid to, attempt 3 the mirror, for slides of 1.5–10 mm.
  Pockets are sub-mm (CAD: yellow x ±0.6 / y ±0.5; feet astride line_green ±1 mm) — no yaw/tilt search. Verified
  on the arm: line_green "stalled +37.4 mm above the seat, slide 1.1 mm → not an xy miss" (seated on attempt 2);
  yellow seated on attempt 1 (compliance slid it 5.0 mm). Yellow's earlier 3× stall was a grasp offset: the
  regrasp recovery seated it at 0.8 mm.
- Verify from the base view: seated u_brown reads tilted ~10° (edge-on markers) → a reading tilted > 5° is
  unusable; the second look from home decides (`1fe84d2`).
- **Iterate per part, not per cycle, when fixing something** (operator): `docs/DEBUG_LOOP.md`, CLAUDE.md §5a.
- Seat-offset history (`ablations/logs/seat_offset_history.json`): every part seats 5–8 mm in −y from the CAD
  prediction (yellow −5…−9, U parts −2…−10); the aim bias absorbs the median, the compliance finds the rest.
  The constant −y bias is the queued calibration item, not a ladder problem.

**2026-10-19 evening — game plan, geometry, per-part iteration.** Operator directions (all in memory notes
and the game plan): the base will not stay fixed (cm, degrees) — everything follows the camera-read base;
solve pose per part but the abstracted insertion primitive carries NO per-part settings (stages, clearances,
free DOFs from CAD + observation); remove the errors we can measure before touching the primitive; iterate
ONE part (pull + reinsert), judged by ease metrics from raw logs, never the FSM label; reads are an
optimization target (measure the grasp error once on the held part, then chain); commit freely.
- **Game plan** (Claude Doc, link in memory `game-plan-doc`): tracks A perception/pose, B one primitive,
  C operations, milestones M0–M3; the planner / execute-mate / vision boundary (for prompt2assembly transfer);
  GPT's review verbatim in `docs/GPT_PLAN_REVIEW_2026-10-19.md` (agrees: base-free before the primitive; adds
  pose freshness gates, measure the seated part not the TCP, re-read the base after contact, 30 single-part
  trials per condition).
- **Ease metrics**: `ablations/insert_ease.py` (attempts, contact→seat s, seat dev in the base frame, lateral
  force-time N·s, stalls); `ablations/insert_stages.py` (time per CAD stage). `docs/INSERT_GEOMETRY.md`: the
  prismatic parts spend 58 % of insert time wedged in a corridor (line_green 5–9 mm into the U openings,
  ±1.1 mm; yellow's legs astride line_green, ±1 mm) that no TCP tilt shows; per-stage rules for the primitive.
- **Loop 1019c** (staging): cycles 1–2 clean (1338 s, 1152 s), cycle 3 failed on line_green turned −4.5° from
  insert-derived U offsets. **Loop 1019d**: cycle 1 clean (1395 s), cycle 2 stopped by a relocation judged
  failed after it had worked (fixed `8fd53c5`); its assembly then passed by hand with two autonomous recoveries.
- **Fixes (all committed)**: seated-part grasp from a close hover read (`seat_grasp(observed_part=)`; the base
  view reads step ~8 mm between two localizer detection sets); gentle clear-area set-down in the assembly
  (sideways stances still release at move_down's 80 mm floor — operator: the topple is acceptable);
  seated-part tops from the rotated CAD mesh (`cad_geometry`, U tops were 24.5 mm low); line_green's turn from
  close hover reads of the seated U parts (`record_follow_reads`, 0.30 m above each seat; the base-view reads
  said 1 mm while the bar found the slot 6 mm across), insert-based fallback capped 1.5°; grip-width guard
  after every close (expected `config.grip_width_mm` ± 8 mm → open, lift, re-pick once; a flat u_brown closed
  across both legs at 69 mm and slipped out — the opening chooser had judged 32 mm clearance from a pose it
  could not have, logged now); fold-aware orientation mean in `observe_parts`; corridor-aware prismatic
  retries (`df1c0a5`: retract above the entrance — the old 10 mm retract stayed inside the opening — Rz
  compliant, re-aim ≤ 2 mm); telemetry meta on the trajectory-reached exit; **held-yaw calibration** before
  every insert (`held_yaw_calibration`: read the held part above the base, turn the wrist by the difference to
  the chain, modulo fold symmetry, 0.3–6°) — line_green read +2.26° on its first use; yellow's marker is not
  visible while held; u_brown is detected only above the base (0 over the taped clear area).
- **Per-part data tonight**: `ablations/part_loop.sh u_brown 3` — iteration 1 (no correction): straight drop,
  contact→seat 0 s, dev (−0.1, +0.1) mm, 0 N·s; iteration 2: 14.5 s on the rim, (+1.7, +2.5) mm, 24.8 N·s;
  iteration 3 never ran: the **workspace light went off** at ~21:45 and the verify failed closed (localizer
  sees nothing). Board then: u_brown, u_orange seated; line_green at (314, −471), yellow at (283, −337).
- **Next**: `compliant_insertion_studio/scripts/when_lit.sh '<cmd>'` gates any command on the light. Resume
  with line_green then yellow (`replay --only … --skip-startup`), then `part_loop.sh u_brown 3` for the held-yaw
  A/B (GPT's experiment 2), then plain-layout cycles (`fmb1_loop.py --layout spots`) before any tipping loop.
  Open: the held-yaw read for yellow (marker hidden by the bridge grasp); line_green's held-xy; the U parts'
  seat spread (2–7 mm; the hover reads put u_brown at (−6.4, −3.0) mm from CAD) — GPT's experiment 1.

---

## How to localize yourself in the repo (5 minutes)

### 1. What the project IS

**Compliant Insertion Studio** = a force-compliant peg-in-hole **insert primitive** + the operator-demo collection methodology that produced it. Operates on a UR5e + OnRobot RG2 + workspace camera. Replaces the broken `prismatic_peg_insertion.py` real-mode insert path with a per-object parameterized algorithm. FMB1 assembly proof-of-concept (u_brown, u_orange, line_green, inverted_u_yellow).

The high-level deliverable is one CLI: `run_assembly_step.py --autonomous-search` that takes any FMB1 part on the table, picks it via camera, and inserts into base1 — no operator hands.

### 2. The four entry-point docs (read in this order)

1. **`CLAUDE.md`** (repo root — *not* `compliant_insertion_studio/CLAUDE.md`, which does not exist) — constraints, tech stack, conventions. Skim first; the Anti-patterns and Conventions sections are binding. On `ur5e-fmb1-demo` this file carries the 2026-08-16 corrections and the current bringup sequence.
2. **`compliant_insertion_studio/docs/AUTONOMOUS_INSERTION_METHODOLOGY.md`** — current working architecture. Calibration, SEARCH director, parameters. Treat as the reference for how things work now.
3. **`compliant_insertion_studio/docs/COLLECTION_METHODOLOGY.md`** — the upstream operator-demo pipeline that produced the data v4 was derived from. Read if you're collecting more data.
4. **`compliant_insertion_studio/.claude/skills/insertion-control-law-derivation/SKILL.md`** — methodology rules (data > guesswork; structural > parametric; tool-frame for sensor features). §13–§19 are the running record of session-by-session lessons. Binding for any FSM code change.

### 3. The session reasoning trace

5. **`compliant_insertion_studio/docs/ITERATION_TRACE_2026-05-06.md`** — full chronological reasoning for the 2026-05-06 session. Every iteration, every dead-end, every hypothesis I had to update. **Read this before re-exploring any of the rejected paths** (e.g., `-r_cop` direction control, position-PD tracking, hardcoded bias correction).

### 4. What's true and binding right now

| File | What it is |
|---|---|
| `analysis/CONTROL_LAW.md` | v4 Found Hole detector spec — `\|fz\|` state-transition predicate, validated 10/10 |
| `analysis/SEARCH_CONTROL_LAW.md` | **Historical.** Specifies the `-r_cop` director, which was tried and **refuted** (see the anti-pattern list below). Its one still-current statement is the SEARCH `selection_vector = (T,T,T,F,F,F)`. For the director actually in production (Archimedean spiral + constant-force + lag-pause + gradient-following) read `docs/AUTONOMOUS_INSERTION_METHODOLOGY.md` §2. |
| `analysis/AUTONOMOUS_RUN_LOG.md` | Per-iteration test journal — append-only |
| `ablations/eval_resources/fmb1_assembly.json` | Source of truth for `gripper_width_mm` and `grasp_id` per object |
| `ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json` | Canonical FMB1 tool-sequence; `replay_real_assembly.py` consumes this |
| `ablations/replay_real_assembly.py` | Real-mode counterpart to `replay_verify.py`. Drives the full assembly via `--assembly-json` + `--only`. |
| `primitives/shared/config.py:DEFAULT_BASE_POSITION` | (+58.25, -472.10) mm — calibrated for FMB1 base1 fixture from u_brown insert (2026-05-07) |
| `primitives/shared/config.py:PER_OBJECT_BASE_OFFSET_M` | Per-object delta (mm) on top of DEFAULT_BASE_POSITION; populated for ALL 4 objects from observed seat TCP. **Both insertion paths apply this offset** via `--final-base-pos` (wired in `translate_object.py`). |
| `primitives/_real_mode_stash/prismatic_peg_insertion.py` | line_green / yellow insert primitive. Force-added 2026-05-07; gitignored elsewhere. Geometric exit: depth_err ∈ [-8, +1] mm AND `\|dz/dt\| ≤ 0.5 mm/s` over 1.5 s, sustained 1 s. |
| `compliant_insertion_studio/scripts/loop_line_green_prismatic.sh` | Smoke-test loop for line_green (regrasp + rotate + translate_object insert) |

---

## Run commands (copy-paste)

> **All of these are real-mode and only run from `ur5e-fmb1-demo`** (or a worktree at tag
> `real-world-verified-2026-05-07`), with the grasp publisher started from that same tree. On
> `main` every pick fails — see the branch warning at the top.

### Bringup

On `ur5e-fmb1-demo` the whole cell comes up in one command, with each stage verified before the
next starts:

```bash
bash compliant_insertion_studio/scripts/stack_up.sh     # driver -> gripper -> camera -> grasp points
bash compliant_insertion_studio/scripts/stack_down.sh   # clean shutdown
```

`stack_up.sh` starts the grasp publisher from `$PWD` deliberately, so run it from the tree you
intend to run the assembly from. It expects the robot powered, brakes released, and the pendant
in **REMOTE CONTROL** — the driver comes up headless, no program loaded and nothing played. The
four scattered bringup commands (`launch_robot.sh real --headless`, gripper bridge,
`launch_camera.sh`, `grasp_points_publisher.py`) still work individually; `SETUP.md` §3 documents
them.

Since 2026-10-15 `stack_up.sh` also starts the **RealSense node with aligned depth** (the localizer
reads its colour topic; the base read, seat check, spacing check and spot planning need the depth)
and drops the camera stack to nice 10. `stack_down.sh` stops that node and the demo RViz too.
RViz with the raw camera: `bash compliant_insertion_studio/scripts/launch_rviz_demo.sh --raw`.

### The hands-off loop, and getting out of trouble (2026-10-15 runbook)

Everything below is a script; nothing needs inline python. All of it from the repo root, on
`ur5e-fmb1-demo`, after `stack_up.sh`.

```bash
# The loop: disassemble -> assemble, N times; fails closed at the first problem
python3 -u ablations/fmb1_loop.py --cycles 3 --start-with disassemble   # parts seated now
python3 -u ablations/fmb1_loop.py --cycles 3 --start-with assemble      # parts on the table
python3 -u ablations/fmb1_loop.py --cycles 3 --layout staging [--all-flat]   # generated layouts, tipped flat
#   per-cycle record: ablations/logs/fmb1_loop.jsonl; logs: ablations/logs/loop_<ts>_*.log
#   staging layouts: ablations/logs/staging_layout.json (last) + staging_layouts.jsonl (all)

# Staging layout pieces (all from the repo root)
python3 -u ablations/staging_layout.py --look [--cycle N] [--seed S]   # look over the region + draw one (no part motion)
python3 -u ablations/tip_part.py PART --observe                       # tip a standing part where it is
python3 -u ablations/disassemble.py --assembly-json x --layout staging --resume-held PART   # finish a stopped one

# One part (the assembly JSON every replay takes)
J=ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json
python3 -u ablations/replay_real_assembly.py --assembly-json $J --only line_green

# Checks (no motion unless noted)
python3 -u ablations/spacing_check.py              # from home: table parts >= 25 mm from base1 + each other
python3 -u ablations/base_depth.py                 # base1 from its depth outline (arm at flange 0.56 over it)
python3 -u ablations/seat_check.py --object line_green   # moves to the 0.56 m view; flat/high/tilted
python3 -u ablations/disassemble.py --assembly-json x --dry-run   # plan spots -> /tmp/disassemble_spots.png
```

| situation | do |
|---|---|
| robot in **PROTECTIVE_STOP** (the run halts by itself, no cleanup motion) | `bash compliant_insertion_studio/scripts/recover_pstop.sh` — unlock, re-send program, lift straight up, home. Leaves the gripper as is. |
| a run must stop now | `bash compliant_insertion_studio/scripts/stop_run.sh` |
| a part is **too close** to base1 / another part (spacing check fails) | `python3 -u ablations/disassemble.py --assembly-json x --relocate PART --grasp-id auto` — picks with the grasp point whose open fingers clear the obstacles, sets it down at a free spot. Part seen badly from home (bottom of the image): add `--pick-view-dy 0.10`. |
| the gripper still **holds** a part after a failed run | `python3 -u ablations/disassemble.py --assembly-json x --place-held PART` (to the staging area), then `replay_real_assembly.py --assembly-json $J --only PART --skip-startup --part-at-staging` |
| line_green seated **tilted** | the replay nudges it by itself (push until 4 N from past the high end, ≤ 3 tries). By hand: `seat_check.py` to see which end. |
| a seated part must come out (bad seat) | `python3 -u ablations/disassemble.py --assembly-json x --only PART --to-staging` |
| robot program dropped (`Connection to reverse interface dropped` in `/tmp/ur_bringup_logs/`) | `stop_run.sh` re-sends it; check the load (`uptime`) — the camera stack must be at nice 10 |
| gripper `Circuit1/2:True` (latched) | power-cycle procedure in the root `CLAUDE.md` |

State the scripts keep (all under `ablations/logs/`): `grasp_tool_offsets.json` (grasp depth per
part, chains insert ↔ disassembly), `seated_offsets.json` (seat − prediction per seated part; the
line_green slot follows u_brown/u_orange from it; disassembly removes entries), `seat_checks/`,
`seat_offset_history.json` (insert aim bias), `assembly_state.json` (verify/occlusion ledger).

**Demo video (2026-10-18).** During the run, only a rosbag on the OptiPlex; the overlay is drawn
afterwards from the poses the localizer PUBLISHED, with its own drawing code — nothing re-detected:

```bash
bash compliant_insertion_studio/scripts/record_demo_bag.sh ~/Documents/fmb1_frames/NAME_bag   # SIGINT to stop
# on dual-a4500 (~/fmb1_videos: render_bag.py, encode_frames.sh, localizer_src/, wireframe/ copied there):
python3 render_bag.py NAME_bag NAME --localizer-src localizer_src --wireframes wireframe --workers 16
bash encode_frames.sh NAME NAME      # NAME_{rgb,annotated,side_by_side}.mp4, real-time
```
~2.4–4 MB/s of bag (1.4–2.1 GB per assembly), ~12 % of one core. The localizer itself publishes no
pose for ~18 % of frames (gaps ≤ 4.8 s) — the overlay is blank there, as the live stream was. The Mac
is reachable only through a Tailscale DERP relay (~0.4 MB/s): copy with `rsync --partial`, no timeout.
dual-a4500's disk is ~97 % full: delete `NAME/rgb` and `NAME/annotated` after encoding.

### The autonomous insert (what worked in iter 8)

```bash
# Start from object on table (full pick path; camera must be up):
python3 -u -m compliant_insertion_studio.scripts.run_assembly_step \
  --object-name u_brown --base-name base1 --grasp-id 1 \
  --mode real \
  --base-offset-xy 0.0 0.0 \
  --fz -9.0 --override-fz-cap \
  --step-back auto --step-back-seconds 5.0 \
  --autonomous-search \
  --search-F-press-N 9.0 --search-Fmax-N 8.0 --search-v-s-mm-s 5.0
# (--grasp-width auto-resolves from fmb1_assembly.json)
```

Two things in that command are load-bearing and were wrong in earlier revisions of this doc:

- **`--step-back-seconds` must be ≥ 1.5 s**, and 5 s is the value the collection protocol uses.
  This gate is what lets the arm come to rest before `zero_ftsensor`. Measured 2026-08-16:
  **0.0 s settle → 15.6 N post-zero bias**, which drove the TCP **116 mm upward** in force mode;
  5.36 s settle → 0.11 N. A shortened gate produces a plausible wrong number that nothing
  downstream can detect. Never trade it for speed.
- **`--search-F-press-N 9.0 --search-Fmax-N 8.0`** are the tuned values `translate_object` passes.
  The 5.0/5.0 pair this doc used to advertise is weaker and **measurably worse on u_brown**.

### Multi-iteration validation with fresh regrasps

```bash
bash compliant_insertion_studio/scripts/loop_autonomous_insert.sh 3 \
  --object u_brown --no-randomize --regrasp
```

Auto-resolves `grasp_id` per object. Camera must be up (`stack_up.sh` does this):

```bash
bash compliant_insertion_studio/scripts/launch_camera.sh --background
```

### Collecting a new GUIDED demo (single)

```bash
python3 -u -m compliant_insertion_studio.scripts.collect_regime_data \
  --object <name> --base base1 --grasp-id <N> --grasp-width <W> \
  --fz 9.0 --step-back-seconds 5.0 \
  --held-quat <QX QY QZ QW> \
  --variations A_pos_x_10mm --reps 1
```

### Re-grasp a held part cleanly (camera-driven)

```bash
python3 -u -m compliant_insertion_studio.scripts.regrasp_held_object \
  --object-name <name> --grasp-id <N> --mode real --skip-camera-check
```

---

## Architecture summary (condensed)

```
APPROACH (Fz=-9N descent)
  ↓ contact (|fz_smoothed|>3N sustained 0.1s, after 1s grace)
SEARCH (autonomous spiral PD director)
  • center: predicted_tcp_xy from CAD chain (no FSM-side bias)
  • r0=1.5mm, pitch=2mm, v_s=5mm/s, R_max=8mm
  • constant-force tracking with sign-flip: F_xy = -Fmax * unit(p_ref - p_tcp) - Kd*v
  • lag-pause: theta only advances when peg-to-ref ≤ 2mm
  • gradient override: when |fz| dropping fast AND <6N, command F_xy = -Fmax*unit(v_peg)
  • stall detector: aborts if peg progress / spiral arc < 15% over 1s
  ↓
v4 detector fires (|fz| transition >4N → <3N sustained 0.3s, recent on-rim in 2.5s)
  OR
global seat detector fires (peg dropped straight through during APPROACH/SEARCH)
  ↓
INSERT_DESCENT (Fz=-9N, xy locked, until predicted_tcp_z reached + motion stopped)
  ↓
DONE
```

Code: `compliant_insertion_studio/wrapper/contact_search_fsm.py` (FSM, SearchDirector, FoundHoleDetector). FSM is reusable; tweak parameters via cfg or CLI.

---

## What's binding (must respect)

From `CLAUDE.md` (project conventions) and `SKILL.md` (methodology):

- **Pendant in REMOTE CONTROL; the driver comes up headless.** `launch_robot.sh real --headless`
  sends the control script over the primary interface — no External Control `.urp` is loaded and
  nothing is played on the pendant. (There is no program named `external_control.urp` on this
  controller.) Headless control **cannot** be established from Local mode. `dashboard_client`
  power-on / brake-release / mode queries do work in Remote; protective-stop recovery is still an
  operator action at the pendant.
- **Force-mode wrench ≤ 5N default**, override only with explicit operator awareness.
- **Settle the arm before any force reference.** ≥ 1.5 s after a Cartesian move before
  `zero_ftsensor` or any software baseline sample. The trajectory controller reports "complete"
  at the commanded position, not at rest. The wait must come *before* the sample, not after it.
- **Contact detection needs a sustain window, never a single-sample threshold crossing.** The
  wrench stream carries ~60 ms impulses reaching 40–60 N with nothing touched (measured at 50 Hz:
  `+0.18, +19.32, +56.98, +20.56, +0.36 N`). Re-zeroing or slowing the move changes the number
  without fixing it.
- **Rotation stays LOCKED in SEARCH** — `selection_vector = (T,T,T,F,F,F)`. See the anti-pattern
  list below; unlocking it broke five consecutive real-arm runs.
- **SIGTERM cleanup must reach safe-state exit.** All states have idempotent cleanup; force_mode → scaled_joint_trajectory_controller switch must complete.
- **No commits unless explicitly approved.** No `Co-Authored-By` lines (operator's global rule).
- **Don't trust FSM stdout outcome labels.** Verify physically from raw CSV. `outcome=success/fsm_seated` ≠ peg in hole — check Δz_to_predicted < 5mm + motion stopped.
- **F/T calibration is three layers**: foundational `set_payload` (per-mount, one-time), session smoke test, per-pose `zero_ftsensor`. The pre-bringup `launch_robot.sh` calls `set_payload` from latest `configs/ft_calibration_*.yaml`.
- **`primitives/shared/config.py:DEFAULT_BASE_POSITION` is calibrated** to
  `[0.05825, -0.47210, TABLE_HEIGHT + 0.0175]` — i.e. **(+58.25, −472.10) mm**, matching the table
  entry above. Don't reset it unless you re-derive from a centered-grasp demo. *(An earlier
  revision of this line said `(0.00152, -0.40378, ...)`; that value was never in `config.py` and
  is ~76 mm away from the calibrated one. Read the value out of `config.py`, not out of a doc.)*
- **A ~11.6 mm disagreement between `DEFAULT_BASE_POSITION` and the camera is expected.** That is
  the documented CAD-prior error (the chain is empirically 5–17 mm off), **not** a moved fixture.
  The base1 fixture is fixed and correct. Do not re-calibrate on the strength of that residual —
  the spiral exists to absorb it.
- **Empirical sign-flip** in SearchDirector: `F_xy = -Fmax * unit(error)`. Multi-axis force-mode in our base_link↔base setup inverts. Don't "clean up" without empirical re-validation.

---

## What's still open (priorities for next session)

### Opened 2026-08-16

0a. **The prismatic seat-acceptance criterion is too permissive — it false-accepted a wedge.**
    `EXIT_GEOMETRIC_Z_TOL_BELOW_M = 8 mm`, but the real separation is far tighter: a good
    line_green seat measured **−1.65 mm** and the accepted wedge **−2.32 mm**, only **0.67 mm**
    apart. `pos_dev` and `ori_dev` do **not** discriminate — the wedge scored *better* on both
    (0.5 mm / 0.45° vs 0.8 mm / 0.74°). This is unsolved; do not treat a `[SUCCESS]` from the
    prismatic path as a seat without checking depth against the good-seat figure.

0b. **`line_green` rotates at grasp.** Known, unsolved. When it does, it seats proud/wedged and
    the only recovery is a full re-seat. Camera pose and force reads are **both** unreliable for
    detecting this state, so there is currently no autonomous detector for it — this is what
    forced the manual intervention in pass 2.

0c. **`prismatic_peg_insertion` hangs after success.** Observed printing
    `[SUCCESS] Insertion complete on attempt 3` and then not exiting. Blocks unattended chaining
    of the assembly.

### Opened 2026-10-14

0d. **[2026-10-15: cause found and fixed — see State of the system: the slot is the line through
    the seated U parts; aim follows it; the nudge flattens the rest.]**
    **line_green ends tilted after some inserts (3 of the last 4).** Depth seat check: pitch
    −2.45° / +3.3° / +3.0°, one end ~5 mm high. A fresh regrasp from the clear area gave the same
    tilt, so it is in the seat, not the grasp. Suspect: a U part (whose regrasp took the other,
    fold-symmetric arm, or which needed an extra staging pass) seated the "other way round" so the
    bar end rests on it — **unverified**. `seat_check.py` is **confirmed by the operator**: every
    time it reported line_green off, it was off (and flat readings were flat). Trust it as the tell.
    Operator's route for a real tilt (QUEUED_FIXES #13): approach the bar **at its measured angle**,
    grasp, safe height as held, fix orientation, reinsert. The top-down reseat cannot fix a tilt.
0e. **Constant-ish −y seat offset** (QUEUED_FIXES #12): ~2 mm world −y bias plus a few mm of
    run-to-run scatter; suspected camera mount (hand-eye). Operator: fix later.
0f. **U parts sometimes tip over when set down in the clear area** (u_orange 2/3 cycles, u_brown
    1/3); the extra pass always recovered. Cause not investigated.
0g. **D415 USB-C plug is worn**: dropped off USB twice during wrist rotations; flipping the plug
    restored 5000 Mbps. The replay refuses a grasp when the camera is not streaming.

### Opened 2026-10-17/18

0h. **line_green tilts ~3.5° after nearly every insert** (one end ~6 mm high); the nudge fixes it
    every time, so the loop passes, but it costs ~1 min per cycle and the cause is not found. The
    operator's account: wedged by a slight roll in an unusual direction.
0i. **Yellow misread streaks in the localizer** (below-table pose for 8–15 s after arm motion) are
    worked around in `move_to_grasp`, not fixed at the source (`aruco_camera_localizer` IPPE choice /
    temporal state during motion). The 30 s wait would not cover a longer streak.
0j. **Vention 7th axis is in E-STOP** (safety-module fuse LED red; its safety box is not wired to the UR
    controller; operator ordering a cabinet key). The carriage is not held, so the base creeps in the
    robot frame — tolerated by the base re-reads. Do not use armctl `connect()` (it auto-releases the e-stop).
0k. **Seated u_brown's marker is edge-on from the base view** — handled by the second look from home;
    a better base view would make it unnecessary.

### Critical-path

1. **Unify the two insertion code paths.** `compliant_insert` (FSM autonomous SEARCH, used for u_brown/u_orange) and `prismatic_peg_insertion` (used for line_green/yellow) live separately and have evolved different compliance configurations. The line_green/yellow case needs Rx/Ry compliance + geometric depth-based exit; the u_brown/u_orange case needs XY-locked descent. Either fold both into one wrapper with per-object compliance config, or formalize the two-track split with a clean interface. Tracking note: `prismatic_peg_insertion.py` lives under `_real_mode_stash/` (force-added to git) — eventual home should be a tracked location like `primitives/inserts/`.

2. **Single-shot reliability is not 100 %.** Today's full assembly required:
   - Manual relaunch of `grasp_points_publisher` after it died silently.
   - Retry on a `force_mode_controller_did_not_activate` (>15 s timeout) — `replay_real_assembly.py` has no built-in retry.
   - Re-run `--only u_orange` after a one-shot camera-detection dropout in `move_to_grasp` (single-poll, no retry).
   Suggested fixes: camera-detection retry loop in `move_to_grasp`; auto-restart `grasp_points_publisher` on death (or fold into replay startup); replay-script-level retry-on-fail.

### Nice-to-have

3. **Two-stage insertion** (operator-flagged): when one part has to clear an alignment phase before the final slot. Spiral can't apply pressure on the underlying object (it'd move). FSM has experimental support but never engaged in a real run.

4. **Sign convention root cause.** `verify_baselink_motion.py` validated single-axis convention; multi-axis SEARCH inverted in practice. The `-Fmax * unit(error)` sign-flip works empirically but the physics gap is unresolved.

5. **Per-object F_press tuning.** Currently 5N for u_brown/u_orange and 1.5N for line_green/yellow (`PER_OBJECT_INSERT_FORCES_N`). Could be auto-tuned from observed rim-contact wrench amplitude.

---

## Reproducibility checklist

To replicate the working full-assembly run (all 4 objects), **from `ur5e-fmb1-demo`**:

1. **Bringup**: `bash compliant_insertion_studio/scripts/stack_up.sh` from that tree. It enforces
   the order (driver → gripper → camera → grasp points) and verifies each stage. Preconditions:
   robot powered, brakes released, pendant in **REMOTE CONTROL**. No program is loaded and nothing
   is played on the pendant — the driver runs headless. (See `CLAUDE.md` §1 for the per-stage
   commands if you need to bring one up by hand.)
2. **Verify pre-flight**: `ros2 topic echo --once /objects_poses_real | grep child_frame_id` shows all 4 parts; `ros2 topic hz /grasp_points_real` ~5 Hz; `ros2 topic hz /tcp_pose_broadcaster/pose` ~500 Hz; `ros2 control list_controllers` shows scaled_joint_trajectory_controller active.
3. **Sequential replay, fail-stop**:
   ```bash
   python3 -u ablations/replay_real_assembly.py --assembly-json <JSON> --only u_brown
   python3 -u ablations/replay_real_assembly.py --assembly-json <JSON> --only u_orange --skip-startup
   python3 -u ablations/replay_real_assembly.py --assembly-json <JSON> --only line_green --skip-startup
   python3 -u ablations/replay_real_assembly.py --assembly-json <JSON> --only inverted_u_yellow --skip-startup
   ```
   On any non-zero exit: stop, inspect `ablations/logs/replay_*_<ts>.log` for the failure reason, address root cause, then continue. Do NOT advance to the next object on failure.
4. **Calibration** (one-time per fixture): `DEFAULT_BASE_POSITION` from a single u_brown insert; per-object delta in `PER_OBJECT_BASE_OFFSET_M` from observed seat TCP vs predicted (see `primitives/shared/config.py` for the calibration arithmetic of each object).
5. **Forensic analysis on failures**: `compliant_insertion_studio/logs/insert_<obj>_<ts>.csv` + `.meta.json`. Schema-v1 columns documented in `wrapper/schema_v1.py`. Both insertion paths now write the same schema, so the analyzer dashboard ingests both.

---

## Anti-patterns the current session worked through (DO NOT repeat)

- ❌ Hardcoding "bias" in FSM from un-centered demos. (Conflates b_obj + g.)
- ❌ Using `-F_lat` or `-r_cop` as autonomous direction signal. (Friction positive-feedback loop.)
- ❌ Position-error PD with default Kp=350. (Below stiction at typical errors.)
- ❌ Spiral starting at θ=0 with center at peg landing position. (Initial direction = arbitrary +X.)
- ❌ Excluding SEARCH/APPROACH from global seat detector. (Misses peg-already-seated.)
- ❌ Tilt-relax detector. Tilt measured < 0.01° throughout in the GUIDED dataset.

  > **CORRECTED 2026-08-16.** The "< 0.01°" figure is a **consequence of the rotation clamp**, not
  > a measurement of contact physics. SEARCH commands `selection_vector = (T,T,T,F,F,F)` —
  > rotation is LOCKED, so of course the peg stays face-down. Do **not** cite this number as
  > evidence that tilt steering is impossible, and do **not** cite it as grounds for unlocking
  > rotation. When rotation *was* unlocked on 2026-08-16 tilt did become responsive (excursion
  > 0.004° → 0.25°, peak 0.59°) — still far below the ~3° GOLD shows at chamfer engagement, and it
  > did not change the outcome. It did break the insert: see the next entry.

- ❌ **Unlocking rotation in SEARCH.** `selection_vector` must be `(T,T,T,F,F,F)` there. `SKILL.md`
  §10 states all-True as a hard rule; that rule is **wrong for SEARCH** and following it broke the
  insert for five consecutive real-arm runs. With rotation compliant, lateral force applies a
  **moment about the grasp point** — the part pivots in the jaws while the gripper translates, so
  TCP displacement stops being peg displacement and every swept-area / coverage figure computed
  from TCP becomes fiction (that is how one session concluded "the hole is not within 6 mm" about
  a hole 3.38 mm away). Every 2026-05-07 run that seated commanded rotation LOCKED — check the
  `cmd_wrench_raw` sidecars before trusting any written rule to the contrary. The all-True rule
  still holds where it was derived: the XY-locked *position-tracked* case refuted at v91 is a
  different thing from rotational compliance during a lateral sweep.

- ❌ Sampling a force reference while the arm is still settling, or shortening a settle /
  step-back window for speed. See "What's binding" above.
- ❌ Single-sample threshold-crossing contact detection. See "What's binding" above.
- ❌ Freezing spiral at chamfer-edge dip. (Doesn't help; needs ACTIVE control to push peg into chamfer.)
- ❌ Per-slot bias hardcoded from non-centered demos. (Same conflation as global bias.)
- ❌ Tuning parameters without first analyzing data. (Project rule per CLAUDE.md.)

Each of these was tried and falsified by data this session. See `ITERATION_TRACE_2026-05-06.md` for the data.

---

## File layout (you'll need these)

```
CLAUDE.md                                         # repo root — project conventions (binding).
                                                  #   NOT compliant_insertion_studio/CLAUDE.md;
                                                  #   no such file exists.
docs/QUEUED_FIXES.md                              # repo root — open fixes from 2026-08-16
                                                  #   (ur5e-fmb1-demo only)

compliant_insertion_studio/
├── .claude/skills/insertion-control-law-derivation/SKILL.md  # methodology (binding)
├── docs/
│   ├── HANDOFF_NEXT_AGENT.md                     # this file
│   ├── COLLECTION_METHODOLOGY.md                 # operator-demo pipeline
│   ├── AUTONOMOUS_INSERTION_METHODOLOGY.md       # working autonomous architecture
│   ├── ITERATION_TRACE_2026-05-06.md             # full reasoning trace
│   ├── SCHEMA.md                                 # CSV+meta schema
│   └── SETUP.md                                  # cold-start runbook
├── analysis/
│   ├── CONTROL_LAW.md                            # v4 Found Hole detector
│   ├── SEARCH_CONTROL_LAW.md                     # historical -r_cop director (refuted)
│   └── AUTONOMOUS_RUN_LOG.md                     # per-iteration journal
├── wrapper/
│   ├── compliant_insert.py                       # main FSM wrapper
│   ├── contact_search_fsm.py                     # FSM + SearchDirector + FoundHoleDetector
│   └── cad_lookup.py                             # predict_tcp_at_seat
├── scripts/
│   ├── stack_up.sh / stack_down.sh               # one-command cell bringup / shutdown
│   ├── run_assembly_step.py                      # full pick→insert pipeline
│   ├── regrasp_held_object.py                    # camera-driven regrasp
│   ├── loop_autonomous_insert.sh                 # multi-iteration test harness
│   ├── collect_regime_data.py                    # GUIDED-mode demo collection
│   ├── launch_robot.sh                           # robot bringup with set_payload (--headless)
│   └── launch_camera.sh                          # aruco_camera_localizer
└── configs/
    └── defaults.yaml                             # FSM cfg (search params, v4 thresholds)

primitives/shared/config.py                       # DEFAULT_BASE_POSITION + grasp lookups
ablations/eval_resources/fmb1_assembly.json       # per-(object,grasp_id) widths
```

`stack_up.sh` / `stack_down.sh` and `docs/QUEUED_FIXES.md` exist on `ur5e-fmb1-demo` only.

If you ask "where's X?", the answer is above.
