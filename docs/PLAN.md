# PLAN: rulings, standing approvals, facts

Project: the UR5e + RG2 FMB cell (this repo, branch `dev`), rescued into DroidStack. Lead session: "FMB1 live performance
context" (`a7f374cd-d7c8-4d94-acfd-b65395804ed3`) on the cell PC `aldrin-OptiPlex-3060`. Operator: Aldrin. WhatsApp group:
"Dimelab". Lead's manual: `~/.claude/skills/autopilot/references/operate.md`.

Rulings are the operator's. Each gets an id and a UTC time; they're applied without re-asking.

## Rulings
- **R1 Rescue with DroidStack** (2026-11-18): "droidstack is mature enough for you to get started we are going to rescue this
  codebase, onboard it with droidstack and do the work ourself". The lead and its workers do the work; this is not a
  prompt2assembly dispatch.
- **R2 Perception** (2026-11-18): "pose6d lets keep that for later and continue using aruco camera localizer and document things so
  that when pose6d needs informatiron we have whatever we can provide". Notes for pose6d: `docs/POSE6D_NOTES.md`.
- **R3 Sim before the cell** (2026-11-18): "ask plan47 about droidstack usage so we can setup the robot with the new camera mount prove
  the new code in sim before we can deploy it on the cell". New code reaches the real cell only after it is proven in a DroidStack
  simulation.
- **R4 Learn from robosort** (2026-11-18): "you can ask robosort agent how it is rescuing a workspace that is what we will also be doing
  in a way".
- **R5 Separate from DroidStack** (2026-11-18 ~21:00Z): "we will be learning a lot about real world deployment that is relevant for droidstack
  which is why we work separately so the droidstack work can happen on the side without issues". This project runs on its own;
  real-deployment lessons are written down for DroidStack (kept here, see R10), and DroidStack's own work proceeds in parallel.
- **R6 Public repo** (2026-11-18): "leave the repo as is not an issue right now" (ros-mcp-server stays public with its current files).
- **R7 WhatsApp** (2026-11-18): "whjatsapp yes we alreaduy have a group so we continue there": the Dimelab group.
- **R8 Grill first** (2026-11-18): "we are yet to grill on what we are going to do": the scope and what "done" means are settled in a
  grill before workers start.
- **R9 Done** (2026-11-18 ~21:10Z): option A. FMB1 assembles hands-off again on the real cell with the new (v3.1) camera mount;
  every new piece of code (camera transform, view poses, picks) is proven in a DroidStack sim cell on dual before it touches the
  robot; a written record of what real deployment taught us is kept for DroidStack.
- **R10 Hold DroidStack requests** (2026-11-18): "we dont message droidstack to plan47 yet we figure out what to do ourself and then we
  can do a final desciision before we decide what to do". No DroidStack requests, needs or lessons go to plan47 until the operator's
  final decision; collect them in `docs/DROIDSTACK_LESSONS.md`. (The FMB1 phase-2 needs in P47-6 were sent before this ruling.)
  Card/box coordination on trouble (P47-5, P29-1) still goes to plan47 / plan29.
- **R11 Approvals** (2026-11-18): "approvals yes as written": the standing approvals table below is in force.
- **R12 WhatsApp via /text** (2026-11-18): "my whatsapp is already logged in mac use /text": post with
  `~/.devdroid/text/runtime/scripts/text.cjs send 120363415188889698@g.us` (Dimelab).
- **R13 Bar** (2026-11-18 ~21:20Z): option A. Done = 3 hands-off certified FMB1 cycles in a row on the real cell from a normal
  (untipped) reset, with the v3.1 mount calibrated. No cycle-time target; tipped starts and speed come after.
- **R14 Camera mount in the robot model** (2026-11-18): "we also have to import our camnera mount cad into what we have in prompt to
  asemly becuase it doesnt have that yet in the env so that the planning program written doesnt involve in the camera mount hiting
  the table". The v3.1 saddle + tower and the D415 body join the robot's collision and visual geometry, so planning keeps them off
  the table. Done in OUR lab's copy of `robots/ur5e_rg2` (P47-1: copy, don't change the shared record; R10: no request to plan47).
  CAD source: robo-chain `.scratchpad/jobs/rg2-camera-mount/cad/` (camera_mount_v31_rg2.step, the DIME assembly STEP, the D415 STEP).
- **R15 Elevated robot base** (2026-11-18): "we have to fix where the ur5e is mounted with extra height liek we already planned for
  what we are doing ewith droidstack". The robot's base sits 0.080 m above the table top, as in the 10-23/10-26 elevated-bench work
  (`compliant_insertion_studio/m4_dispatch/environments/dimelab_elevated/`, `realcell_env/MEASURED_CELL.yaml`
  frames.base_link_above_table); the lab's environment declares it.
- **R16 Deadline** (2026-11-18): none. P1 -> P3 as fast as the evidence allows; the lead tells the operator in Dimelab when P3
  needs him at the cell (plugging the D415 in, the real runs' go).
- **R17 OnRobot models may be published** (2026-11-18): "dont worry about the do not publish we are going to be publishing i already
  told plan47 that is why we are releasing it with paper itself". The OnRobot RG2 / quick changer models are released with the
  paper; the "never public" part of P47-1 for them no longer binds our work. Keep NOTICE-OnRobot.md beside them wherever they go.
  The cell's own source for the RG2 is the lab's DIME-LAB packages (onrobot_rg2_description, onrobot_quick_changer_description,
  onrobot_rg2_control: the CAD meshes, the 4-bar arc width <-> joint model); the paper's ur5e_rg2 was reconciled against them.
- **R18 All boxes** (2026-11-18 ~22:00Z): "mac, a4500, dual-a4500, rtx6000 all of them are for use". Supersedes P29-2's "a4500 stays
  off for us". The box rules still apply on each (check the card right before, a hold where a hold tool exists, never touch another
  user's work: a4500's local user and rtx6000's borrower win). P29-1 (dual) is unchanged: on dual, Isaac/ROS only through DroidStack.
  Off dual, nothing simulated may reach the cell's DDS domain: an Isaac run of ours either loads no ROS bridge, or runs its whole
  ROS graph host-local (ROS_LOCALHOST_ONLY=1 and its own ROS_DOMAIN_ID) on a box that is not the cell PC (lead, 11-18).
- **R19 DroidStack issues** (2026-11-18 ~22:10Z): "you can interact with plan47 when needed to launch and issues with droidstack or you
  fork and fix it yourself isaacsim for the planner and you dont have to start from scratch because we already have a codebase".
  Relaxes R10 for launch blockers and DroidStack issues: message plan47 (claude 13eddc on dual), or fork DroidStack and fix it,
  handing the fix back through plan47. The planner is proven in Isaac Sim, built from the code we have (this repo's planner,
  the paper's Isaac twin), not from scratch.
- **R20 Real pick/place tests now** (2026-11-18 ~23:30Z): "im assuming trying to grasp and move home or place it down doesnt have
  to wait for the new program to assemble". The cell's existing, proven code (primitives, replay) runs real pick / place / home
  tests on the v3.1 mount now; R3's sim-first gate applies to the NEW planner delivery. Each motion run still gets his go.
- **R21 Prepare a4500 and rtx6000 as DroidStack boxes** (2026-11-18 ~23:55Z): "both a4500 and rtx600 has to have that setup that
  is the whole ponit". Both get `droidstack boxes prepare` (engine + supervisor under aaugus11, the --system-service root step
  where Ubuntu 24.04 needs it), so live checks / collision / renders run through DroidStack there too. The box rules hold:
  a4500's local user and rtx6000's borrower win; check the card right before; disk floors; never touch others' processes.
- **R22 Sim proof = the full FMB1 assembly in the twin** (2026-11-19 ~00:30Z): "sim can perform the full assembly fmb1 has been
  solved so many tiumes in my paper end to end without compliance ... a regular insert will reach the target because all the pose
  is knownw ... as long as i can see a assembly performed in dt and rendered shown in whatsapp we can use that ... the seam where we
  use execute mate or drive mate ... to perform the insert with compliance". The R3 bar: the cell's code assembles FMB1 end to end
  in the digital twin with a regular (non-compliant) insert, rendered and shown in the Dimelab group. Compliance in sim is queued
  for later. On the real cell the insert stays compliant, through the drive-mate / execute_mate seam. Wanted ("would be nice"):
  paper2's idea - ground-truth poses only for what the camera can see, so the planner works as it must on the real cell.
- **plan47, measured** (2026-11-19 ~01:20Z): dual's LIVE engine is 989cff8d9172 (platform 3e6b494c, prepared 11-17), both cards kit +
  render_warm OK; our refusals named our old abbd0a57. So the cell PC installs 3e6b494c (not dev f1748570), re-submits to dual today,
  and re-prepares a4500 from 989cff8d9172. Not on dual yet: ROS (no --ros-distro) and card-scoped image steps (no --probe-image) -
  tell plan47 if our road needs them; they go into dual's next re-prepare.
- **plan47 correction** (2026-11-19 ~01:00Z): the system-mode Kit/render_warm/ROS probes ARE built on DroidStack dev (a52865ca,
  1a321bf6, dbc2274e, 11-13..15); our engine abbd0a57 (platform 2648fe53, 11-13) predates them, and its cardless sizes are unmeasured
  bounds (dev: collision model 1024 MiB, isaac-python:usd 256 MiB). Fix = upgrade the cell PC to dev (platform f1748570, engine
  5dcce0724ab4) and re-prepare a4500 from it; plan47 re-prepares dual. Lane A retired. rtx6000 flags when space is reclaimed:
  --system-service, our own Isaac 6.1 under a root-controlled path, --ros-distro humble, --disk-floor-mib 20480.
- **plan47** (2026-11-19 ~00:30Z, superseded by the correction above): the system-mode kit/render_warm/ROS probes are DroidStack's top priority; our worker may be named a
  DroidStack lane to build them (GPT builds, Claude reviews, droidstack-opus lands) - no private fork. Cardless class sizes sent for a
  ruling. rtx6000: HOLD until the builder names flags + disk floor (plan29 reclaiming box copies there). a4500 one-off renders fine.
- **R23 MuJoCo road** (2026-11-19 ~01:40Z): operator: "work on gazebo and mujoco ... get started on a fork or however plan47
  recomends". plan47: as DroidStack work, no fork and no cell-side bridge - lane ds-mujoco-robot-cell on dual (off ds-mujoco 972b0a72,
  brief .pa/BRIEF.md): slice 2 = a declared robot cell under MujocoAdapter with the ROS 2 robot contract when declared; design note
  first, Claude review, plan47's ruling, build after ds-mujoco slice 1 lands. The cell's MJCF is lab data (our lab). Gazebo later,
  through the same SimulatorAdapter seam. Worker w4-mujoco.
- **Cable** (2026-11-19): the D415's cable runs over the links to the PC. Operator wants it modelled and guarded: a cable test (PC end
  unplugged, operator watching, slow joint sweeps) -> per-joint limits enforced in one place + the UR safety config; awaiting his
  photos and a time. Today only move_cartesian caps |wrist_3| at 4.0 rad.
- **plan47 ruling, MuJoCo slice 2** (2026-11-19 ~02:20Z, lane .pa/RULING-plan47-r4.md): design r4 APPROVED to build after ds-mujoco
  slice 1 lands; phases A (route + realizer + fidelity, no ROS), B (planner road + ROS contract), C (wrench + FMB two-stage
  acceptance); grasp = FRICTION with a measured slip bar (weld never counts toward acceptance); tool_wrench extends robot-binding v2 in
  place. Four rounds of independent Opus review (lane .pa/REVIEW-r1..r4.md).
- **R24 Hold live tests** (2026-11-19 ~02:50Z): "lets hold off on the live test let me see the assembnly video in mujoco first". No
  real-cell runs (the u_orange retry, the cable test) until the operator has seen an FMB1 assembly video in MuJoCo.
- **R25 MuJoCo video the DroidStack way** (2026-11-19 ~03:05Z): "we can do it with A itself then but do you really have to wait for it to
  land before use?" - no interim connector; the video comes from the cell code on DroidStack's MuJoCo road (ds-mujoco-robot-cell
  phases A-C). Start as early as plan47 allows (asked: phase A on ds-mujoco's gate-passed head c1f21ed2, rebase at landing).
- **R26 No part masses** (2026-11-19 ~03:20Z): "I dont have weight of objects what I can tell is a friction grasp doesnt slip". The
  sim slip gate bounds the mass instead of measuring it: CAD volume x solid PLA density as the upper bound, 20 % of it as the lower,
  no slip at both ends at the real 40 N grip; the real grasp's observed no-slip is the reference; no friction tuning unless a declared
  physical value is wrong.
- **R27 Use the fork now, land later** (2026-11-19 ~07:20Z, WhatsApp): "there is no point in waiting for droidstack to land ... i dont
  see why we arent using the fork to do the work we can incorporate it in droidstack dev branch later". The MuJoCo assembly video is
  built and RUN from our lane fork ds-mujoco-robot-cell (slice 1 c1f21ed2 + Phase A + Phase B + C) without waiting for any landing or
  plan47's live-graph word; landing into DroidStack dev follows later through plan47's gate. Runs off dual (P29-1): a4500 CPU (or
  the Mac), ROS host-local (R18), never on the cell PC's ROS network. plan47 informed, not asked.
  plan47 (07:30Z): no conflict; conditions: localhost-only DDS + own domain, a4500's colleague wins (check CPU/memory, bounded
  process tree, Humble only in a container or on the Mac), keep receipts/logs/MJCF digests (no OnRobot meshes), each phase still reviewed.
- **R28 Claude worker for MuJoCo** (2026-11-19 ~14:50Z, WhatsApp "B"): the workspace hit its Codex spend cap at 13:14Z, stopping every
  Codex worker. The operator chose to continue the MuJoCo work on a Claude worker (w5-mujoco, session 8e48c8b3) rather than raise the
  cap or wait. Same goal, bars and binding rules as w4; it checkpoints w4's uncommitted work first.
- **R29 Fresh sequence planner instead of more patches** (2026-11-19 ~evening, WhatsApp): "if it's easier to assemble it with fresh
  code with the rules we can do that ... I feel like we are bolting fixes on top of fixes ... maybe even cleanup ... unnecessary math or
  calculation dragging down with multiple rotations". Lead recommendation taken: one new module plans a part's whole
  pick -> reorient -> place-down -> regrasp -> reorient -> insert, checks every candidate against ALL rules in one place (camera side,
  mount/D415 vs table and arm, held part, physical self-collision, wrist_3/cable limits) and minimises total wrist travel over the
  sequence; primitives become thin executors of planned joint targets (goal_guard kept). Perception, gripper, compliant insert unchanged.
  Old per-step planning stays until the new one beats it in sim, then is retired. Design note -> review -> u_orange in sim -> FMB1 on the
  staging layout -> review before any real motion.
- **R30 Record without the camera unless asked** (2026-11-19 evening): "yes record without camera unless I ask". record_demo_bag.sh
  now records poses, joints, wrench and gripper only; CAMERA=1 adds the colour stream (and DIAG=1 + CAMERA=1 the aligned depth).
  The 150 GB of October bags were stripped of image topics on his word ("clear the fmb1 video and image stuff"); the mp4 renders
  were deleted. Receipts: ~/Documents/fmb-cell-lead/out/cleanup-20261119-*.
- **R31-R33 (2026-11-20 ~04:20Z, operator in the terminal)**: R31 "the objects don't slip on grasp or whatever motion you are doing":
  the twin holds a verified grasp rigidly (weld from the verified close to the open; semantics matched to DroidStack/prompt-to-assembly via
  plan47); the no-slip measurement stays as a regression check. R32 "there was no problem with the range after our cable check": no
  per-part wrist_3 budget; hardware limits and the existing |wrist_3| cap only, wrist travel stays in the planner's cost. R33 "drop the
  object straight down ... has worked well for us in the past": place-downs release at move_down's floor as the real cell does.
  RG2 "40 N" per finger vs total: not stated in OnRobot's public datasheets/manuals (searched 11-20); moot for the twin under R31.
- **R34 (2026-11-20 ~11:50Z, operator on WhatsApp)**: "this is not the same program I asked you to refer earlier". Use the PROGRAM behind the
  prompt-to-assembly Peg-in-hole board video itself (Opus authoring run 1966d686, dual:~/prompt2assembly/.pa/runs/1966d686.../delivery:
  planner.py, motion.py, urkin.py, rosio.py, decl.py), ported to our MuJoCo twin and cell, with the camera kept off the table. Borrowing its
  ideas into seqplan (the lead's 11-20 05:01Z recommendation) was not what he asked. Our hard rules go in only as vetoes on its choices
  (camera side, mount/D415 clearance, held-part, cable window with wrist_3 not the cheapest joint, calibrated DH); the insert stays ours.
  Its code is private: it lives in the lab (no remote), never in this public repo. The lead checks every video frame by frame before it
  reaches the operator ("you should be seeing this and asking agents to fix it").
- (lead, 2026-11-20 ~19:00Z, under the operator's 10-19 "read once, then chain"): the board port may read a held part ONCE at the insert
  hover and use it only if it agrees with the chained pose within 3 mm / 3 deg; otherwise it refuses and stops (no harmful fallbacks).
  The read must pass the cell's still-read rule (>= 0.8 s still, >= 3 fresh detections, median). RGB-only at ~0.12 m on the v3.1
  mount is untested: first-real-run checklist.
- **R35 (2026-11-20 ~19:25Z, operator on WhatsApp)**: "A". The first real FMB1 run is driven by the board-video program (R34's port) on
  road (c): RealIO over the cell's proven seams (drive-mate inserts with the post-insert gate and persistent latch, the Grip contract
  with witnessed holds, move_plan/move_down behind the one checker), one part at a time, u_brown first, with the operator present.
  Before the first move: rebuild pa_gripper_msgs in the live workspace (operator's go), a read-only start capture, then his go.
- **R36 (2026-11-20 ~19:45Z, operator on WhatsApp)**: before the first real deploy, (1) a twin VIDEO of the board program on the real
  captured scene for him to gauge, and (2) a Codex review of the real-arm code run on his Mac (this PC's Codex budget may be spent).
  The pa_gripper_msgs rebuild needed no call from him (no motion, nothing running used it): done 11-20 by the lead, backup kept.
- (lead, 2026-11-20 ~23:30Z) RELEASE CANDIDATE for the first real FMB1 run: lab fmb-cell-lab-w8 tag rc3b-fmb1-first-real (415a606) on
  cell dev 629208a; record ~/Documents/w8-out/rc3/RC.md. Opus final review RC1 -> RC2 -> RC3b: VERDICT GO FOR SUPERVISED FIRST RUN
  (fmb-cell-lead/out/review-rc1-final.md). Still gated on: the operator's go and presence; yellow >= 10 cm from base1 and the FMB3
  pieces off the table; the R36 Codex review on his Mac (asleep); a re-capture + dry-real on the prepared table (no code change).
  Run: u_brown only; stop only with stop_run.sh, e-stop in hand.
- (lead, 2026-11-20 late) RC3c REPLACES RC3b: lab tag rc3c-fmb1-first-real (d4d9468), same cell dev 629208a. The 9-layout physics panel
  stopped every run at step 1: the startup opened the jaws with width_max (110 mm raw), which grip_adapter refuses (> OPEN_M + 5 mm)
  and control_gripper rejects (0-100 mm). On the real arm RC3b would have homed, then stopped at home (grip_open_failed; safe, nothing
  held). Fix: the startup commands OPEN_M (0.100 m), and the jaws are still modelled at width_max. The test is pinned to
  grip_adapter's OPEN_M. Delta reviewed by the lead: GO (one planner line plus twin-only --noise-view). DryIO accepted any width, which
  hid this; w8 is adding the adapter's band to DryIO. The other gates above are unchanged.
- **R37 (2026-11-20 ~17:40 local, operator on WhatsApp)**: "Okay let's try u brown then", a go for the RC3c u_brown run. The lead
  asked before any motion for: his presence with the e-stop; yellow at least 10 cm from base1 and the FMB3 pieces off (the D455 at
  17:41 showed neither done); the R36 Codex review waived for this supervised try or done first (lead recommends waiving for u_brown
  only, and doing it before the four-part run).
- **R38 (2026-11-20 ~17:50 local, operator on WhatsApp)**: table prep is the robot's job, not his: "make sure you have all the
  capabilities to do operations like this". We have recovered parts like this before, and even removed one part from on top of
  another. Now: the old code's `disassemble.py --relocate inverted_u_yellow --grasp-id auto` (dry run first, spot shown to him), with
  him present. FMB3 pieces stay unless the dry run shows a move near them. Queued after the first run (w8): the board program's own
  pre-step moves any part within 10 cm of base1 out of the way before the assembly.
  CORRECTED the same hour (lead): disassemble.py --relocate reads the part from a top-down hover, which the v3.1 D415 cannot see
  (memory d415-v31-view-limits), so it is out. w8 builds `run_board --relocate PART` plus relocate-then-assemble now, BEFORE the
  first run: twin run on the realscene-5 scene, then an Opus delta review (new motion), then rc4 flies with him present.
  rc4 = lab tag rc4-fmb1-relocate (148552b), with --survey-only and a fail-closed depth free-spot check. Opus review: NOT READY
  (fmb-cell-lead/out/review-rc4.md). P1: --only waived the off-surface flags for a relocated part, and the set-down at read height
  + 8 mm could drop it unbounded. P2: free fall and topple spacing, the depth margin narrower than the finger model, an ambiguous
  stop code after the pick, no end-to-end test. w8 is fixing these as rc4b; the same reviewer re-reviews the delta.
  rc4b (5d6e734) re-review: NOT READY (out/review-rc4b.md). The code fixes hold, but the live localizer snaps tilt <= 10 deg and
  z along the ray within 50 mm (filter_config.yaml, checked by the lead), so a slightly leaning or raised part reads as flat:
  level_rest can't see it. For the first run (rc4c): the operator checks by eye that the parts to relocate rest flat and touch
  nothing; --survey-only writes each part's depth top vs its CAD top. Queued: an in-program depth rest check. P2s: depth margin
  15 mm around the footprint; open a held part only within ~10 mm of its release height.
- (lead, 2026-11-20 ~23:30 local) RELEASE CANDIDATE rc4c = lab tag rc4c-fmb1-relocate (5617982) on cell dev 629208a; record
  ~/Documents/w8-out/rc4c/RC.md. Opus re-review: VERDICT GO FOR SUPERVISED RUN (out/review-rc4c.md; RC.md wording fixes only).
  Sequence: (a)+(b) --survey-only (homing, views, two depth reads, no pick) -> (c) dry-real on the live capture (on dual) ->
  operator's by-eye check that u_orange and yellow rest flat and touch nothing, plus the YAML snap grep and mtime check -> (d)
  --relocate-near-base --only u_brown. Gated on: his presence with the e-stop; the R36 Codex review (asked to waive for this try).
  Supersedes RC3c.
  11-21: phantom audit (review-nest-rule P1: the live localizer slides parts within 50 mm of table rest along the ray, seated
  parts included). For rc4c's u_brown run, no motion decision uses a seated read; worst case the safe `unobservable` stop
  (w8, dry-reals with a phantom DryIO mode, dual ~/w8-dry/phantom). Seated u_brown slides ~3.5 mm, not 28. Not a gate. Open: the
  table_z question: closed. Live robot_config.yaml table_z -0.080 = cell TABLE_HEIGHT -0.080 (-0.0825/-0.0805 were a .py default
  and a stale docstring); the first survey records the depth plane anyway.
- (2026-11-21 ~03:00 local, operator on WhatsApp) "Why is there so many movements from home to move to grasp without grasping".
  Lead's answer: the reads need aimed view hovers (v3.1), and there are three wastes: the pick hover -> view hover -> pick hover
  round trip, a duplicate pick-hover leg for yellow, and a depth trip to the staging view per relocation. Cuts on branch
  w8-fewer-moves (review before flying). Lead recommends flying u_brown on the reviewed rc4c1 first; asked him.
- (lead, 2026-11-21 ~04:00 local) RELEASE CANDIDATE is now rc4c1 = lab tag rc4c1-fmb1-relocate (009dfe2), cell dev 629208a,
  record ~/Documents/w8-out/rc4c1/RC.md. The rc4c -> rc4c1 delta was checked by the lead: on the real road, record-only fields in
  --survey-only (survey-rest-depth.json table planes, only in run_board's survey-only branch). Everything else is twin-only
  (twin_dh), dry-only (phantom, release drop) or tests. Lab suite 100 passed / 2 skipped; dry-reals unchanged. The rc4c GO carries
  over. Twin DH re-runs: all 12 hover reads 0.68-2.30 mm, so the panel's hover stops were the twin's nominal FK.
- fewer-moves (w8-fewer-moves e851aa8): Opus review GO FOR SUPERVISED RUN (out/review-fewer-moves.md). Dry-real 56 -> 45 legs; twin
  u_brown seated 53 s sooner. P2-1: the depth reuse only for the first relocation and < 60 s old; P3s (direct transits start/end at
  SAFE_Z, fallback tests). Fixed as rc5-fmb1-relocate, the lead checks the delta, then it's offered to the operator with rc4c1 as the
  fallback.
- (lead, 2026-11-21 ~05:45 local) RELEASE CANDIDATE rc5 = lab tag rc5-fmb1-relocate (c427da8, branch w8-fewer-moves), cell dev
  629208a; record ~/Documents/w8-out/rc5/RC.md. The delta e851aa8..c427da8 was checked by the lead: survey depth reused once, < 60 s
  and nothing released (else a fresh staging-view read before the plan read); direct transits need both ends >= SAFE_Z - 5 mm;
  survey-only has no extra read. Lab suite 109 passed / 2 skipped. Dry-real 56 -> 46 legs, 95 -> 85 steps; twin -52.7 s
  (fewer-moves run). rc4c1 (009dfe2) is the fallback. The same live-table sequence and operator precondition apply.
- **R39 (2026-11-21 ~07:50 local, operator on WhatsApp)**: "Okay let's do the arrangement in the real robot and then send me photos
  of the scene": rc5 relocate u_orange + yellow only (no u_brown insert), photos after. Lead asked whether he or someone is at the
  cell with the e-stop (R35's supervised first run) and for his eye check of yellow's lean. Run from a clean detached worktree
  ~/Documents/fmb-cell-lab-rc5 via ~/Documents/w8-out/rc5/live/run.sh (survey -> dry-real on a4500 by w8 -> relocate).
  11-21 ~07:50 he said "Yea go ahead". The rc5 survey ran on the real cell (homing, 3 views, no contact) and stopped fail-closed,
  unobservable_at_start: (1) base1's depth CAD fit was -4.5 mm off the table height. ablations/base_icp.read_icp skipped
  config.depth_to_m (the 11-18 D415 correction). Fixed, dev fef912f (raw -67.0 -> corrected -61.9 vs -62.5, same frame). (2) Yellow
  was not seen from any view; w8 is investigating. Log ~/Documents/w8-out/rc5/live/survey.log.
- **R40 (2026-11-21 ~08:00 local, operator on WhatsApp)**: "confirm if the tool0 ee position doesn't go below the max allowed
  limit of the runs we did when the real cell was running". Lead's check over 547 October real-run motion logs: lowest overall
  -0.2 mm (a horizontal set-down at the clear area; rc5 0.5 mm, the same move); lowest top-down 147.8 mm. rc5's flat-U grasp
  descends to 142.6 mm (judged at the on-part width; the cell's floor at the 64 mm close command is 144.9). Rule as rc5b: top-down
  tool0 >= 147.8 mm and the floor judged at the commanded close width; any pose >= -0.2 mm.
  Under R40 the board's flat-U grasp (across 70 mm) fails the pads-hold rule (pad centre 5.1 mm ABOVE the part's top at 147.8).
  Lead ruling: relocate a flat U with the cell's proven leg grasp (fmb1_assembly.json grasp_id 1, the cell's runtime grasp
  depth). The stand-up from the 70 mm grasp stays open. Yellow unseen: from the survey's 4 stances it sits in a blind area
  (x < -0.16 or the finger-hidden strip near the robot base); rc5b adds observing every missing part at every stance plus a
  coverage search (up to 3 aimed view hovers), fail-closed.
  11-21 (operator: "Yellow is in the reorient stage where it was placed last by you"): yellow is at the clear area (-0.300,
  -0.269), set down there on 11-18 16:25 by the lead's pick-reorient test (calib_bags/pickreorient_20261118_162509_inverted_u_yellow).
  That's the blind zone. The lead had wrongly told the twin it stood behind base1. It's ~340 mm from base1, so R39 relocates
  u_orange only. u_brown's stand-up needs the clear area: rc5b checks the set-down spot for parts and depth and stops by name
  if it's occupied. No "real scene" twin video without a real survey read.
- **R41 (2026-11-21 ~08:45 local, operator on WhatsApp)**: "that's why i said we set the objects up in staging area before
  assembling": the arrangement puts EVERY part in the staging region before assembly, not only the near-base ones. rc5b: a part
  outside the staging region or within 100 mm of base1 is relocated to a free, readable staging spot. Today u_orange and yellow
  move; u_brown and line_green stay. Assembly starts from the staging region every time.
- **R42 (2026-11-21 ~09:15 local, operator on WhatsApp)**: "Move the home after any activity": every run ends at home (success,
  survey-only, prep-only, and every fail-closed stop with empty jaws); a stop holding a part stays put for its recovery. The lead
  homed the arm from the R39 survey's stop (move_home --joint-space, jaws empty; tool0 133/-492/488 mm). In rc5b.
  He also asked about the ~2 h: the time is checking (suite, review, live survey, twin video), not "pick from anywhere".
- (lead, 2026-11-21 ~12:50 MST) rc5b = lab tag rc5b-fmb1-relocate (7aac13a) + cell branch w8-rg2-tip-measured (785a764, 6af2e27;
  the fingertip model corrected to the measured 230.86 mm closed tip). Suites clean (lab 124 passed with -n 8 in 446 s; cell 102),
  dry-reals: R39 u_orange success, R41 stage-all success (u_orange 282 mm, yellow 313 mm from base1), near-base u_brown stops at the
  stand-up as expected; all end at home (R42). Opus review running (briefs/review-rc5b.md).
  Stand-up ruling: port the cell's own leg-grasp reorient (with_regrasp: pick flat by the leg, rotate, place at the clear area with
  the cell's guarded 5 N touchdown via move_cartesian --max-force 5, regrasp grasp 1 top-down) on branch w8-standup-leg -> rc6.
  rc5b review: NOT READY (out/review-rc5b.md). P1: --prep-only fell through into the full assembly when nothing needed moving.
  P2s: the tip correction disabled tier 2's RG2 housing (fails safe to the box); no run-time check that the cell tree has the
  correction; finish_home trusts bookkeeping, not the gripper; R40 doesn't cover drive-mate (gate for insert runs). The fingertip
  correction, leg grasp, R40 coverage of planner moves, finish_home's signal/latch handling and the search were checked as correct.
  Fixes as rc5c (w8, before rc6).
  rc5c = lab tag rc5c-fmb1-relocate (77fca13) + cell w8-rg2-tip-measured 93bd6d1: P1 fixed (prep-only always stops), tier-2
  housing restored (cell 112 passed, 0 skipped), CELL_TIP_MODEL_UNCORRECTED start refusal, finish_home reads the gripper. Lab 133
  passed (-n 8, 437 s). Re-review running.
  rc6 stand-up: the sideways U can't reach a touchdown above R40/move_down's floor (the touch is at flange z -29.8 mm, the floor
  ~0.0). The cell released ~31.8 mm up, which was October's -0.2 mm stop. Lead: R33 already rules it ("drop at move_down's floor,
  12-30 mm, worked well"), so a declared drop, limit 37 mm (STANDUP_DROP_TOO_HIGH), the landing read + 2 passes decide.
- (lead, 2026-11-21 ~13:20 MST) RELEASE CANDIDATE rc5c: Opus re-review GO FOR SUPERVISED RUN (out/review-rc5c.md). The cell's
  fingertip correction is merged into dev (1224b61; 75 affected tests pass); the seams services were restarted on it. Live script
  ~/Documents/w8-out/rc5c/live/run.sh (survey | stage = --stage-all --prep-only) from the detached worktree
  ~/Documents/fmb-cell-lab-rc5c. Open, insert runs only: R40 on drive-mate (P2-4). P3: finish_home's width units slip (fails safe).
- **R40 confirmed (2026-11-21 ~13:50 MST, operator in the terminal)**: "yes keep both": top-down tool0 >= 147.8 mm, and any other
  pose (the sideways regrasp place-down at the clear area) >= -0.2 mm, both from the October motion logs (the sideways set-downs
  bottomed at -0.2 / 2.6 mm, move_down's sideways floor ~0). The live page is back at https://s.robo-chain.dev/live/fmb1-cell-0dc845/
  (user units fmb-live-page = mjpeg_view /annotated_stream + D455 + this session's transcript, fmb-live-tunnel = cloudflared; slug
  re-set on the Mac with a login shell for npx).
- **R43 (2026-11-21 ~14:45 MST, operator in the terminal)**: "okay lets start the work im not at the cell but that is the point of
  these reviews": reviewed code (rc5c) runs on the real cell without him present. The live page and stop_run.sh are the watch.
  The rc5c survey ran 14:49-14:57: all 5 parts read (yellow at the clear area (-306, -279) mm, upright, found by search view 2), the
  depth tops match CAD for the read parts (u_orange 21.6 vs 21.5 mm), homed (R42). The lights went off/on 14:39/14:53; the exposure
  keeper now returns to auto on one saturated frame (dev b8a4020). The eye-check precondition is replaced by the depth record plus
  photos: yellow stands alone where the robot set it down on 11-18.
  rc5d (w8): survey-scene in the CAD frame (it had been written in the board frame; dry-reals only); KEEP_CLEAR emptied (operator: "there
  is no calibration paper on the yellow part"; the 11-18 sheet is gone). Step (c) on the live readings: relocate_no_spot. The live
  staging depth holds untracked objects: a ~160x180 mm hollow tray (FMB3 base3), a ~30 mm strip (pegs), and a >100 mm object (likely
  the D455 table camera). Choice put to the operator: (A) clear them by hand, or (B) a new staging layout on depth-free table (lead
  recommends B; w8 mapping the free table).
  Correction (operator 11-21): the untracked objects are the furniture-bench SQUARE TABLE (top + legs), not FMB3. The >100 mm object
  is the D455 table camera (150-184 mm, x 0.40-0.45, y -0.53 to -0.45). rc5d tagged (df9205c: the survey frame fix, KEEP_CLEAR
  emptied; lab 138 passed). Free-table proposal R = x [0.22, 0.40], y [-0.57, -0.22] (all 4 FMB1 parts fit, spots >= 80 mm from the
  D455). Sent to the operator. Lead: rc5e makes untracked depth objects static obstacle boxes for the whole run, plus a 60 mm spot
  keep-away from boxes > 100 mm; a probe dry-real on R.
  Probe dry-real on R (rc5d, live readings): relocate_no_spot. R is free table but too tight for the spot rules (u_brown stays in
  R's middle; the yaws are relative to the read yaw; 20 mm grid). Even with every part lifted, only 2-3 of 4 fit. Correction sent to
  the operator. Options: A move the D455 + square table off the old region (works with rc5d as is), or B packing-aware spot
  choice (new code + review). Lead recommends A if someone can be at the cell.
  rc5e = lab tag rc5e-fmb1-obstacles (6130a65): untracked depth objects become static obstacle boxes for the whole run (every leg,
  check_path, the shortcut, the held part, the RG2 and mount), a 60 mm spot keep-away from boxes > 100 mm, fail closed if
  unreadable; lab 146 passed. The live map has 13 boxes, incl. the D455 (195 mm), the square table, and an UNKNOWN 193 mm object at
  base1's -x edge (-17, -463) mm, asked of the operator. With the boxes, R fits 3 of 4 at best, so the lead now recommends A firmly.
  Opus review of rc5e running.
  rc5e review: NOT READY (out/review-rc5e.md). P1: box tops are capped by the D415's near limit (~0.41 m, i.e. 0.17-0.21 m above the
  table at the region hovers), so the D455 and box 2 may be taller than mapped. P2s: the survey's forced flight is unchecked against
  pre-survey rules and boxes; one p99 per box hides a post; pre-survey legs vs the D455; box 6 = depth 45 mm at u_orange's corner (the
  CAD top is 20.8; a lean or a prop); clear-area coverage. Fixes as rc5f (w8): a clipped box reaches up to SAFE_Z, a per-cell envelope,
  a persisted map for pre-survey legs, a part-rest depth check refusing the pick, 95 % coverage. Lead still recommends someone at the
  cell (A + check u_orange + identify box 2).
  rc5f = lab tag rc5f-fmb1-obstacles (0010dfb; lab 159 passed): clipped boxes are 1.2 m columns after the survey, map reads at 610 mm,
  a persisted map + START_TOO_LOW_FOR_UNMAPPED_TABLE, PART_NOT_AT_REST_DEPTH, the clear area required at 95 %, survey hovers at
  560 mm (DEPTH_MIN 0.41). Probe on the live files: obstacle_map_unreadable (clear area 25 % from the old captures). Diagnostic:
  u_orange refused (45.6 mm vs CAD 20.8), and box 2's column blocks the way home. The table needs a person (D455 + square table off
  the staging region, u_orange flat, box 2 identified or removed). Opus review of rc5f running.
  rc5f review: NOT READY (out/review-rc5f.md). P1: a view seeing only an object's bottom un-clipped its column (the clip test used
  other views' returns); known-clipped boxes were still flown over during the survey and expired from the saved map after 2 h; the
  "no unattended run until box 2 is identified" rule was prose only. P2s: invalid "contradicted" drops, prior map replaced not merged,
  exit 0 when it couldn't get home, missing depth counted as free. Fixes as rc5g.
- **R44 (2026-11-21 ~18:30 MST, operator on WhatsApp)**: "i placed those square table objects there for a reason. there is plenty space
  to stage fmb1 parts i dont know how i can help you with whtever u orange problem you have": the square table and the D455 stay;
  staging uses free table anywhere suitable (not a fixed rectangle); u_orange and box 2 are ours to resolve. rc5g: free-table
  staging (IK-checked, corner-first 10 mm grid, world-axis yaws), a second aimed view before a rest refusal, a second-angle re-read
  of a one-view clipped box (likely the D415 cable).
- (2026-11-21 ~19:50 MST, operator in the terminal) The square table was flagged for its ArUco markers ("the aruco marker connection
  is what i flagged it for like how fmb has markers"): it should be a KNOWN object to the localizer (marker layout + CAD), like
  the FMB parts, not an unknown depth lump. Its CAD is prompt2assembly's (dual); the marker work is the pose6d session's on the Mac
  (8d65759c). A read-only agent is finding the marker data and checking for marker-ID collisions with FMB (a safety question).
- (lead, 2026-11-21 ~20:30 MST) R40 inside drive-mate (review P2-4) MERGED into dev (8e8726e, branch lead-r40-drivemate 54e559b):
  tool0_floor.py; R40_TOOL0_FLOOR=1 set by drive-mate for its motion children (align, wrist turns, touch probes, hover, prismatic,
  compliant force mode with a per-tick live watchdog); refused by name; unchanged when unset. Opus review FIT TO MERGE
  (out/review-r40-drivemate.md): no P1/P2; predicted seats 167-200 mm, all above 147.8. Queued P3s: an upward-escape rule (a pure
  rising lift from below), a 5 mm trip margin + pose staleness in the watchdog, refusals keyed on the child's final result,
  missing tests (compliant/prismatic trip paths). drive-mate server restarted on it.
- Square table markers (out/square-table-markers.md): AprilTag 36h11 IDs 4-23 (top 4-7, legs 8-23); layout in the Mac's
  robo-chain fuzzydroid square-table-threading assets (placement.json, 19.84 mm black square). The live localizer (5x5 + 4x4) does
  not decode them; there are no exact ID clashes, but synthetic renders showed rare 4x4 misreads as FMB3 parts (fork_yellow, line_brown).
  rc5g: only this run's assembly parts + base count as known (foreign reads logged, depth stays an obstacle). Adding the table to
  the localizer (the AprilTag dictionary, duplicate ranking, per-model marker size, 5 meshes, real-tag ID check) is queued after
  the first FMB1 insert.
- rc5g (5ecd7ee) + rc5h (26c9ea8) tagged (lab 182 passed): the rc5f review fixes, R44 free-table staging + second looks, foreign
  reads ignored. Live-data probe: UNRESOLVED_COLUMN_ON_PATH on box 2 (col_-17_-461): no feasible view sees 150-184 mm there.
  Lead's evidence that box 2 is the D415's own cable hanging into the assembly view: the D455 photo right after the survey shows
  nothing tall at base1's far edge, and depth only from ~150 mm to the near limit. rc7 (stand-up + u_brown insert) rebased on rc5h,
  dry end-to-end OK, tag pending. Opus review of rc5h running (incl. how to recognise self-observation safely).
  rc5h review: NOT READY (out/review-rc5h.md): the "could it have seen higher" test uses the colour frame, but aligned depth is empty in
  the left ~150 px band; the second look flew before its column was protected; and 6 P2s. BOX 2 SOLVED (lead, 11-21 ~21:00 MST): a
  re-run of the reviewed rc5c survey-only (homed) grabbed the wrist colour image at the assembly view
  (w8-out/rc5c/live/box2check/wrist_assembly_view.jpg). The RG2's open fingertips fill its top corners, at the depth near limit.
  LEAD DESIGN CHANGE -> rc8: a self mask of the gripper + the invalid-depth band; a declared envelope (work zones + measured keep-out
  columns for the D455 and the square table) for planning; depth only as a fail-closed zone verifier (ZONE_NOT_CLEAR); the dynamic
  obstacle geometry (clustering, columns, second looks, prior map) removed from the planner path. w8 writes DESIGN-rc8.md first.
  w8: box 1 ("the D455") is ALSO the gripper (same pixels, u 532-621 / v 4-172, at the 0.41-0.44 m near limit, in two views). With a
  self mask (RG2 + changer + mount, 15 px) and the invalid band, no tall object remains in the R43 reads: base1, the square table
  top 35 mm + legs 30-32 mm, the 3 parts, and a 63 mm object by the robot base. The D455 isn't in the mapped area. Corrected to
  the operator. rc8 envelope: below SAFE_Z only in the work zones, transits >= SAFE_Z (table + 0.36 m), so a fixture outside the zones
  below 0.30 m is safe by construction; the square table gets declared keep-outs. rc7 tagged and parked
  (rc7-fmb1-standup-insert 56f8d6f, 193 passed).
  DESIGN-rc8 accepted (w8-out/rc8/DESIGN-rc8.md, envelope-live.png). Layout fmb1_envelope_2026_11_22: zones assembly / clear area /
  FMB1 staging (x 0.19-0.40, y -0.62..-0.25); keep-outs = the square table top + legs (+30 mm); the 4-part packing fits with 5 mm
  slack. Correction (w8): SAFE_Z bounds tool0, and the fingertips hang ~0.23 m lower, so outside the zones EVERY moving point stays
  >= table + 0.32 m (transits at home height; CheckedIO ENVELOPE_LOW_OUTSIDE_ZONE); fixture rule: below table + 0.30 m outside the
  zones. Coarse 30 % body mask accepted (the verifier fails closed on coverage); zone views are chosen with the mask applied, plus a
  clear-area view.
- (2026-11-21 ~21:45 MST, operator in the terminal) "the d455 on the table is on the other extent of the table after the staging
  region": it stands at the far end, beyond the staging region and outside every work zone (low motion never goes past the staging
  zone; transits keep >= table + 0.32 m). The square table as a known object comes AFTER the first FMB1 insert ("that is the reason i
  placed it after the staging region").
- (lead, 2026-11-21 ~22:40 MST) rc8 rulings: with the gripper's real envelope (87 mm along the jaw axis, 114 mm across) and
  full-height keep-outs, the staging band was only ~80 mm. So: keep-outs at the masked map's measured height (p99) + 30 mm (the body
  may pass above the square table, never the fingertips or the held part); far edge by height (below table + 0.10 m nothing past
  x 0.40; at any height nothing past x 0.55; the D455 is beyond 0.62 per the operator); the 4-part packing DECLARED as slots in the
  layout, verified offline (SLOT_BLOCKED at run time, no run-time search).
- **R45 (2026-11-21 ~23:30 MST, operator on WhatsApp: "these seem like too many unnecessary checks or tests that is taking way too
  long")**. Lead's process rules from the week's time breakdown (out/time-breakdown-week-1118.md: 74.5 agent-h; real robot ~107 min;
  the board program 14 min of look-only runs; 54 reviews, ~31 real arm-safety P1s caught; the delay was mostly scope added before
  flying, wrong table guesses, and dead sessions):
  1. Opus review only for code that can move the arm; nothing else waits on a review.
  2. A 4-6 min read-only real survey is the first step of every new piece; never gate a run on reading a photo.
  3. On the operator's go, fly the smallest reviewed scope that day; add features after.
  4. No long twin panels on the path to the robot (the twin checks run beside it, not as gates).
  5. After two NOT READY rounds on one mechanism, check its premise on the real cell before a third.
  6. Targeted tests per change; the full suite once per flown release candidate.
  7. Watchdog + done-file waits; nothing on the robot's path depends on dual.
- (lead, 2026-11-22 ~00:05 MST) rc8 real look-only survey (ce28afb, ~6 min, home true) stopped ZONE_NOT_CLEAR. Checked on the raw
  frames (w8-out/rc8/live-survey/*_marked.png, zone_check_3views_lead.png): 7 of 9 cells sit on the same pixels (u 650-675, v 196-201)
  in all 3 views, on the self mask's lower edge between the fingers, reading 90 / 16 / 19 mm over bare table = D415 false depth beside
  the RG2 body. The other 2 are the edges of read parts (u_brown corner, line_green end, 12-15 mm, just past footprint + 10 mm). Cover
  and the table plane passed. The real table matches the R43 start (yellow at the clear area; brown, orange, green in staging). Sent
  to w8 for rc8b: a fringe ring outside the mask = UNKNOWN; known part explains returns within 20 mm up to its top + 5 mm. Then
  survey-only on rc8b, then the arrangement (supervised).
- **R46 (2026-11-22 ~00:30 MST, operator on WhatsApp: "Isn't the point of the DT view with cad enough to get a cleaner mask with
  a extra offset")**: the self mask is rendered from the CAD (RG2 meshes at the measured opening + the D415 mount, through the
  hand-eye, from the depth imager's view) plus a small offset, never from collision boxes; it must not throw away depth that is not the
  gripper. The box mask covered ~40-50 % of the image (the D415's own boxes + the RG2 body box projected from the lens). Returns the
  CAD mask doesn't cover are found out, not masked.
- (lead, 2026-11-22 ~00:45 MST) w8's CAD self mask (RG2 meshes + v3.1 mount, colour + depth-imager views): 30 % of the image at
  +30 px (the box mask 40-50 %); covers the real fingertips (left tip ~29 px low vs CAD, open: hand-eye roll or finger asymmetry).
  It does NOT cover the 11-22 notch or the R43 near-limit clusters at the flown opening (jaws open). The D415 IR frame (home, lights
  off; w8-out/rc8b/ir/) shows a dot-free patch at the top centre (u ~520-830, v 0-210): the gripper blocks the projector there, so the
  false depth is the gripper's projector shadow. Ruling: CAD + 30 px wired; the projector shadow becomes UNKNOWN too (CAD from the
  projector origin, or a measured map); confirmed on the rc8b survey with IR on. Camera op: toggling enable_infra off at runtime
  stopped the depth pipe; enable_depth false->true restored it (identical valid share/median). No run was affected.
- (lead, 2026-11-22 ~00:35 MST) Projector-shadow fix: w8's CAD-from-projector render covers 61-71 % of the R43 clusters and 3/6 notch
  pixels; a static IR map over-flags 20-43 % of the image. Ruling: rc8b = CAD + 30 px, flown survey-only to collect IR at the view
  stances (a short review of its survey path first); rc8c = per-read texture gating (a pixel with no projector dots in that read's own
  infra1 frame is UNKNOWN; a missing/stale infra1 frame fails the read; threshold from the rc8b pairs); one Opus delta review
  ce28afb..rc8c before the arrangement flies. infra1 now ON (dev d1112c0 stack_up: 1280x720 @ 15 fps; at 30 fps the depth module stalled).
- (lead, 2026-11-22 ~02:20 MST) rc8b (1acdfff) survey-only flown (review-rc8b-survey GO, no P1): rc 0, home, 0 refused; 3 views per
  zone; every zone ZONE_NOT_CLEAR (assembly 149 / clear 139 / staging 113 unexplained, phantoms to 231 mm) - expected without the
  texture gate. IR per read saved (w8-out/rc8b/live-survey/): the phantoms sit in the top-centre dot-free band in every view; the
  Vention carriage (y -0.15..-0.04) is outside every grown zone. Data handed to w8 for rc8c's gate; dual on hold for plan47 pb12+Kit.
- (lead, 2026-11-21) nest branch, seed 9 (twin DH): 3/4 seated (line_green through the nest rule); yellow stopped on RG2 tip vs
  seated line_green (true 3.2-6.2 mm < generic 10 mm). Ruling: use the cell's proven grasp rule (ablations/grasp_depth.py, 386e31e,
  the operator's pad-centre rule) for every part's grasp depth. Below h_n the fingertip blocks against a certified S need >= 4 mm on
  exact CAD (SEAT_CLEARANCE_M, pinned to its source); the rest of the RG2 and the camera keep 10 mm. Then re-run + nest re-review.
- **plan47 rulings** (DroidStack coordinator, 2026-11-18 ~20:3xZ, ref bbef00; facts from DroidStack platform dev f1748570):
  - P47-1: COPY `robots/ur5e_rg2` (composite + binding + NOTICE-OnRobot.md) from prompt-to-assembly into our lab; don't re-onboard.
    The OnRobot meshes are research-use only and ruled out of any public release, so the lab stays PRIVATE.
  - P47-2: what runs today is the manipulation pack's `pose` task kind: prove the camera stream at the new mount, a pick of each
    FMB part and a placement at its pre-insertion pose. Insertion and `assembled` judging wait for the assembly pack's phase 2.
  - P47-3: FMB work that needs the assembly pack goes through plan47 as DroidStack work (folded into phase-2 acceptance), never a
    private copy.
  - P47-4: a sim pass is not hardware qualification; only a qualified real binding (conformance checker on the real robot) plus a
    sim-proven delivery goes to hardware, and that decision is the operator's. Ask plan47 before changing any robot record shared
    with the paper workspace.
  - P47-5 (2026-11-18): card use on dual = RoboSort's interim rule: right before use check `nvidia-smi --query-compute-apps`,
    `docker ps` and the card holds; take a hold for the run; never touch another session's process; message plan47 only on
    trouble; switch to the DroidStack claim tool when plan47 says it is live. The lab staying local with no remote is confirmed.
  - P47-6 (2026-11-18): the FMB1 phase-2 needs are recorded as the assembly pack's acceptance input. Split: the force-limited
    search (<= 5 N, lateral <= 6 N, rotation locked) is OUR planner's behaviour; the pack provides the `assembled` kind, its judge
    and the force limits as declared bounds the planner reads. plan47 says when phase 2 has a shape to review.
- **plan29/co-coordinator facts** (2026-11-18, ref 656953):
  - P29-1 SAFETY: DDS isolation is a safety control. Every Isaac or ROS 2 run of ours on dual goes through DroidStack dispatch
    with its DDS island, NEVER a bare `ros2 launch` on a shared network, so no simulated /joint_states or trajectory can reach the
    real robot's domain beside the cell PC. A step that needs the cell PC and dual on one DDS domain = message plan29 first.
  - P29-2: dual's / is tight (~106 GiB free vs the 90 GiB floor; ledger /var/tmp/.pa-disk-headroom.lock.d). Bulk data stays on
    the cell PC or a USB drive, never dual's /. a4500 stays off for us.
- Parked: pose6d integration (R2); FMB3 runs; the tipping-start (phase 2) cycles; the M4 prompt2assembly planner
  (`compliant_insertion_studio/m4_dispatch/`); the M3b RL driver.

### Standing rulings carried from earlier sessions (memory notes, 2026-10-19 .. 11-16)
- Commit and push to `dev` freely with `ablations/push_dev.sh`; never `main`; no `data/`, `_references/`, `logs/` (10-23).
- Change primitives, tolerances and defaults on the evidence without asking; still ask before protective-stop unlocks and
  physical table work (10-23).
- FMB1 first; FMB3 only as final validation (10-23).
- When the trusted measurement is missing, stop or re-measure; never fall back to a weaker one that can drive into a part (10-23).
- Stop a run as soon as its results call for a fix (10-20). Fix forward on `dev`; go back to a `ckpt/` tag only when asked (10-23).
- No per-part settings: the insertion derives stages, clearances and DOFs from CAD (10-19).
- The repo is self-contained: no reads from `~/Documents/aruco-grasp-annotator` or `prismatic-manipulation` (11-16).

## Standing approvals (see ~/.claude/skills/autopilot/references/authority.md)
APPROVED 2026-11-18 (R11).

| Without asking | Ask first |
|---|---|
| CPU/offline work, code, tests | Deleting evidence or anything not provably re-creatable |
| GPU runs from a committed script after the card-free check | System changes on a shared box (apt, sudo, /etc, systemd) |
| Removing our own re-creatable bytes, with a receipt (path, bytes, re-creates-from) | Scope/acceptance changes; instruction files that bind others (incl. this repo's CLAUDE.md) |
| Downloading public models/weights/images to the data root, within disk floors | Paid compute or API spend |
| Commits and pushes to this repo's `dev` branch | Anything public: repos, posts, PRs, registering/submitting |
| Coordination messages to plan29, robosort and other leads; to plan47 only for card/box trouble until R10 is lifted | Changes to other projects' repos or shared envs |
| Robot motion on the real cell that the replay/primitives already do, with the operator's run go | Robot motion from new code not yet proven in sim (R3); protective-stop unlocks; table work |

## Decisions (the lead's, inside the rulings)
- D1 (2026-11-18): the DroidStack lab is a separate PRIVATE workspace on the cell PC, `~/Documents/fmb-cell-lab` (local git, no
  remote until the operator OKs a private one), because this repo is public on GitHub and the lab holds OnRobot meshes (P47-1).
  It is made with `droidstack init` and dispatches its Isaac steps to dual-a4500 (plan47: prepared, 2 cards; a4500 is shared with
  a colleague whose work wins; rtx6000 not prepared).
- D2: the planner delivery is the decision code from this repo (pick/place sequencing, view poses, insertion strategy, the camera's
  hand-eye use), reaching the robot only through DroidStack's robot contract; drivers (UR, RG2/quick changer, D415,
  aruco_camera_localizer) are adapters behind it (plan47 §1). DroidStack itself is a pinned, read-only dependency (robosort's pattern).

## Operator-only actions outstanding
- Plug the D415 back in on a USB 3 (5000 Mbps) port (not enumerated on 2026-11-16).

## Facts established (don't re-derive)
- The cell: UR5e `192.168.1.111` (PolyScope 5.17.3), RG2 on a quick changer, wrist D415, cell PC i5-8500T with no GPU
  (`ONBOARDING.md` §1).
- New camera mount v3.1 since 2026-11-10; CAD tool0 -> colour optical (0, 0.0889, 0.0839) m, quat xyzw (0.258819, 0, 0, 0.965926);
  calibrated 2026-11-18 (two-height ChArUco hand-eye): the localizer's `robot_config.yaml` holds tool0 -> colour optical
  (-0.000801, 0.084790, 0.092719) m, quat xyzw (0.25427, -0.00818, -0.00632, 0.96708), plus the D415 depth correction; the camera
  is on tool +Y.
- SAFETY (w5, 2026-11-19): the October FMB1 u_orange/yellow two-pass stances were camera-up on the OLD (-Y) mount but are
  camera-DOWN on v3.1; replayed with the v3.1 geometry, u_orange's place-down puts the mount 33-34 mm into the table. No real
  two-pass run on v3.1 without the camera hard rule (camera side not below horizontal within 150 mm of the table, mount/D415
  >= 30 mm above it and >= 10 mm from the arm, fail closed): MERGED into dev 0d42b1a (5 review rounds, final READY TO MERGE).
- SAFETY (w5, 2026-11-19): on dev, prismatic_peg_insertion (line_green / inverted_u_yellow / FMB3 bars) switches passthrough + force
  mode ON and sleeps 2.5 s BEFORE its try: a stop signal in that window skips every restore (force mode stays on, the trajectory
  controller stays off), and a failed activation leaves the scaled controller off. Fixed on cell branch w4-mujoco 654044d (with a
  deferred-signal safe-state cleanup): MERGED into dev 0d42b1a. Before the first real run: a twin live check of prismatic killed
  mid-force-phase, rotate_object's 60 s budget, and the round-1 deferred items (review-w4-mujoco-cell-r5.md P2s).
- SAFETY (w5, 2026-11-19, after the merge): dev's physical arm capsules (from the twin's Menagerie collision geometry) do NOT enclose
  the real UR5e surface (UR visual meshes): up to 27 mm outside on the upper arm, 10-14 mm on others. The physical self/mount/held-part
  checks can under-estimate clearance by that much; the review's 94/94 sweep used the same capsules as truth (circular). Fix: capsules
  proven to enclose the visual meshes (triangle-level test), branch w5-enclose -> review -> merge, with the flown-history re-check.
  No real run before it merges. MERGED 2026-11-20 into dev 720a5d8 (Opus r2 READY TO MERGE; same 5/4730 flown refusals as dev).
- SAFETY (lead, 2026-11-20): real-path gaps from the w4 round-1 deferrals MERGED into dev b66d665 (w5-realpath-gaps, Opus r1+r2 READY):
  empty-gripper joint-travel and wrist_3 caps; the camera rule and the table/self check on both fix-orientation swings; rotate_object's
  replay timeout 90 s, with ROTATE_OUTLASTS_CALLER refusing before any send. Still owed before the first real run: the prismatic stop
  test (URSim 5.17.3 + ur_robot_driver 2.14.0 on a4500), and re-timing rotate_object on dev 720a5d8 (tier-2 adds ~9 s per process).
- (lead, 2026-11-20 later) URSim stop test DONE (lab bf4c166): pre-try and mid-press kills restore and hold still; a failed restore left
  force mode stopped but the controllers switched, and the replay carried on. Fix w5b-insert-restore: Opus r1 NOT READY (the replay's latch),
  being fixed. NOTE: on this tree FMB1's line_green and inverted_u_yellow go to touch_insert, not prismatic (PRISMATIC_ROUTE_OBJECTS is
  FMB3's parts); u_brown/u_orange use compliant_insert. The replay-level controller check covers every insert route. rotate_object budgets
  for every caller merged (dev 65c29ec).
- DroidStack engine `abbd0a57` (0.1.0) is installed on the cell PC at `~/.droidstack`, with boxes dual / a4500 / rtx6000 recorded.
  Its assembly and perception packs are "NOT READY — placeholder" (source snapshot `~/droidstack-src-2648fe5`, 2026-11-13).
- This repo (`inbarajaldrin/ros-mcp-server`) is PUBLIC on GitHub (unauthenticated API 200, 2026-11-18); prompt-to-assembly is private.
- Camera in DroidStack: a camera is a core observation stream publishing ROS 2 image topics (DroidStack docs/design/camera-streams.md);
  declare the D415 rigidly on tool0 at the CAD transform with its intrinsics; the perception pack is not needed for that (plan47 §2).
- The real road needs the cell to expose exactly the contract: FollowJointTrajectory, /joint_states (use_sim_time off), TF incl.
  tool0 -> camera, the gripper width/state action, the camera image topics (plan47 §5).
- robosort (2026-11-18) read for the pattern: dual:~/Documents/RoboSort (dev) docs/PLAN.md, ROADMAP.md, docs/rescue-findings.md,
  planner/README.md; in-Kit ros2_control proven in plan-anything setup/scripts/isaac61/incell_cm_carrier.py @78373beb2.
- `ssh dual` (alias added 2026-11-18) reaches dual-a4500 as aaugus11; dual can ssh back to this PC as aldrin.
