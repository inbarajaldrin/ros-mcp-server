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
