# ROADMAP

Ticked only with evidence (a receipt path, run dir or commit). The next unblocked item first. Rulings in `docs/PLAN.md`.

## P0 Setup
- [x] 0.1 Home and steering files: this repo, branch `dev`; `docs/PLAN.md`, `ROADMAP.md`, `docs/findings.md` (e66c88f); lead folder `~/Documents/fmb-cell-lead/`
- [x] 0.2 Standing approvals recorded (R11, 2026-11-18)
- [x] 0.3 Neighbours told: plan47 (answered, P47-1..6), plan29/co-coordinator (told 2026-11-18), robosort (answered; pattern in PLAN.md facts)
- [x] 0.4 Heartbeat: session cron a3dd7e9e, hourly at :23 (expires after 7 days; re-create on resume); "Dimelab" WhatsApp group carries project posts
- [x] 0.5 First worker running: w1-lab-cell, Codex thread 01a1f073-6860-7881-8b70-3939ccd5ee9b on the cell PC (2026-11-18)
- [ ] 0.7 a4500 and rtx6000 prepared as DroidStack boxes (R21): w2-boxes, Codex thread 01a1f103-ad0d-7870-8754-0194075d4040 (2026-11-18)
  (a4500 system-mode prepared 11-18 23:44Z, tx 00b2141e; Isaac/ROS dispatch waits on DroidStack's system-mode probes, plan47's lane;
  rtx6000 on hold: 34 GiB free, no Isaac, borrower's work on the card)
- [x] 0.6 pose6d notes written (R2): `docs/POSE6D_NOTES.md`

## P1 The cell in DroidStack (R1, R3)
- [x] 1.1 Lab decided: private `~/Documents/fmb-cell-lab` on the cell PC, Isaac on dual-a4500, delivery vs adapters (PLAN.md D1, D2)
- [x] 1.0 `droidstack init` the lab (lab commit 74888ca, `~/Documents/fmb-cell-lab`); machine paths: owed with the dual dispatch setup
- [ ] 1.2 Robot: `robots/ur5e_rg2` copied from prompt-to-assembly (P47-1) with its NOTICE; `droidstack cell check` passes
  (copied + local `cell check` valid for all four tasks under the certified engine 989cff8d9172, lab ddda149; LIVE check owed:
  dual's admission is closed by a DroidStack supervisor bug (L7) until plan47's fix + re-prepare; a4500 re-prepare waits on its card)
  11-20: dual admits again (engine 5a027edcfaac); the live cell run pose-u-brown is WORKLOAD_FAILED (exit 2 after 2.1 s, request
  54a53d01456dfaa203b49cf98e691182), unreadable until plan47's step-evidence install; not retried.
  11-21 on engine 3fd8948f16bf (step-evidence installed): pose-u-brown WORKLOAD_FAILED is now readable: ISAAC_ROS_RMW_UNDECLARED
  (dual's ROS road exports no RMW_IMPLEMENTATION; the box's ROS probe certifies it as fastrtps). Box-side; sent to plan47.
  Request 702aa533c10fad74b136ec4e676acc1e; lab w7-dual-live 0808020, lab-notes/receipts/2026-11-21-w7-dual-live/.
- [x] 1.3 The v3.1 mount + D415 body added to the robot copy's collision and visual geometry (R14), and the D415 declared as a
  camera stream on tool0 at the CAD transform, so the sim renders what the real camera sees and planning keeps the mount off the table
  DONE 11-20 (lab branch w7-dual-live 7599048, lab-notes/receipts/2026-11-20b-w7-dual-live/README.md): `collision model` on dual,
  engine 5a027edcfaac, request 4c49e11cdd3c36d994e9f50ce8370b99 completed; model.json has 20 links incl. rg2_base_link (11-19: 19,
  none), whose 3 meshes match the mount USD (lower_saddle, upper_tower, d415_body); collision_probe.py sees rg2_base_link vs the table
  (home clear; first seed-47 hit 19.98 mm). Run from a slim sparse checkout of the same lab commit (all 264 files sha256-identical):
  dual's disk bound counts staged inputs as writes (plan47 fixing in ds-disk-bound-staged).
  11-21: re-confirmed from the FULL lab tree on engine 3fd8948f16bf (request 8ac8cf354690b1a7d5fa09e82433b4e3, no disk-bound
  refusal; the output is byte-identical to the 1.3 model, published_sha256 b16212c3 reproduced locally); lab w7-dual-live 0808020.
- [ ] 1.4 Environment: robot base 0.080 m above the table top (R15), base1 placement and the cell's staging / clear regions
  (`realcell_env/MEASURED_CELL.yaml`, `primitives/shared/cell_layout.yaml`) (declared, base at 0.080 m checked by the lead; table size is
  the paper's, a typed gap)
- [ ] 1.5 FMB1 parts as assets; `pose` tasks: pick each part, place it at its pre-insertion pose (P47-2)
- [x] 1.6 Told plan47 the FMB1 needs for the assembly pack's phase 2; recorded as acceptance input (PLAN.md P47-6)

## P2 Prove the new camera mount in sim (R3)
- [ ] 2.1 Rendered D415 views at the cell's view poses (home, part view, base view, clear area) match what the code expects;
  (home: calibrated sim vs real 0.42 px, lab b0e161e; camera-aimed base / part / depth / staging views all in frame, lab 594fbb5;
  open: the clear area and the insert-path views - audit running)
  view poses re-derived for the 30 degree tilt
- [x] 2.2 The localizer + depth CAD fits on rendered frames find the parts within the cell's bars (pick seed, base read)
  DONE 11-19, lab 58a30ab (lab-notes/2026-11-19-isaac-perception/attempt-7-verdict.md + consumed-perception-table.md): what the cell
  consumes - published snapped pose, 5-frame still median, depth pick / base ICP - vs PhysX frame-time GT, all 7 targets pass the
  code's bars: <= 0.32 mm XY, depth-pick z +0.17..+0.20 mm, <= 0.28 deg; base ICP 0.41 mm. The 72 deg yellow flip is a raw-PnP branch
  the localizer's IPPE table alignment removes before publishing. Real anchor: the 11-18 upright u_orange pick. Raw PnP reads parts
  7-9 mm LOW in z (removed by the table-rest snap). LIMIT: hand-eye error is zero by construction in the twin and markers are
  noise-free, so this proves the pipeline's logic, not the real cell's accuracy.
- [x] 2.3 A pick-and-place of one FMB1 part from the new code passes in sim
  DONE 11-20 in MuJoCo (r49 attempt 10, below): every part picked, carried and placed; u_orange and yellow also set down and
  regripped. Every grip was judged against its expected width (6/6 in the RG2 band) and welded only on the attach verdict.
- [ ] 2.4 R22: FMB1 assembled end to end in the twin by the cell's code (regular insert), rendered for the operator; then with
  FOV-gated poses (paper2). w3-sim-assembly, Codex 01a1f10e-baa8-7ea3-8a39-07fec69100a7, on a4500 (2026-11-19)
  Isaac road PARKED 11-19 06:30Z: the wrist-reaction stop gate refused (stop overshoot 157/110/67 N at 60/240/480 Hz vs the
  30 N bar; the mass-load delta also fell with rate, 0.74 -> 0.49 -> 0.28 N). Rigid contact on a position-controlled arm needs
  compliant-contact tuning; the assembly video goes through the MuJoCo road (R23-R25). Nothing deleted; cell branch w3-sim kept.
  w3 moved to 2.2 (perception on Isaac-rendered frames).
  FIRST HALF MET 11-20 in MuJoCo (R23-R25, R27-R33): r49 attempt 10, the counting run. FMB1 was assembled end to end from the
  flat start (staging seed 48) by the cell's primitives on seqplan plans (board-v1), with VALID pacing: 0.80% missed ticks,
  0.9986x real time. Seats: u_brown 0.077 mm / 0.005 deg, u_orange 0.075 / 0.004, line_green 0.103 / 0.012, inverted_u_yellow
  0.147 / 0.014; 81 steps, 0 failures. Receipt: a4500 ~/pa-scratch/w4-mujoco-live/runstate/fmb-full-r49-a10 (archive
  r49-seqplan-a10). Video: ~/w5-mujoco-videos/fmb1-r49-attempt10-seqplan-physics-PASS.mp4 (sha256 1d5ed42b), sent to the operator.
  Config: cell w5-seqplan 9fa2978; lab model 7ae1e6a7 with island sleeping plus wake-after-detach (A/B vs sleep-off: same
  verdicts, seats within 0.01 mm; lab 5de5cfe); hand weld on the attach verdict; stamp-paired capture; 6 CPUs.
  LIMIT: perception is ground truth in the twin (no FOV gating yet); the second half (FOV-gated poses, paper2) is open.
  11-21: attempt 10 re-run on the REAL DroidStack lane (plan47's landed weld with partial-open release, engine 3fd8948f16bf, lane
  w5b-weld-seam-r2 c2aa45a9, no stand-in): 4/4 (0.079 / 0.076 / 0.105 / 0.146 mm), pacing valid (1.46 % missed), all 6 held
  releases attested at partial widths; lab e454da7 mujoco/evidence/w5b-lane-a10/r2-a10/.
  The board program's own twin runs (rc4c / rc5, FOV-gated reads with noise) are in w8-out/ (rc4c relocate + u_brown, fewer-moves).

## P3 Back on the real cell (done = R13: 3 hands-off certified FMB1 cycles in a row, untipped reset, v3.1 calibrated)
- [x] 3.1 D415 plugged in (operator), stack up, raw frames at home and the base view (2026-11-18: lab
  lab-notes/2026-11-18-real-home/real-d415-home.png; survey frames ~/Documents/calib_bags/v31_20261118_141729, _151830_spot2b)
- [ ] 3.2 Hand-eye calibration for v3.1 from the base1 survey; written to `robot_config.yaml` with the old values kept
  (solved 11-18: two ChArUco spots + base1, 123 frames, 0.95 px; held-out 0.73 px vs 9.0 px CAD; t (-0.8, 84.8, 92.7) mm;
  lab lab-notes/2026-11-18-real-home/handeye-v31-final_nof.json. Writing it waits on the operator)
- [ ] 3.3 View poses checked on real frames; one FMB1 part, then a full FMB1 run
