# ROADMAP

Ticked only with evidence (a receipt path, run dir or commit). The next unblocked item first. Rulings in `docs/PLAN.md`.

## P0 Setup
- [x] 0.1 Home and steering files: this repo, branch `dev`; `docs/PLAN.md`, `ROADMAP.md`, `docs/findings.md` (e66c88f); lead folder `~/Documents/fmb-cell-lead/`
- [x] 0.2 Standing approvals recorded (R11, 2026-11-18)
- [x] 0.3 Neighbours told: plan47 (answered, P47-1..6), plan29/co-coordinator (told 2026-11-18), robosort (answered; pattern in PLAN.md facts)
- [x] 0.4 Heartbeat: session cron a3dd7e9e, hourly at :23 (expires after 7 days; re-create on resume); "Dimelab" WhatsApp group carries project posts
- [x] 0.5 First worker running: w1-lab-cell, Codex thread 01a1f073-6860-7881-8b70-3939ccd5ee9b on the cell PC (2026-11-18)
- [x] 0.6 pose6d notes written (R2): `docs/POSE6D_NOTES.md`

## P1 The cell in DroidStack (R1, R3)
- [x] 1.1 Lab decided: private `~/Documents/fmb-cell-lab` on the cell PC, Isaac on dual-a4500, delivery vs adapters (PLAN.md D1, D2)
- [x] 1.0 `droidstack init` the lab (lab commit 74888ca, `~/Documents/fmb-cell-lab`); machine paths: owed with the dual dispatch setup
- [ ] 1.2 Robot: `robots/ur5e_rg2` copied from prompt-to-assembly (P47-1) with its NOTICE; `droidstack cell check` passes
  (copied + local `cell check` valid for all four tasks, lab b4ef879, re-run by the lead 11-18; LIVE check on dual owed: L3/L4)
- [ ] 1.3 The v3.1 mount + D415 body added to the robot copy's collision and visual geometry (R14), and the D415 declared as a
  camera stream on tool0 at the CAD transform, so the sim renders what the real camera sees and planning keeps the mount off the table
  (sim USD: saddles + D415 box + camera at the CAD transform, done; the planner's collision model is still the paper's, regeneration owed)
- [ ] 1.4 Environment: robot base 0.080 m above the table top (R15), base1 placement and the cell's staging / clear regions
  (`realcell_env/MEASURED_CELL.yaml`, `primitives/shared/cell_layout.yaml`) (declared, base at 0.080 m checked by the lead; table size is
  the paper's, a typed gap)
- [ ] 1.5 FMB1 parts as assets; `pose` tasks: pick each part, place it at its pre-insertion pose (P47-2)
- [x] 1.6 Told plan47 the FMB1 needs for the assembly pack's phase 2; recorded as acceptance input (PLAN.md P47-6)

## P2 Prove the new camera mount in sim (R3)
- [ ] 2.1 Rendered D415 views at the cell's view poses (home, part view, base view, clear area) match what the code expects;
  (home rendered in Isaac 6.1 on a4500, fingertips within 3.4 px of the CAD camera, lab 66011a2; real-joint + calibrated-camera comparison running)
  view poses re-derived for the 30 degree tilt
- [ ] 2.2 The localizer + depth CAD fits on rendered frames find the parts within the cell's bars (pick seed, base read)
- [ ] 2.3 A pick-and-place of one FMB1 part from the new code passes in sim

## P3 Back on the real cell (done = R13: 3 hands-off certified FMB1 cycles in a row, untipped reset, v3.1 calibrated)
- [x] 3.1 D415 plugged in (operator), stack up, raw frames at home and the base view (2026-11-18: lab
  lab-notes/2026-11-18-real-home/real-d415-home.png; survey frames ~/Documents/calib_bags/v31_20261118_141729, _151830_spot2b)
- [ ] 3.2 Hand-eye calibration for v3.1 from the base1 survey; written to `robot_config.yaml` with the old values kept
  (solved 11-18: two ChArUco spots + base1, 123 frames, 0.95 px; held-out 0.73 px vs 9.0 px CAD; t (-0.8, 84.8, 92.7) mm;
  lab lab-notes/2026-11-18-real-home/handeye-v31-final_nof.json. Writing it waits on the operator)
- [ ] 3.3 View poses checked on real frames; one FMB1 part, then a full FMB1 run
