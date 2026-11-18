# ROADMAP

Ticked only with evidence (a receipt path, run dir or commit). The next unblocked item first. Rulings in `docs/PLAN.md`.

## P0 Setup
- [ ] 0.1 Home and steering files: this repo, branch `dev`; `docs/PLAN.md`, `ROADMAP.md`, `docs/findings.md`
- [ ] 0.2 Standing approvals recorded (operator's yes on the PLAN.md table)
- [ ] 0.3 Neighbours told: plan47 (DroidStack), plan29 (box sequencing), robosort (asked how it rescues a workspace)
- [ ] 0.4 Heartbeat scheduled; the "Dimelab" WhatsApp group carries project posts
- [ ] 0.5 First worker running
- [x] 0.6 pose6d notes written (R2): `docs/POSE6D_NOTES.md`

## P1 The cell in DroidStack (R1, R3)
- [x] 1.1 Lab decided: private `~/Documents/fmb-cell-lab` on the cell PC, Isaac on dual-a4500, delivery vs adapters (PLAN.md D1, D2)
- [ ] 1.0 `droidstack init` the lab; machine paths declared; `droidstack doctor` / `status` clean for what is declared
- [ ] 1.2 Robot: `robots/ur5e_rg2` copied from prompt-to-assembly (P47-1) with its NOTICE; `droidstack cell check` passes
- [ ] 1.3 Wrist D415 declared at the v3.1 mount (CAD transform) so the sim renders what the real camera sees
- [ ] 1.4 Environment: table plane 0.080 m below the robot base, base1 placement and the cell's staging / clear regions
  (`realcell_env/MEASURED_CELL.yaml`, `primitives/shared/cell_layout.yaml`)
- [ ] 1.5 FMB1 parts as assets; `pose` tasks: pick each part, place it at its pre-insertion pose (P47-2)
- [ ] 1.6 Tell plan47 the FMB1 insertion task the cell needs first, for the assembly pack's phase 2 (P47-3)

## P2 Prove the new camera mount in sim (R3)
- [ ] 2.1 Rendered D415 views at the cell's view poses (home, part view, base view, clear area) match what the code expects;
  view poses re-derived for the 30 degree tilt
- [ ] 2.2 The localizer + depth CAD fits on rendered frames find the parts within the cell's bars (pick seed, base read)
- [ ] 2.3 A pick-and-place of one FMB1 part from the new code passes in sim

## P3 Back on the real cell
- [ ] 3.1 D415 plugged in (operator), stack up, raw frames at home and the base view
- [ ] 3.2 Hand-eye calibration for v3.1 from the base1 survey; written to `robot_config.yaml` with the old values kept
- [ ] 3.3 View poses checked on real frames; one FMB1 part, then a full FMB1 run
