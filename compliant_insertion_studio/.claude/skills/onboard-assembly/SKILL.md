---
name: onboard-assembly
description: >-
  Bring a new assembly (FMB2, a new base, new parts) onto the UR5e + RG2 cell and assemble it: readiness
  check from CAD, cell layout, per-part runs with recording, how to verify a seat, and the fixes that FMB3
  needed (2026-10-20). Use when the operator says a new assembly / base / set of parts is on the table, asks
  to "assemble FMB2/FMB3", or when a part of a non-FMB1 assembly fails to pick, insert or seat.
---

# Onboarding a new assembly on the cell

FMB3 (base3 + line_brown, line_red, fork_orange, fork_yellow) went from "on the table" to 4/4 seated on
2026-10-20. Everything below was needed for it; FMB1 still runs through the same code.

## 1. Offline readiness (no robot)

```bash
python3 ablations/onboard_check.py fmbN          # READY / NOT READY with the fix for each line
python3 ablations/grasp_depth.py --assembly fmbN --write   # if the grasp-depth entries are missing
```

Inputs it checks, all CAD/data: meshes, grasp points + the assembly grasp, closed grip width (annotator jaw
width - 14 mm clearance; matched FMB1's measured widths within 0.7 mm), symmetry file, seat pose in the base,
grasp depth (fingertips clear of the base and seated parts), the parts seated under each part, the recorded
tool sequence. No per-part numbers go into code: if a table is missing, derive it from CAD.

Routing is still a name list in `primitives/translate_object.py`: bars and parts that straddle seated parts
(lines, forks, inverted U) go to the prismatic insert; U-type parts use the compliant FSM. To be replaced by
a CAD rule.

## 2. The table

- The active layout in `primitives/shared/cell_layout.yaml` must match the table: assembly region (base
  read flies over it), clear area (regrasp spot, must be empty), staging region (where THIS assembly's parts
  wait), parked region (another assembly; never scanned). Add a layout, don't edit a used one.
- Look before moving anything: grab `/annotated_stream` from home, the clear area and the staging view;
  crop to the base by projecting its CAD square (the operator: "crop the image to the base").
- Two assemblies on one table: check for cross-dictionary marker collisions (base1's 5x5 id 1 decoded as a
  4x4 id 5 = phantom fork_orange). The localizer drops a detection when another dictionary's covers the
  same quad; verify nothing phantom appears.
- The operator watches `http://<cell tailscale IP>:8080/` (annotated stream with the +-x/+-y compass,
  `/depth_view` height map, raw camera). Not Foxglove.

## 3. Run one part at a time, recorded

```bash
NO_AIM_BIAS=1 BASE_FROM=depth python3 -u ablations/replay_real_assembly.py \
  --assembly-json ablations/ground_truth_resources/Assembly_fmb_assembly_N_results.json --only PART
```

Start a DIAG bag around each part (`DIAG=1 compliant_insertion_studio/scripts/record_demo_bag.sh DIR`,
stop with SIGINT in a trap). Check the exit code; stop and iterate on the first problem.

## 4. Base pose: what to trust

- **Read the base with an empty gripper, or from a view the held part does not block.** The held part is
  rigid with the camera; from the insert hover it hid markers and the read jumped 4-8 mm while nothing had
  moved the base (line_red stalled on the rim six times at that aim). The pre-insert read now picks a
  top-down view shifted so the base lands in the free part of the image (`held_clear_read.py`), reads,
  and returns to the exact start pose so the held-orientation chain stays valid.
- **Depth outline vs markers per base.** On base3 both the seated line_brown and the empty line_red
  channel sat +4 mm in x of the marker frame and on the depth outline: use `BASE_FROM=depth` there. Check a
  new base the same way (slot walls / a seated part in depth vs CAD) before trusting either.
- Next step (operator): register the CAD model to the depth point cloud for the base and seated parts —
  every surface, not a rectangle fit.
- The base is not fixed. It moves under inserts and pushes (20 mm over the line_red recovery). Re-read
  between parts.

## 5. Is it seated?

- Never the FSM label. From the insert CSV: final `tcp_z` vs `predicted_seat_z` (forks: 0.6 mm, still).
- Depth, but **close and tilted**: `seat_check.py --object P --base B` views from 0.47 m tilted 25 deg from
  the part's outer side, heights against the base-top plane. Straight down, the overhead lights reflect off
  glossy tops (a seated line_red read 3 mm high at one end). Depth bar-top heights carry ~1 mm noise.
- The operator can see the cell: when they say it is seated, the measurement is what to fix.

## 6. Fixing a bar that did not seat

- Shifted along its length (axial `shift_mm` >= 3): axial nudge from the high end first.
- One end high with that end's lip-side edge high (per-end edge heights): sideways nudge away from that
  edge at the high end (`nudge_part.py --toward +x --at-long -58`), pad at the part's top, tips above the
  base top.
- **Taps, not pushes**, to the CAD face, never past it: the base slides (1 mm past the face turned base3
  1.7 deg; a slow axial push slid it 4 mm). Operator: "a little tap would have prevented the drag".

## 7. Recovery

- Part still held after a failed insert: `disassemble.py --assembly-json J --place-held PART` sets it down
  at the clear area (its own base/grasp from the assembly files); rerun with `--part-at-staging`.
- Stopping: `compliant_insertion_studio/scripts/stop_run.sh` (dashboard stop first).

## 8. FMB1-only assumptions found so far (grep before trusting a new assembly)

`fmb1_assembly.json`, `'base1'`, `DEFAULT_BASE_POSITION`, `line_green` in ablations/ and primitives/.
Fixed on 2026-10-20: seat poses, spacing check, grip widths, grasp depth, seated-under list (base3 read
FMB1's order), finger-room check, place-held, seat check. Still FMB1-only: `record_follow_reads`
(line_green), the insert-yaw experiment hook.
