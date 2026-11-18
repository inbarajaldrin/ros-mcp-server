# Findings

Numbered root causes and measured facts. Never renumber. Each gives the date, the evidence path and the consequence.
Earlier findings live in the operator's memory notes and `compliant_insertion_studio/docs/`; these start with the rescue.

- **F1** (2026-10-25) The old wrist camera mount let the D415 rotate on its bracket: after a wrist rotation caught the cable
  and faulted the arm, the camera looked at the ceiling with the arm top-down (frames in the lead's job tmp, 08:32-08:36)
  -> a new printed mount (v3.1, 11-10); the localizer calibration and the view poses are the old mount's and must be redone.
- **F2** (2026-11-16) The repo read CAD data and the dashboard client from `~/Documents` folders that only existed on the cell
  PC, and picked base1's assembly file by filesystem order -> copied into the repo (`assets/cad_data`, `scripts/ur_dashboard.py`),
  one `FMB_CAD_DATA` / `FMB_ROS_WS` rule, base1/base3 files pinned (commits 89f4b26, 6e5dd86).
- **F3** (2026-11-16) No ROS stack running and no D415 on USB on the cell PC; the camera is on the new mount but unplugged
  (`lsusb`) -> operator action in `docs/PLAN.md`.
- **F4** (2026-11-18) DroidStack's assembly and perception packs are placeholders ("NOT READY"); its proven road is simulation
  (M1, Franka FR3 in Isaac Sim 6.1) and real hardware is not qualified (`~/droidstack-src-2648fe5/docs/`) -> the FMB work and
  the camera need DroidStack work through plan47 (autopilot rule 8), asked 2026-11-18.
