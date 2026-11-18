# Lessons for DroidStack from the real cell

Held here, NOT sent to plan47, until the operator's final decision (PLAN.md R10). Each entry: date, what the real cell showed,
the evidence, and what DroidStack would need.

- **L1** (2026-11-18) A real robot cell needs its perception node to be an adapter behind the robot contract, and the camera's
  mount transform changes with hardware (the D415 moved to a new printed mount, 11-10): a lab needs a way to re-declare a camera
  mount and re-qualify it without re-onboarding the robot. Evidence: `docs/findings.md` F1, robo-chain TF-FOR-AGENT.md.
- **L2** (2026-11-18) The cell PC beside a real robot must never share a DDS domain with simulated runs (plan29 P29-1): the dispatch
  DDS island is a safety control, not a convenience. A real-cell DroidStack setup should state and enforce this.
- **L3** (2026-11-18) A lab whose command host has no GPU (the cell PC) cannot get its first live `cell check` on a shared box:
  the card-scoped `isaac-python` road refused `BOX_NO_RESOURCE_CLASS` (the verb declares 8192 MiB, the largest cardless system class
  on dual-a4500 is 7998 MiB) and the card-scoped `cell run` reserved then refused `ROAD_PROBE_FAILED` (the system-mode Kit probe is
  not built). Evidence: lab `lab-notes/receipts/2026-11-18-local-and-dispatch.json` (requests 0bc98f62…, b5bf0418…). Needed: a
  card-scoped Isaac road that admits on a box whose other card is busy.
- **L4** (2026-11-18) The legacy dispatcher holds the whole box, so any foreign GPU work on either card refuses it
  (`FOREIGN_GPU_WORK_PRESENT`, robosort's Isaac on GPU1 while GPU0 was free). Correct for safety, but on a shared 2-card box it
  means a lab can wait indefinitely. Evidence: same receipt, payload a83bcd5e…. Needed: per-card admission (what L3 asks for).
- **L5** (2026-11-18) A real cell's gripper has its own description and driver packages (DIME-LAB onrobot_rg2_description /
  onrobot_quick_changer_description / onrobot_rg2_control: meshes, mimic tree, width <-> joint arc); the paper's copied robot was
  reconciled against them but does not point at them. Onboarding a real cell should ask for the cell's own description packages
  and the driver's width <-> joint model before copying a robot record, so a lab's finger state can be driven from the real
  gripper's reading.
