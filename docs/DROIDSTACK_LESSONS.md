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
- **L6** (2026-11-19) Our refusals L3/L4 (ROAD_PROBE_FAILED "system-mode probe not built yet", BOX_NO_RESOURCE_CLASS on 8192/16384 MiB)
  were the command host's OWN stale engine (abbd0a57, platform 2648fe53, 2026-11-13): dev had built the system-mode Kit/render_warm/
  ROS probes and measured the cardless sizes (1024 / 256 MiB) within two days. Nothing told us: `doctor`, `status` and the refusal
  text all read as a platform gap. Needed: a refusal or doctor row that says when the engine is behind the fleet/dev ("probe built in
  a52865ca; this engine is 2648fe53"), so a lab doesn't chase a gap that's already closed.
- **L7** (2026-11-19) A lab that pointed Kit's private kit-state/caches inside a request's published output root (to keep them out of
  the shared Isaac install) made Kit write a symlink into published outputs; the supervisor refused the link but then retried forever
  instead of settling the request, which closed DroidStack admission on dual for every user (request 6511771…). plan47's fix lane:
  settle as invalid with the reason at finish, and move Kit's kit-state out of published outputs. Lesson for labs: where Kit's state
  may live should be the platform's decision, stated in the docs, not each lab's.
