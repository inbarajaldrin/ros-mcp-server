# Lessons for DroidStack from the real cell

Held here, NOT sent to plan47, until the operator's final decision (PLAN.md R10). Each entry: date, what the real cell showed,
the evidence, and what DroidStack would need.

- **L1** (2026-11-18) A real robot cell needs its perception node to be an adapter behind the robot contract, and the camera's
  mount transform changes with hardware (the D415 moved to a new printed mount, 11-10): a lab needs a way to re-declare a camera
  mount and re-qualify it without re-onboarding the robot. Evidence: `docs/findings.md` F1, robo-chain TF-FOR-AGENT.md.
- **L2** (2026-11-18) The cell PC beside a real robot must never share a DDS domain with simulated runs (plan29 P29-1): the dispatch
  DDS island is a safety control, not a convenience. A real-cell DroidStack setup should state and enforce this.
