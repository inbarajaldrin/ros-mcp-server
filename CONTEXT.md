# FMB assembly cell

A UR5e + RG2 cell that assembles FMB peg-in-hole parts onto a base that is not fixed, through the seams in
`compliant_insertion_studio/docs/ARCHITECTURE_SEAMS.md` (vision → planner → drive-mate → ExecuteMate).

## Language

### Poses

**Measured base**:
The base pose from the latest camera or depth read, before any correction.
_Avoid_: observed base (when an aim is meant), base position

**Aim**:
The base pose an insert is sent to: the **Measured base** moved by **Corrections**. It carries the corrections that made it (`seams/aim.py`).
_Avoid_: target (for this), final base pos, corrected base

**Correction**:
A named, bounded shift or turn of an **Aim** with its source record (follow, bias, yaw_exp).
_Avoid_: offset, fudge

**Follow**:
A **Correction** toward where the parts a part sits among actually seated.
_Avoid_: seated offset (that is a record), neighbour shift

**Seat**:
Where a part rests when inserted, from the **Seated CAD**.
_Avoid_: hole, pocket (when the CAD seat is meant)

**Seated CAD**:
The assembly file with each part dropped onto its supports (`fmb_assemblyN_seated.json`); the reference every seat measurement is against. The nominal assembly file keeps design clearances.
_Avoid_: "CAD is wrong", nominal CAD (when the seated one is meant)

**Seat error**:
How far a seated part sits from its **Seat**, with the frame it is expressed in and its source.

### Seams

**Mate**:
One part-into-reference relation of the assembly graph (`fmb1/mate:00`); the unit drive-mate drives and ExecuteMate certifies.

**drive-mate**:
The insertion driver behind `/ur5e/drive_mate`: brings the held part into the **Mate**'s seated set under uncertainty.
_Avoid_: insert primitive, translate_object (that is one adapter's implementation)

**ExecuteMate**:
The certifier: says whether a **Mate** is seated. Does not move parts.

**Episode**:
The ExecuteMate journal since the last certified first mate.

## Relationships

- An **Aim** is made from exactly one **Measured base** and zero or more **Corrections**.
- A **Follow** reads the **Seat errors** of parts already seated.
- drive-mate takes one **Mate** and a target; ExecuteMate certifies each **Mate** once per **Episode**.

## Example dialogue

> **Dev:** "line_green missed by 6 mm. Was the measured base off?"
> **Operator:** "Check the aim first. The measured base was fine; the follow correction had no seat errors for the U parts, so the aim was the bare base."

## Flagged ambiguities

- `observed_base` in the replay held both the **Measured base** and an **Aim** (servo shifts folded in). Resolved: two
  terms; `args.measured_base` is the measurement, `seams.aim.Aim` is what is sent.
- "target" in DriveMate.action is a vision pose of the reference, but the replay sends an **Aim**. Open until drive-mate owns the aim (AR3).
