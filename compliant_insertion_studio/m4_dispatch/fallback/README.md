# Fallbacks for fmb1_assemble_ur5e_rg2_realcell_gen

## A. The engine refuses a FREE Body1 (occurrence:00)

Nothing in `sim_instrumentation/layout_generator.py` forbids drawing the reference part (free occurrences are
address-sorted and each needs only a domain; statics come from `pose_from_fixed_layout`), but no published task has
drawn it. If `python -m sim_instrumentation.bundle_layout` refuses with a code naming occurrence:00, make Body1
static at the MEASURED real-cell pose instead:

1. copy `fmb1_realcell_base_static.json` into the bundle as `layouts/fmb1_realcell_base_static.json`;
2. in `task.yaml` replace
   ```yaml
   free: [occurrence:00, occurrence:01, occurrence:02, occurrence:03, occurrence:04]
   static: []
   ```
   with
   ```yaml
   free: [occurrence:01, occurrence:02, occurrence:03, occurrence:04]
   static:
     - occurrence: occurrence:00
       pose_from_fixed_layout: fmb1_realcell_base_static/Body1
   ```
   delete the `occurrence:00:` domain line and the `base_start` region, and add under `layouts:`
   ```yaml
   fixed:
     - name: fmb1_realcell_base_static
       path: layouts/fmb1_realcell_base_static.json
       sha256: <sha256sum of the file>
       frame: world
       provenance: "Body1 at the measured base1 pose of real run 78 (see the record); movers = fmb1_flat_start's witness positions"
   ```
3. `world_from_assembly` stays `witness_only` (the referee reads placement from `in_region: assembly`).

The static pose keeps the base at a REAL placement (0.107 m toward -x of the twin's (0.005, 0.48), turned 201 deg in
world), so a planner that hard-codes the twin's declared site still fails - which is the M4 point - but it loses the
per-episode base variation. The orientation map cell->graph (Rz(-90), +90 deg) is unverified: if a draw's Body1 looks
turned the wrong way against the real photo in README §2, the map is the first suspect.

## B. The operator wants the UNTIPPED (standing) start instead

Replace the three `stance:` domains with the control's (no stance key, `flip` absent):
```yaml
occurrence:01: {region: staging, yaw: free, rest: settled}
occurrence:03: {region: staging, yaw: free, rest: settled}
occurrence:04: {region: staging, yaw: free, rest: settled}
```
That is the cell's phase-1 start (runs 93-108): every part in its seated orientation up to yaw, 1 pass each.
