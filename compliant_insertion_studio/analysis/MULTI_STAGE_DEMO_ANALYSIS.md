# Multi-stage demo analysis (n=1: inv_u_yellow GUIDED 2026-05-06)

Offline analysis of `insert_inverted_u_yellow_20260506_095933.csv` using the
new CAD-derived contact classifier. This is the only existing inverted_u_yellow
GUIDED demo; treat findings as preliminary until the planned 5-demo collection
lands.

## Z-clustered contact regions

CAD candidates (target=inverted_u_yellow grasp_id=2, base1, with u_brown +
u_orange + line_green already seated):

| Surface | predicted TCP-z |
|---|---|
| SEAT (final) | 219.90mm |
| BASE_RIM | 245.40mm |
| OBJ:u_brown | 259.90mm |
| OBJ:u_orange | 259.90mm |
| OBJ:line_green | **265.35mm** |

Empirical sustained-contact clusters from the run:

| # | z (mm) | duration | classifier | residual |
|---|---|---|---|---|
| 1 | 263.41 | 4.82s | **OBJ:line_green** | −1.94mm ✓ |
| 2 | 244.47 | 34.81s | **BASE_RIM** | −0.93mm ✓ |
| 3 | 226.77 | 2.31s | UNKNOWN (best SEAT) | +6.87mm |
| 4 | 213.65 | 2.17s | UNKNOWN (best SEAT) | −6.25mm |

Clusters 1 and 2 match CAD predictions to ±2mm — the classifier identifies
"on line_green's top" vs "on base rim" reliably. Clusters 3 and 4 are
transitional/final, neither at clean candidate surfaces. Cluster 4 (213.65mm)
is the final resting position — 6.25mm BELOW predicted SEAT, consistent with
the FALSE_SEAT_CLAIM verdict (descent_mm=49.7mm exceeded the 45mm "real seat"
threshold) and supports the hypothesis that the run did NOT seat correctly.

## Operator drag pattern between intermediate and base contacts

```
t=0         APPROACH descent
t≈3s        peg lands on line_green-top (cluster 1, z=263.41mm)
t=3..8.5s   operator drags peg laterally on line_green
              path = 16.7mm,  net displacement = 11.7mm
              F_lat median 1.45N, max 6.08N
              Fz median 8.21N (descent force pressing peg down)
t=13.85s    operator pressed SIGUSR1  (note: BEFORE peg dropped to base!)
t≈14s       FSM → INSERT_DESCENT (Fz=-9N, xy locked at SIGUSR1-time tcp_xy)
t=15.42s    peg arrives at base rim (cluster 2, z=244.47mm)
              transition through line_green hole took ~1.4s
t=15.42..51.86s  peg sat at base rim 36 seconds
              not seating — xy locked from SIGUSR1 was line_green-hole, not base-hole
t=51.86s    finally broke through to lower z
t=53..58s   transitional contacts (clusters 3, 4)
t=58s       final tcp_z=213.65mm (FALSE_SEAT_CLAIM per verifier)
```

## Key findings

1. **The current GUIDED → single-SIGUSR1 → INSERT_DESCENT flow does NOT work
   for two-stage inserts.** It only captures ONE hole position. The operator
   marked the line_green hole, but the actual base hole has different xy.
   After the peg dropped through line_green it locked onto the WRONG xy and
   couldn't seat.

2. **Operator's SIGUSR1 timing is ambiguous in two-stage runs.** Was the
   operator marking the upper hole (line_green) or the final base hole? The
   meta records one position but doesn't disambiguate.

3. **The CAD classifier reliably distinguishes line_green-top vs base-rim
   contact** at sub-2mm accuracy, even with line_green sticking ~19mm above
   the base outer top.

4. **The "stuck at base rim" period (36 seconds at z=244.5mm)** is the
   smoking gun for the missing second-stage. Peg has Fz=-9N pushing down,
   but xy is misaligned with base hole, so peg doesn't seat. Operator lost
   control after SIGUSR1.

## Implication for the FSM

A multi-stage state machine is required:

```
APPROACH ──┬─ contact: classify ──┬─ SEAT       → DONE  (single-stage seat-on-touchdown — already fixed)
           │                       ├─ BASE_RIM   → GUIDED/SEARCH (today's path)
           │                       └─ OBJ:<name> → STAGE_A
           │
STAGE_A: peg on intermediate.
   - GUIDED variant: lock Z (current behavior), let operator drag and SIGUSR1
     when above intermediate's hole
   - autonomous variant: drop Fz, push through hole (no spiral — would drag the
     intermediate sideways)
   - exit: detect z-drop transition to next contact cluster
   ↓
STAGE_B: peg on base rim (or next intermediate) — recursively classify
   - if BASE_RIM: route to existing GUIDED/SEARCH/INSERT_DESCENT path
   - if next OBJ:<name>: recurse to STAGE_A on that object
   ↓
DONE
```

The FSM needs `STAGE_A` (intermediate-engagement) state with semantics:
- locked Z (operator drag) OR force-press-down-no-spiral (autonomous)
- transition out triggered by z-drop to next CAD-candidate surface

For data collection: each demo should capture **two SIGUSR1 events** (one per
hole), and the meta should record `hole_observed_operator_stages: [stage_a, stage_b]`.
The collection script's prompt loop needs to wait for the next GUIDED entry
after the first SIGUSR1, not assume one-and-done.

## Force/torque signature per phase

Statistics over the 60s ACTIVE phase, broken into the contact-cluster windows:

| Phase | t window | F_lat median (N) | F_lat P95 (N) | T_lat median (Nm) | T_lat P95 (Nm) |
|---|---|---|---|---|---|
| APPROACH (peg in air) | 0..3s | 0.33 | 0.84 | 0.013 | 0.033 |
| ON line_green TOP | 3.5..8s | 1.51 | 3.19 | 0.304 | 0.689 |
| FALLING through line_green hole | 8.5..15s | 0.46 | 2.11 | 0.026 | 0.227 |
| **STUCK AT BASE RIM** (35s) | 15.5..50s | 2.03 | 3.25 | 0.250 | 0.361 |
| **PRE-BREAKTHROUGH** (last 2s) | 50..52s | **6.73** | 12.16 | **1.181** | 2.317 |
| DROPPING through base hole | 52..56s | 14.58 | 25.38 | 3.013 | 5.875 |
| FINAL (after-drop) | 56..60s | 11.69 | 13.14 | 2.791 | 2.987 |

Key signatures:

- **`stuck-at-rim` vs `dragging-on-line_green-top`** are NOT cleanly
  separated by F_lat alone (2.03N vs 1.51N). T_lat doesn't help either
  (0.25Nm vs 0.30Nm). To discriminate "wedged" vs "operator drag",
  better feature is **z-velocity stability**: stuck-at-rim has z_std of
  0.039mm over 35s (locked); dragging has higher transient variation.
- **Pre-breakthrough** has 3-5× F_lat and T_lat ramp-up vs stuck steady-state.
  This is the autonomous-detection signal — F_lat sustained > ~5N or T_lat
  > ~1Nm could trigger "we're about to drop, prepare for stage transition."
- **Dropping-through-hole** has 7× F_lat over stuck (14.58N vs 2.03N) and
  high z velocity. Easy to detect.

For the multi-stage FSM, this gives:
- **Stage A→B transition (intermediate hole drop)**: detect z-drop spike
  + F_lat transient + tcp_z transition to next CAD-candidate cluster.
- **Stuck-at-base-rim alarm**: if `|dz/dt| < 0.5mm/s` for >5s AND
  `tcp_z ≈ expected_base_rim` AND `fz > 5N`, declare "wedged on rim, need
  lateral search." Trigger the spiral SEARCH director (the GUIDED→
  INSERT_DESCENT-locked-xy path can't recover from this).

## Open questions (to answer with the 5-demo collection)

1. Do all 5 inverted_u_yellow demos produce the same line_green-top → base-rim
   z-transition, or does the operator sometimes find an unblocked path through?
2. What's the variance in drag-path length on line_green's top? (1 demo:
   16.7mm path for 11.7mm displacement — operator was inefficient.)
3. Is the line_green-hole xy consistent with CAD prediction? (We have one
   data point — needs more for variance.)
4. Does the autonomous version need a force-press signature (Fz spike pattern)
   to detect "peg dropped through hole" vs "peg still wedged"?

## Cross-run classifier validation (n=202 historical runs)

A separate offline replay against every existing `insert_*` run with valid
CAD prediction confirmed the fix is broadly safe:

- **0 false-positive SEATs** — no `outcome=success` run had contact tcp_z
  within 5mm of predicted seat. The fix wouldn't prematurely DONE any
  previously-successful insertion.
- **2 rescued abort cases** — both are u_brown runs where peg sat at SEAT
  depth but FSM aborted on lateral_stall before recognizing the seat:
  - `insert_u_brown_20260506_082007` (residual −1.74mm, lateral_stall)
  - `insert_u_brown_20260506_195220` (residual −2.19mm, today's smoke test)
  Both would correctly DONE under the new SEAT short-circuit.
- **Best-match label distributions match physics**:
  - 196 runs best-match BASE_RIM at 0.86mm median |residual| (the dominant
    first-contact mode for autonomous u_brown / u_orange runs that need
    the spiral)
  - 2 runs best-match SEAT at 1.96mm median |residual| (the seat-on-touchdown
    cases now correctly handled)
  - 2 runs best-match OBJ:line_green at 1.60-1.91mm |residual| (the only
    inverted_u_yellow runs that hit the upper hole before any SIGUSR1)

Per-object best-match distribution matches the expected physical regime:
- u_orange: 179 BASE_RIM (autonomous spiral path)
- u_brown: 8 BASE_RIM, 2 SEAT (mostly spiral, occasional perfect alignment)
- inverted_u_yellow: 11 BASE_RIM, 2 OBJ:line_green (multi-stage, sparse data)

## Files referenced

- `compliant_insertion_studio/wrapper/cad_geometry.py` — classifier (committed)
- `compliant_insertion_studio/logs/insert_inverted_u_yellow_20260506_095933.{csv,meta.json}` — demo
- `ablations/eval_resources/fmb1_assembly.json` — assembly_order
- `~/Documents/aruco-grasp-annotator/data/fmb_assembly1.json` — seat poses
- `~/Documents/aruco-grasp-annotator/data/wireframe/line_green_wireframe.json` — bbox
