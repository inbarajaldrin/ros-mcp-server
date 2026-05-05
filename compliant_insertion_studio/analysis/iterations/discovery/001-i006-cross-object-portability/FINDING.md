# Finding — 001-i006-cross-object-portability

## Headline

**I006 (path/bbox ratio as wide-arc-vs-tight-spiral discriminator) DOES NOT generalize across objects. The "wide arc" GOLD signature is u_orange-specific.**

I006 must be downgraded — keep its u_orange-only descriptive value, but DO NOT use the path/bbox ratio (or bbox magnitude alone) as a portable cross-object FSM trigger. Per-object thresholds would be required, and even then the FAIL u_orange overlaps three of four GOLD groups.

## Cross-object table (10 s post-contact window, May-3 GOLD demos)

| Group              |  n | path_len_mm (p50) | bbox_diag_mm (p50) | path/bbox ratio (p50) |
|--------------------|---:|------------------:|-------------------:|----------------------:|
| GOLD u_orange      | 10 |             13.66 |             4.67   |                2.82   |
| GOLD u_brown       | 10 |             13.12 |             2.11   |                6.26   |
| GOLD inv_u_yellow  | 20 |             12.69 |             2.18   |                5.98   |
| GOLD line_green    | 20 |             12.64 |             2.61   |                4.82   |
| FAIL u_orange      | 69 |             11.37 |             2.17   |                5.38   |

## What the numbers say

1. **Path length is universal** (~12-14 mm in 10 s). Both successful demos and FAIL attempts move the TCP about the same total distance. This piece of I006 stands.
2. **Bounding-box magnitude is u_orange-specific.** GOLD u_orange sweeps ~4.7 mm; GOLD u_brown / inv_u_yellow / line_green only sweep ~2.1-2.6 mm — *the same as FAIL u_orange* (~2.2 mm). The "wider physical arc" claim from FINDINGS.md §3 was a u_orange artifact, not a universal signature.
3. **Path/bbox ratio: u_orange GOLD is the outlier.** u_orange GOLD ratio = 2.82, while every other GOLD group sits at 4.8-6.3 — straddling or exceeding the FAIL u_orange median of 5.38. There is no universal threshold that separates GOLD from FAIL on path/bbox alone.
4. **GOLD ratio spread (max-min) = 3.44** — larger than the entire GOLD-FAIL gap on u_orange. The metric is dominated by per-object geometry, not by success/failure quality.

Hypothesis for why u_orange GOLD differs: u_orange's slot geometry (deep, square cross-section near the rim) demands a wider lateral hunt to engage the chamfer; the other three parts seat with more compact motion. That's a per-object **insertion mechanics** difference, not a universal control-law signature.

## Implications for staged hypotheses

- **H103 ("Stuck-at-rim auto-replan: command 3mm jog if Fz_t > 6 N for 2 s with xy_disp < 1 mm")**: the `xy_disp < 1 mm` part is portable (FAIL u_orange and many GOLD demos overlap at ~2 mm bbox; tightening to <1 mm picks up only the truly stuck cases). The path/bbox ratio component should be dropped from H103.
- **H101 ("Replace spiral with directed sweep")**: still backed by I003 (direction) + I004 (1.2-1.7 mm displacement) + I005 (F_lat magnitude). Does not depend on I006.
- **The "tight spiral vs wide arc" framing from FINDINGS §3 should be retained as a u_orange diagnostic**, not a universal control gate.

## Recommended STATE.json edit

I006: change `"portability": "verified_u_orange_only — needs verification on other objects"` → `"portability": "verified_u_orange_only_REFUTED_other_objects (GOLD u_brown / inv_u_yellow / line_green sweep <2.6mm bbox, ratio 4.8-6.3, overlapping FAIL u_orange 5.38)"`.

Mark this open question removed.

No new pending hypothesis surfaces (the negative result rules out a path-shape FSM gate, doesn't motivate a new one).

## Files written

- `HYPOTHESIS.md` — question
- `FINDING.md` — this file
- `metrics.json` — machine-readable per-object stats + portability decision
- analyzer at `analysis/scripts/10_path_bbox_cross_object.py`
- analyzer output at `analysis/data/path_bbox_cross_object.json`
