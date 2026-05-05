# Finding — line_green seat depth is a geometric constant

**Invariant** (new, **I010**): line_green peg geometry bottoms at 7.3 ± 0.4 mm post-contact across 18 of 20 GOLD demos. IQR=0.35 mm. The other 3 FMB1 objects (u_orange, u_brown, inv_u_yellow) all bottom at 30–32 mm with comparable per-object IQR. line_green is shorter — geometric, not algorithmic.

**Evidence**: `metrics.json` in this dir. Source script: `analysis/scripts/11_line_green_seat_depth.py`.

**Why this matters**:
- Confirms I002 portability (post-collapse dz/dt = -6.4 mm/s) is universal across all 4 objects — line_green just truncates earlier.
- Universal seat predicates that hardcode `final_z_drop ≥ 25 mm` will never fire on line_green. Need per-object depth or fraction-of-expected.
- One of the four targets in PROJECT.md scope (line_green) is now characterized geometrically.

**Portability**: object-specific (line_green only; the *finding* generalizes — every object has a CAD-determined seat depth).

**Discovered**: 2026-05-05.
