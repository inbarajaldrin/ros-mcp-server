# Insert geometry: what each part meets on the way down, and where it actually hangs

Data: every logged insert since 2026-10-17 (`compliant_insertion_studio/logs/insert_*.csv` + `.meta.json`,
127 inserts), read with `ablations/insert_stages.py` (time at each height above the seat) and the
dwell-band probe in the 2026-10-19 session (tilt, xy, yaw, Fz at the dwell). CAD: the seated meshes,
rotated by the assembly quaternion (`cad_geometry._load_seated_top_above_origin`, fixed 2026-10-19 —
before that the seated U parts' tops were 24.5 mm too low). Heights are in the base frame (base top = 17.5 mm).

## The structure (CAD)

| part | seated bottom..top (mm) | what it meets going down (height above its seat) | clearance in the last stage |
| --- | --- | --- | --- |
| u_brown, u_orange | −13.5..56.5 (on their side, open top) | base rim +31.0 → pocket (seat) | spiral finds it within 8 mm |
| line_green | 11.6..37.4 (bar, 140 × 25.9) | tops of both U parts +45.0 → **inside both U openings** (28 mm wide) → rests on the U inner floor at 11.0 = its seat (+0.6) | ±1.1 mm across the bar; yaw < 1.3° |
| inverted_u_yellow | −8.0..62.0 (legs 31.5 tall, bridge underside 42.4) | top of line_green +45.5 → **legs astride line_green** → base rim +25.5 → pockets (seat) | ±1 mm astride line_green; pockets x ±0.6, y ±0.5 |

So line_green's seat is the U parts' interior, not the base (it never touches the base rim: 0.8 % of its
time there), and yellow's seat is reached only after its legs have threaded line_green for 20 mm.

## Where the time goes (all inserts since 10-17, ACTIVE phase)

| part | n | on a CAD surface | at the seat | in a corridor (no surface) |
| --- | --- | --- | --- | --- |
| u_brown | 35 | rim 68.5 % | 6.3 % | — |
| u_orange | 33 | rim 66.5 %, u_brown's top 1.0 % | 6.2 % | +28 mm 5 % (rim with the part tilted in the jaws) |
| line_green | 28 | U tops 8 % | 23.7 % | **+37..39 mm: 58 %** = 7 mm into the U openings |
| inverted_u_yellow | 31 | line_green's top 0.2 %, rim 0.8 % | 23.8 % | **+30..32 mm: 58 %** = legs 14 mm down over line_green |

The U parts' insert is one stage: rim contact, spiral, drop. The two prismatic parts are two-stage and
the time is spent **wedged in the first corridor**, where CAD has no surface:

- line_green, dwell +37..39 mm, Fz −5 N, TCP tilt 0.02°: the bar's bottom is 7 mm below the U tops.
  In the two 10-19 cycle-3 aborts its xy at the dwell was 0.9 / −0.1 mm across the bar (fine) and
  −7.3 / −5.7 mm along it (free direction) with a −4.5° commanded turn: a **yaw wedge**. Successful runs
  cross the band in 1.7–6.8 s with the turn ≤ 3°.
- yellow, dwell +30..32 mm, Fz −4.7..−5.1 N, TCP tilt 0.02°: the legs are 14 mm engaged astride
  line_green, 5.5 mm above the rim. Aborts had x (across line_green) −5.3 mm or y −2.6..−6.0 mm; the
  successful runs pass the band in 1–3 s at |xy| < 0.5 mm. The 39 s success (14:03) sat at (−0.1, −0.5)
  mm and still took 39 s: a yaw wedge again (the compliance has Rz free on attempt 1 only).

The TCP never shows the wedge as tilt (0.01–0.03° in every run): the RG2 pads and the part take the
compliance, not the flange. So a corridor stage is recognised by **height between two CAD stages +
pressing force + no descent**, never by tilt. The old labels ("tilt_unrecoverable", "between contacts")
were naming this without the geometry.

## What this means for one primitive (game plan, Track B)

Per stage, derived from the seated meshes at run time (no per-part table):

| stage kind | detect | allowed motion | exit |
| --- | --- | --- | --- |
| free descent | height above the next surface | z only | height reached |
| surface contact (rim, a seated part's top) | height = CAD surface ± 2 mm, Fz rises | xy search (spiral/directed), Rz locked | drop > 2 mm |
| corridor (clearance < 1.5 mm, length from CAD) | height inside the corridor span, Fz ≥ cap, no descent | Rz free, xy compliant within the clearance, re-aim from the measured slide (bounded by the clearance) | height past the corridor's end |
| seat | height = seat ± 2 mm, motion stopped | settle | sustained 1 s |

Recovery is a function of the stage, not of the attempt number: a corridor wedge gets a small lift
(≤ the engaged depth) and a re-aim by the measured slide; a surface stall gets the search; a seat
within tolerance is accepted even when the aim was off (compliance found the slot).

## Open

- yellow's and line_green's corridor stages need the yaw at the wedge relative to the base, not the
  world (base yaw changed 1 → 3° on 10-19 afternoon); the probe used world yaw.
- u_orange's +28 mm dwell (5 %) and its seat spread (y p10–p90 8.9 mm) are the U parts' own open item:
  the part sits differently in the jaws run to run (grasp offset), see the game plan's Track A.
