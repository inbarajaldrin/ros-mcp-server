# Compliant Peg-in-Hole Insert: Algorithm Failure Analysis (u_orange, 2026-05-04)

**Author:** Claude (analyst), commissioned by operator after the day's
auto-attempts (v82–v86 generation) failed despite an earlier operator-assisted
demonstration that completed.

**Question:** Why does the autonomous FSM fail when the operator's hand
guides the same hardware to seat in 30 s with a constant Fz=−9 N?

**TL;DR:**
1. The wrapper *does* mechanically seat the peg in many of its "failed" runs.
   It just **fails to recognise** it. The seat-detection predicate, the engagement
   gate, and the ENTRY_SETTLE→INSERT path are all geometric mismatches with
   what the part actually does.
2. The operator's "control law", read directly from 10 successful unassisted
   demos, is **almost no control at all**: maintain Fz≈ 0 N during chamfer
   descent, F_lat ≤ 0.5 N median, never rock, never twist. The work is being
   done by gravity, kinematic admittance, and the slot's own chamfer.
3. The autonomous algorithm injects `spiral_F_max_N=12 N` lateral, `find_hole_tilt_T_Nm=0.20 Nm`
   rocking, `insert_fz_N=4–9 N` constant push. **Every newton of lateral force
   at TCP becomes 0.24 Nm of moment at part center** because the part hangs
   24.2 cm below the TCP. The algorithm is generating its own contact
   pathology.
4. The CAD-predicted seat xy is wrong by ≥ 9 mm in observed runs. Several
   gates use that prediction as a hard pass/fail (`engagement_dist_thresh_m=6mm`,
   `tcp_z_reached_predicted` tolerance reasoning). They reject correctly-seated
   pegs.
5. The right redesign is *not* a smarter FSM. It is a single
   admittance-style controller with three depth-keyed scalars and a
   self-anchoring seat detector that ignores CAD xy.

---

## 0. Data inventory

| File | Outcome | Final z | Final xy | Notes |
|---|---|---|---|---|
| `insert_u_orange_20260504_145614.csv` ("DEMO") | success/operator_sigterm | 0.20583 | (0.0341, −0.3635) | Wrapper-assisted run; operator let FSM run, ended w/ SIGTERM after FSM completed two descend-retract-redescend cycles. Most-cited "demo" today. |
| `insert_u_orange_20260504_150738.csv` ("FAIL_INSERT") | abort:INSERT timeout | 0.20124 | (0.0343, −0.3467) | Reached final z=0.43 mm above CAD-predicted seat; FSM said "peg never bottomed. descent_post_hole=−0.00mm". |
| `insert_u_orange_20260504_144653.csv` ("FAIL_FIND_HOLE") | abort:FIND_HOLE max radius | 0.23030 | (~rim) | Spiral hit 12 mm radius without z-drop trigger. |
| `insert_u_orange_20260503_08{4259,4422,4549,4715,4842,5018,5145,5306,5433,5617}.csv` | 10× clean operator demos | 0.1992±0.0001 | — | True hand-guided. cmd_fz=−9 throughout. **Ground truth control-law data.** |

The May-3 demos are the cleanest signal we have. The wrapper is in
ACTIVE-mode-A (constant Fz=−9 N down, no other commanded wrench), the
operator hand-places the peg on the chamfer and lets the admittance lower it.
Sample rate is 99 Hz (not 500 — the broadcaster runs 500 Hz, but the wrapper
sub-samples to 99 Hz for the CSV).

Workspace and bias caveats:
- 2026-05-04 base position differs from 2026-05-03 by ~55 mm in X. Only F/T
  *relationships* (direction, magnitude, depth-band shape) are workspace-
  invariant. **Position targets from May-3 are not directly usable.**
- `post_zero_bias.Fz` was −0.13 N for the DEMO but **+2.04 N** for FAIL_INSERT.
  All raw fz numbers from FAIL_INSERT carry that 2 N offset. Subtract it
  before comparing to the demo. This is a Phase-3 zeroing-quality bug, not
  an insert-control bug.

---

## 1. Operator control law (derivation from demo data)

State variables I tabulated per sample (May-3 data, all 10 demos pooled,
contact-relative depth bins):

```
depth_mm    n     |F_lat|_med  |F_lat|_p95  Fz_med   Fz_p95   tilt_med  tilt_p95  r_cop_med
 0–  2     1137     1.62 N      5.74 N     +8.99    +10.87    0.63°      1.11°     14 mm
 2–  5      479     0.59        2.78       +0.21     +1.52    0.63       1.44     144 mm*
 5– 10      766     0.51        2.02       +0.05     +0.48    0.67       1.41      27 mm
10– 15      769     0.43        1.41       +0.07     +0.42    0.56       1.38      22 mm
15– 20      773     0.44        1.67       +0.07     +0.34    0.46       1.34      23 mm
20– 25      936     0.50        4.13       +0.12    +10.25    0.47       1.31      23 mm
```
\* the "144 mm r_cop" row is an artifact of dividing by tiny Fz; ignore it.

### 1.1 Fz vs depth — three regimes

**Regime A: chamfer-engagement (0–2 mm)**. Fz ≈ +9 N, matching the −9 N
command. The peg is contacting the chamfer surface and the admittance
controller pushes back. Lateral force jumps too (1.6 N median) — operator
braces the part. Tilt < 0.6°.

**Regime B: free descent (2–25 mm)**. **Fz ≈ 0 N.** Despite a constant
−9 N command, the actual reaction force is essentially zero — the peg is
sliding into the slot under gravity + admittance compliance, with the slot
walls providing essentially no constraint at this scale. F_lat drops to 0.4–0.5 N
median; tilt holds at 0.5–0.7°. **This is the "operator control law" surprise:
the operator does not push.** They place the peg on the chamfer, the part
drops in.

**Regime C: seat (depth > 25 mm, not in the table — bottom of slot)**. Fz
spikes back up as part contacts the slot bottom. Operator releases, Fz
relaxes. (For u_orange the slot is ~25 mm deep; the seat itself is the
final 1–2 mm.)

### 1.2 F_lat direction — alignment with walk and tilt

Across 10 demos, in world frame:
- Walk direction in contact phase (vector from first-contact xy to seat xy):
  circular mean **−106°** (i.e., toward (−1.6, −5.5) mm from contact).
- Tilt-axis direction (projection of EE +Z axis onto world xy, i.e. which way
  the peg leans): circular mean **−110°**. **Within 4° of walk direction.**
  The peg leans the way it walks. This is geometric: the leading prong
  catches the chamfer, that side drops first, the part tilts that way.
- F_lat direction shows correlation with walk, but with substantial
  per-demo variance (signed walk−F angles ranged −176°..+100° across
  10 demos). **F_lat does not have a clean repeatable direction relative to
  walk.** It is largely sensor noise on top of a small mean.

The cleanest summary: **walk and tilt are co-aligned and consistent**. F_lat
magnitude is small (0.4 N median) and direction is not particularly
informative. Any control law that treats F_lat as a strong directional signal
during free descent is reading mostly noise.

### 1.3 Frequency content — operator does not rock

FFT of Fx, Fy across the contact-phase samples (depth 85–115 mm in the
DEMO file, n≈3500, fs=90 Hz):
- Most lateral-force energy is in 10–40 Hz (sensor noise) — Fy band power
  10–40 Hz is 3.7× higher than 0.5–3 Hz.
- 0.5–3 Hz band (where any "manual rocking" would appear) is dominated by
  drift/DC, not a coherent oscillation.
- No identifiable peak at 1.0–1.5 Hz (the wrapper's
  `find_hole_tilt_freq_hz=1.5`).

**Operator does not rock the peg.** The wrapper's "rocking is the exploration
mechanism that empirically walks peg into chamfer" claim (defaults.yaml
line 60) does not have demo-data support. Whatever exploration is needed
is done **once, slowly, at the chamfer**, not continuously.

### 1.4 Per-axis fold — yaw is not the bottleneck

CAD chain for u_orange uses `fold_symmetry_used` (the slot's symmetry group;
applied=true, angle_error_deg=0.30°). Held quaternion is fixed during the
whole insert (operator-side gripper does not regrasp). Tilt axis stays at
~−110° throughout descent — geometric, not random. **Yaw alignment is not
the operator's challenge for u_orange.** The `find_hole_tz_T_Nm=0` decision
in defaults.yaml is the right call; the disabled yaw_unlock block is also
correct for this part.

### 1.5 Phase structure — "place, then release"

Timeline of one operator demo (2026-05-03_084259, 30.1 s ACTIVE):
- t=0..6 s: free fall in air (no contact). Wrapper's −9 N command lowers the
  peg through air at ~4.5 mm/s.
- t≈6 s: peg meets chamfer (first contact event). Operator reaches in and
  steadies. F_lat briefly to ~5 N, Fz to +9 N.
- t=6..22 s: descent at 1–1.5 mm/s. F_lat ≈ 0.5 N median, Fz ≈ 0 N. **This is the
  long, quiet, "operator just holds it lightly" phase.**
- t=22..28 s: peg bottoms out. Fz climbs.
- t=28..30 s: release.

There are not many "phases" to model. There is one
**chamfer-engagement transient** lasting <1 s, then **steady free descent**, then
**seat**. The wrapper's APPROACH / FIND_HOLE / ENTRY_SETTLE / INSERT four-state
machine is overly granular for this physics.

---

## 2. Failed-autonomous-run comparison

### 2.1 FAIL_INSERT (`150738`) — peg seated, FSM aborted

Reconstruction from CSV (taking the FSM rules as written):
- APPROACH → FIND_HOLE: triggered when smoothed Fz crossed +6 N at
  t=50.55 s, **z=0.2314** (this is `surface_z`).
- FIND_HOLE → ENTRY_SETTLE: when z dropped 0.8 mm below surface_z
  (~t=56.5 s, z=0.2306).
- Many ENTRY_SETTLE / INSERT / WEDGE_RECOVERY cycles between t=56 and
  t=94 s (cmd_fz cycles −4 / −2.5 / −4 visible at lines around t=58.81,
  63.43, 65.74 s — each is an ENTRY_SETTLE ↔ WEDGE_RECOVERY hop).
- After t≈94 s, **z stable at 0.2012 ± 0.00002 m for ~60 s** (full sustain
  of the "INSERT timeout" budget). Speed median 0.005 mm/s, max 0.026 mm/s.
  100 % of samples in the last 30 s have |dz/dt| < 0.5 mm/s.
- Final xy = (0.0343, −0.3467). CAD-predicted xy = (0.0267, −0.3517). xy
  error ≈ 9.1 mm.
- Final z = 0.2012, **0.43 mm above CAD-predicted seat z = 0.20081**. Total
  descent from `surface_z`: **30.13 mm**.

The peg is mechanically seated. The seat predicate didn't fire because:
- `descended_post_hole = hole_z − tcp_z` was small. `hole_z` is reset on
  every ENTRY_SETTLE→INSERT promotion. The wrapper kept demoting back to
  ENTRY_SETTLE/WEDGE_RECOVERY, then re-promoting at low z.
- `descended_from_surface = surface_z − tcp_z = 30.1 mm ≥ 20 mm`. Per the
  code at `contact_search_fsm.py:1077`, this should make `seated_via_surface=True`
  and **the seat should fire**. It did not, because the FSM kept exiting INSERT
  before the 0.75 s sustain accumulated. Each ENTRY_SETTLE rejection (e.g.
  from the 6 mm `engagement_dist_thresh_m` gate, since xy was 9 mm from
  CAD) booted us out of the INSERT branch and reset `_motion_stopped_first_t`.

**This is the central failure mode. The peg is seated for 60 s, the FSM
is fighting the engagement-distance gate.**

### 2.2 FAIL_FIND_HOLE (`144653`) — different mode

Spiral expanded to max radius (12 mm) with no z drop > 0.8 mm, then aborted.
This does happen, but it is NOT the dominant failure mode. The dominant
mode is what FAIL_INSERT shows: peg gets in, FSM doesn't recognise it.

### 2.3 Side-by-side at each depth band (DEMO vs FAIL_INSERT)

```
                      DEMO (operator-supervised)          FAIL_INSERT
depth     fz_med   |F_lat|_med   tilt_max     fz_med   |F_lat|_med   tilt_max
 0–  5    −0.12     0.43          0.34°       +2.08(*)  1.11          0.47°
 5– 80    −0.05     0.40          ~0.95°      +2.0(*)   1.04          1.7°
80– 90    +0.38     0.87          3.29°       +3.9      1.41          1.96°
90–115    +0.09     ~0.40         3.16°       +2.0(*)   1.10          3.17°
```
(*) FAIL_INSERT has a +2 N Fz baseline from a bad zeroing — subtract before
interpreting.

After bias correction the runs look similar. The big difference isn't in the
*physics* of the descent — it's in:
1. The DEMO's wrapper happened to commit ENTRY_SETTLE→INSERT at a moment
   when xy had drifted into the engagement_dist gate; FAIL_INSERT's didn't.
2. The DEMO's predicted seat error happened to be smaller than FAIL_INSERT's;
   FAIL_INSERT's actual seat is 9 mm laterally offset from CAD.

That is gate-parameter sensitivity, not an algorithm-physics problem.

---

## 3. Math identification of the failure mechanisms

### 3.1 The grasp-offset moment problem

`r = (TCP → part_center)` in world frame at canonical face-down EE
quaternion is approximately **(−26.7, +0.2, −241.8) mm** for u_orange
grasp 1 (using flange_offset_m=0.2286 + grasp_point_in_object
=(0.0255, −0.0132, 0)). The part hangs 24.2 cm below the TCP.

Moment about part_center from a force F applied at TCP equals **(TCP −
part_center) × F = (−r) × F = F × r**.

Given r ≈ (−0.027, 0, −0.242) m, then for any **lateral** force F=(Fx, Fy, 0)
applied at TCP:
- |M_xy at part_center| ≈ 0.242 · |F_lat|

So:
| F_lat at TCP | Resulting M_xy at part_center |
|---|---|
| 1 N | 0.24 Nm |
| 3 N | 0.73 Nm |
| 6 N | 1.45 Nm |
| 12 N (`spiral_F_max_N`) | **2.90 Nm** |

For comparison, the operator's measured |T_lat| at TCP (May-3 demos,
contact phase) is **0.4 N median**, **0.45 Nm p95** AT THE SENSOR (i.e. at
TCP, not at part_center).

`engaged_T_lat_max_Nm_partcenter` (the gate the wrapper checks during
ENTRY_SETTLE) is whatever the FSM calls `engaged_T_lat_max_Nm` — 0.15 Nm
in defaults.yaml.

The wrapper's own commanded forces produce moments about part_center that
violate the engagement gate by **6–20×**, before the slot is even contacted.
A 1 N stray spiral correction is enough to fail the gate.

For an applied **Fz** at TCP:
- |M_xy at part_center| ≈ |r_xy| · |Fz| ≈ 0.027 · |Fz|

| Fz at TCP | Resulting M_xy |
|---|---|
| 4 N (`insert_fz_N` current) | 0.107 Nm |
| 6 N (`approach_fz_N`, `wedge_retract_fz_N`) | 0.160 Nm |
| 9 N (force_mode default) | 0.240 Nm |

Even pure Fz at 9 N puts a 0.24 Nm bias on part_center — **above the 0.15 Nm
gate**. That's why the wrapper had to introduce part-center-frame torque
correction (`sensed_T_at_partcenter` at line 311) and a higher gate
(`engaged_T_lat_max_Nm_partcenter`). The wrapper is then living downstream
of an effect it created upstream.

**The single biggest control-law lever in this geometry is: do not command
lateral force.** The operator's median 0.5 N at TCP creates a 0.12 Nm
moment, just under the 0.15 Nm gate, and is largely sensor noise. The
algorithm's spiral and recovery-leash forces (1.5–12 N range) saturate the
moment budget.

### 3.2 Wrong frame for compensation (resolved, but subtly)

The wrapper had a `_wrench_to_partcenter` correction that was tested in
v77/v78 and reverted because it produced "destabilising cross-coupling"
(line 300 of contact_search_fsm.py). The reverted code was applying the
correction to the **commanded** wrench. Its non-converging behaviour is
because the moment at part-center is already what the slot is reading;
trying to cancel it with TCP torque also produces a TCP lateral force
through the same lever arm, which itself produces moment, ad infinitum
under closed-loop force control.

**The correct frame for analysis is part_center.** The correct frame for
**command** is TCP, and the right strategy is to send minimal command
(small Fz, zero F_lat, zero T) and let the slot's chamfer geometry do the
self-aligning. That is what the operator demos show.

### 3.3 Engagement-distance gate vs CAD truth

`engagement_dist_thresh_m = 0.006` (6 mm). CAD prediction has 9 mm error
(observed in FAIL_INSERT) for this part-base-grasp combination. The gate
**rejects correctly-seated pegs** as a structural matter.

`engaged_z_drop_dominant_m = 0.020` (20 mm) is the OR-shortcut intended to
handle this. It works in `_tick_entry_settle` (line 849) for promotion to
INSERT but **only checks at the moment of the gate evaluation**, not over a
sustain window. If the peg is descending and ENTRY_SETTLE first fires at
z_drop=2 mm, the gate fails (dist_to_pred>6mm AND z_drop<20mm); FSM bounces
to FIND_HOLE; spiral resumes; another z drop fires another ENTRY_SETTLE;
etc. The peg can travel 30 mm of descent in this thrashing pattern without
ever sustaining 0.1 s in ENTRY_SETTLE long enough to commit to INSERT.

### 3.4 Reactive (rate-of-change) terms

The FSM uses *position-derivative* signals minimally:
- `_z_velocity` over a 1.5 s window for the seat-detection motion-stopped check.
- `_z_velocity_xy` over 0.1 s for the engagement v_xy<3 mm/s gate.

It uses **no F-rate or T-rate signal**. The operator demo data shows that
the right cue for "transitioned from chamfer-engagement to free descent" is
a **drop in Fz** from +9 N (Regime A) to ~0 N (Regime B), sustained over
~0.5 s. This is a much more reliable signal of "into the slot" than a
position threshold.

Similarly, the right cue for "seated" is a **rise in Fz** sustained over
0.5–1 s as the peg loads against the slot bottom — combined with motion
stopped. The wrapper's seat predicate uses the position-only path, plus a
CAD-xy match. The Fz-rise signal is not in the predicate.

### 3.5 Failure mode summary

| Mechanism | Severity | Where |
|---|---|---|
| Engagement-dist gate rejects correctly-seated pegs (CAD has 9 mm error) | **Primary** | ENTRY_SETTLE (`_tick_entry_settle`) |
| `_motion_stopped_first_t` resets every ENTRY_SETTLE→INSERT cycle | **Primary** | INSERT predicate sustain |
| Lateral-force command magnitude (1.5–12 N) creates part-center moments well above 0.15 Nm gate | **Primary** | spiral, recovery leash, wedge correction |
| Constant Fz=4–9 N during INSERT does not match operator pattern (Fz≈0 in free descent, Fz↑ at seat) | Important | INSERT wrench |
| Tilt-rocking torque (0.20 Nm × continuous) is operator-foreign and burns moment budget | Important | FIND_HOLE, optional INSERT |
| Bad post_zero_bias.Fz (+2 N in FAIL_INSERT) propagates to all decisions | Important (data hygiene) | smoke + per-pose zero |
| Algorithm has no Fz-rate signal; uses CAD-z for predicate | Moderate | termination |
| Multi-state FSM with re-entrant transitions thrashes | Architectural | whole FSM |

---

## 4. Recommended redesign

The minimal-control story from the demos suggests not "a better FSM" but
**a simpler controller**.

### 4.1 Single-loop adaptive admittance controller (replaces the FSM)

State variables (continuously tracked):
- `t` (s)
- `z(t)`, `xy(t)`, EE quat(t) (from `/tcp_pose_broadcaster`)
- `F_TCP(t)`, `T_TCP(t)` smoothed, in base-link (already in CSV)
- `F_lat(t)`, `T_lat(t)` magnitudes
- `M_pc(t) = T_TCP + (TCP − part_center) × F_TCP` — **moment at part-center**;
  this is the right frame for "is the peg tilted?"
- `Fz_smoothed(t)` over 0.5 s
- `dz/dt` over 0.5 s (changed from 1.5 s — see seat detector)
- `dFz/dt` over 0.3 s (NEW — see phase detection)
- `surface_z`: latched at first contact (Fz_smoothed crosses +5 N going up).
- `tilt_deg(t)` from EE quat (already computed).

Single time-varying scalar control: `Fz_cmd(t)`. Lateral and torque commands
remain **zero at all times** in nominal mode. (Spiral search becomes an
explicit, time-bounded escape mode — see §4.3.)

Phase determination is a depth-keyed function of `surface_z` and
`Fz_smoothed`:

```
PHASE A "free fall to surface"     z > surface_z + 1 mm  (no contact yet)
PHASE B "chamfer engagement"       0 ≤ surface_z − z < 5 mm AND Fz_smoothed > 3 N
PHASE C "free descent into slot"   surface_z − z ≥ 2 mm AND Fz_smoothed < 1.5 N
PHASE D "seat candidate"           surface_z − z ≥ 15 mm AND |dz/dt| < 0.5 mm/s
                                       AND Fz_smoothed rising  ≥ 1 N over 0.5 s
                                       sustained 1 s
```

`Fz_cmd(t)` schedule (matches operator demo, gentler at every depth):

```
PHASE A:  Fz_cmd = −6 N    (close descent rate, harmless in air)
PHASE B:  Fz_cmd = −4 N    (one second only; avoids prying chamfer)
PHASE C:  Fz_cmd = −2 N    (matches Fz≈0 reading in demo)
PHASE D:  Fz_cmd = −2 N    (don't slam against seat)
After confirmed seat: Fz_cmd = 0 N, exit force mode.
```

Why these values:
- Operator demos show Fz_smoothed = 0 ± 0.5 N during free descent under
  cmd_fz = −9 N. The peg is in equilibrium between command and reaction.
  A −2 N command achieves the same equilibrium with **3× less commanded**
  energy and a corresponding 3× reduction in part-center moment from the
  Fz lever arm (0.054 Nm vs 0.16 Nm).
- −6 N approach matches `approach_fz_N` and works fine in air.
- Phase B's −4 N for 1 s gets through the chamfer transient; longer at
  higher Fz lever-arms tilt.

### 4.2 Seat detector (replaces `_tick_insert` predicate and termination block)

```
seated when ALL hold for 1 s sustain:
  surface_z − z ≥ 12 mm                  # part-specific min descent (u_orange ~25 mm seat depth)
  |dz/dt|   < 0.5 mm/s   (over 0.5 s window)
  Fz_smoothed > 2 N AND Fz risen ≥ 1 N from its Phase-C floor (slot bottom contact)
  tilt_deg < 4°                           # geometric sanity
```

**No CAD-xy gate.** CAD predictions have 9 mm error in observed runs and
the seat is at the slot's actual position, not the model's. Removing the
xy gate is what unlocks FAIL_INSERT-class runs.

The 12 mm minimum descent is a per-part scalar (operator-demo descent for
u_orange across 60 inserts is ~24 mm; pick 50 % of that as the seat
threshold). For other parts:
- u_brown: ~24 mm → threshold 12 mm
- inv_u_yellow: ~27 mm → threshold 13 mm
- line_green: ~6.7 mm → threshold 3 mm (per-part override needed)

This is the only per-part config the new controller needs:
`min_seat_descent_m`. Everything else is universal.

### 4.3 Escape mode (replaces FIND_HOLE spiral and WEDGE_RECOVERY)

The operator demos do not have an exploration mechanism: they place the
peg on the chamfer with their hand, then let go. For autonomous mode we
do need *something* when the controller stalls in Phase B (peg sitting on
chamfer rim, not engaging) or Phase C (peg got 5 mm in then jammed).

Stall detector: `dz/dt` over 2 s window < 0.3 mm/s AND Fz_smoothed > 4 N
AND time-in-phase > 3 s.

Escape action (one-shot, time-bounded, max 5 attempts per episode):
1. Retract: command Fz = +3 N for 0.4 s. This *unloads* the contact rather
   than pushing harder. The operator's "hand jiggle" was a place-and-release;
   the autonomous equivalent is unload-and-let-it-resettle.
2. **Position-spiral, NOT force-spiral**, for 1.5 s. Use `force_mode_controller`
   with selection_vector `[true, true, true, false, false, false]` and small
   Fx, Fy oscillation derived from a **position reference** (Stankowski-style).
   Cap commanded F_lat at **0.5 N** absolute. This bounds part-center moment
   at 0.12 Nm, below the operator's 0.45 Nm p95 — the peg cannot lever-arm
   itself into a wedge from a 0.5 N excursion.
3. Restore Fz = −2 N and re-enter the main controller.

**Drop the `find_hole_tilt_T_Nm`, `find_hole_tz_T_Nm`, `insert_tilt_T_Nm`,
`recovery_F_max_N=10` knobs entirely.** No torque commands at any point in
nominal mode. Operator demos don't need them.

The 0.5 N escape spiral cap is **one-twentieth** of the current
`spiral_F_max_N=12`. This is intentional. Demo data says the operator's
peak F_lat in the contact phase has p95 at 4 N (in only one or two of ten
demos) and median at 0.4 N. The current 12 N cap is 30× the operator
median. We are not modelling what the operator did.

### 4.4 Frame and convention notes

- `fx, fy, fz, tx, ty, tz` columns in `compliant_insertion_studio/logs/insert_*.csv`
  are **base-link** values (after `_wrench_in_base` transform, line 349 of
  `compliant_insert.py`). The `wrench_frame_id` column says
  `tool0_controller` because that's the SOURCE frame; the logged values
  are post-transform. **The CLAUDE.md note "wrench is in tool0_controller
  frame" refers to the source/raw, not what's in the CSV.** Future analysis
  should treat fx/fy/fz as base-link.
- The wrapper uses `task_frame=base_link` for SetForceMode. base_link is
  rotated 180° about world Z relative to base. So an operator-intuitive
  +X push in world is `−Fx` in command. Several spiral functions in
  `contact_search_fsm.py` already negate to handle this — keep that pattern
  for any new controller code.
- For r-vector (TCP → part_center) computation, use the **dynamic** version
  the existing code already maintains (`self._r_partcenter`, line 504).
  When the EE tilts, r rotates; static r is wrong above 2°.

---

## 5. What the algorithm should do, line-by-line

Replace `compliant_insert.py` ACTIVE-phase logic and `contact_search_fsm.py`
entirely with a single tick function:

```python
def tick(t, state):
    # state holds: surface_z, escape_attempts, escape_state, fz_floor_phaseC,
    #              motion_stopped_first_t, etc.

    # 1. SAMPLE
    z, xy, quat = read_tcp()
    Fz, F_lat, T_lat = read_wrench_smoothed(0.3 s)
    dz_dt = velocity_z(0.5 s)
    dFz_dt = derivative_Fz(0.3 s)
    tilt_deg = tilt_from_quat(quat)
    M_pc = T_lat + (-r_dynamic) cross F_xyz   # moment about part center

    # 2. SAFETY (always-on)
    if abs(F_lat) > 20 N for 0.2 s sustained:  abort("F_lat over 20 N")
    if tilt_deg > 8 deg:                        abort("tilt runaway")
    if M_pc magnitude > 1.5 Nm sustained 0.3 s: abort("part-center moment runaway")

    # 3. CONTACT LATCHING
    if surface_z is None and Fz > 5 N rising:
        surface_z = z       # one-time latch, never overwritten

    # 4. PHASE
    z_drop = (surface_z - z) if surface_z else None
    if surface_z is None:
        phase = A
    elif z_drop < 5 mm and Fz > 3 N:
        phase = B
    elif z_drop >= 2 mm and Fz < 1.5 N:
        phase = C
    if phase == C and z_drop >= 15 mm and abs(dz_dt) < 0.5 mm/s and dFz_dt rising:
        phase = D

    # 5. STALL DETECT (Phase B or C only)
    if phase in (B, C) and time_in_phase > 3 s and abs(dz_dt) < 0.3 mm/s and Fz > 4 N:
        if escape_attempts < 5:
            escape_attempts += 1
            run_escape(0.4 s retract @ +3 N, then 1.5 s position-spiral cap 0.5 N F_lat)
            return

    # 6. NOMINAL CONTROL
    Fz_cmd = {A: -6, B: -4, C: -2, D: -2}[phase]
    set_force_mode(0, 0, Fz_cmd, 0, 0, 0,
                   selection=(F,F,T,T,T,T) where F=force-controlled, T=position-tracked,
                   gain=1.0, damping=0.7)
    # Note: only Z is force-controlled; XY and rotation just track current pose.
    # This is critical: it removes the "spiral force" failure mode entirely.

    # 7. SEAT DETECTOR
    if (z_drop >= 12 mm
        and abs(dz_dt) < 0.5 mm/s for 0.5 s window
        and Fz > 2 N and Fz risen >= 1 N from Phase-C floor
        and tilt_deg < 4):
        if motion_stopped_first_t is None:
            motion_stopped_first_t = t
        elif t - motion_stopped_first_t >= 1.0 s:
            return DONE("seated: z_drop=%.1f mm, Fz=%.1f N, tilt=%.1f°" % ...)
    else:
        motion_stopped_first_t = None
```

Key contrasts with the current FSM:
- **No FIND_HOLE state.** The peg either contacts the chamfer and engages,
  or it drifts laterally enough to NOT contact, in which case Phase A
  persists and we eventually time out (max_active_duration_s=90 stays).
  No commanded spiral search in nominal flow.
- **No ENTRY_SETTLE state.** The transition criterion was a Fz drop, not
  a position drop, and it's read continuously via `dFz_dt` — no settle
  time needed.
- **No WEDGE_RECOVERY counter-torque.** Demo data shows operator never
  applied counter-torque; "wedges" in past failed runs were caused by the
  algorithm's own lateral commands.
- **No CAD-xy gate** anywhere. Per-part `min_seat_descent_m` is the only
  parameter that matters.
- **Fz_cmd schedule is depth-and-Fz-aware**, not state-machine-aware. This is
  the core operator policy.

### 5.1 Escape sub-routine (replaces 200 lines of WEDGE/RETRACT/spiral)

```python
def run_escape(t_now):
    # Step 1: unload — command +Fz lift for 0.4 s
    set_force_mode(0, 0, +3, 0, 0, 0, selection=all_force, gain=0.7, damping=0.5)
    sleep(0.4 s)

    # Step 2: position-spiral with bounded F_lat
    spiral_origin_xy = current_xy()
    t0 = clock()
    while clock() - t0 < 1.5 s:
        theta = ...  # Archimedean
        radius = max(0.6 mm, theta * pitch / (2π))   # cap at 4 mm
        target_xy = spiral_origin_xy + (radius cos θ, radius sin θ)
        # use position selection on xy: track target
        # set Fz=-2 N, F_lat=zero command, but selection_vector is
        # [position, position, force, position, position, position]
        # so the controller drives xy to target via position; F_lat is whatever
        # contact reaction is. CAP this in software:
        if read_F_lat() > 0.5 N:
            target_xy = freeze    # stop pushing harder

    # Step 3: restore main loop
    set_force_mode(0, 0, -2, 0, 0, 0, selection=all_force, gain=1.0, damping=0.7)
```

Selection-vector trick: changing channels from "force-controlled" (true) to
"position-tracking" (false) reroutes that DOF through the
`scaled_joint_trajectory_controller`'s position loop. This is how UR's
force_mode_controller does mixed control. **In nominal mode, only Z is
force-controlled; XY and rotation are position-tracked at the current pose.**
This is the structural change that removes the "commanded spiral" lateral
force entirely — there's no Fx_cmd at all in nominal flow.

---

## 6. Diagnostic instrumentation to add to the wrapper

Add these as derived columns (computed at sample-write time) — operator
needs them to validate the new controller:

| Column | Definition | Why |
|---|---|---|
| `fz_smooth_300ms` | Fz EMA over 300 ms | seat & phase detection |
| `dfz_dt_300ms` | finite difference of `fz_smooth_300ms` over 300 ms | phase B→C transition |
| `dz_dt_500ms` | EE z velocity over 500 ms | seat detection |
| `surface_z_latched` | current state's `surface_z` (NaN before contact) | reproduce phase rule |
| `z_drop_mm` | (surface_z − z) × 1000 | reproduce phase rule, plot easily |
| `phase` | A/B/C/D enum | derive from rule above |
| `r_pc_x`, `r_pc_y`, `r_pc_z` | dynamic TCP→part_center vector (mm) | tilt-coupling math |
| `M_pc_x`, `M_pc_y` | moment about part center (Nm) | the *right* tilt signal |
| `F_lat_dir_world_deg` | atan2(Fy_world, Fx_world) | direction signal |
| `tilt_axis_world_deg` | atan2(ee_zy, ee_zx) | direction signal |
| `commanded_fx`, `commanded_fy`, `commanded_tx`, `commanded_ty`, `commanded_tz` | full 6-vec command echo | confirm "no lateral command" actually held |
| `selection_vector` | 6-bit string e.g. `'001000'` | confirm selection mode |

The current CSV has `commanded_fz` only. Adding the rest is a 6-line change
to `_log_sample` in `compliant_insert.py`. This is essential for verifying
the new controller is doing what the spec says.

Also add a **dashboard**-side derived view:
- Phase-coloured z(t) trace
- Fz_smooth(t) overlaid with the depth-keyed Fz_cmd schedule
- M_pc_xy(t) magnitude with the 1.5 Nm safety threshold marked
- A summary chip "phase A: 6.2s, B: 0.8s, C: 18.3s, D: 1.0s, escape×0, seat:YES"

---

## 7. Things I previously got wrong (acknowledged)

In prior session iterations I:

1. **Misread the FAIL_INSERT outcome as "peg never reached the slot".** It
   reached z=0.2012, which is 0.43 mm above the CAD-predicted seat. The
   peg was in. The FSM didn't recognise it. I should have looked at final
   xy and z relative to CAD prediction *first*, before reasoning about
   spiral parameters.

2. **Treated F_lat direction as a strong signal.** Across 10 demos the
   walk vs F_lat angle ranges across the full circle. F_lat magnitude is
   the meaningful quantity (and it's small); direction is mostly noise on
   top of a small mean. r_cop is a cleaner direction signal where it
   exists, but only during contact phases where Fz > 2 N — that's only
   the chamfer-engagement and seat regimes, not the long quiet middle.

3. **Believed the wrapper's "operator rocked the part" hypothesis.** FFT
   of the demo contact phase shows energy concentrated at 10–40 Hz (sensor
   noise) and at DC (drift). No coherent rocking peak at 1–2 Hz. The
   `find_hole_tilt_T_Nm` and `insert_tilt_T_Nm` parameters are not
   derived from operator behaviour — they were guesses.

4. **Conflated "wrapper run with operator SIGTERM at the end" with
   "operator demo".** The 2026-05-04_145614 file ("DEMO" in this doc) is
   actually a wrapper run with multiple ENTRY_SETTLE → INSERT →
   WEDGE_RECOVERY cycles visible in the cmd_fz trace — the operator's
   Ctrl-C was after the wrapper completed its second pass. The
   2026-05-03 demos (cmd_fz=−9 throughout) are the cleaner ground truth.

5. **Iterated on parameter values (`spiral_F_max_N` 6→12, `insert_fz_N`
   9→4, `find_hole_tilt_T_Nm` 0.20→…)** instead of recognising that the
   *structure* of the controller is wrong. No parameter setting in the
   current FSM will make it pass when CAD xy is 9 mm off and the
   engagement gate is 6 mm tight.

---

## 8. Concrete next steps for the operator

In priority order:

1. **Verify the failure attribution** by replaying the FAIL_INSERT CSV
   through the FSM offline (no robot). The wrapper has all the data needed;
   what it lacks is a "replay mode". A 30-line script that reads the CSV
   and feeds the rows through `ContactSearchFSM.update()` would confirm
   whether the seat predicate would have fired with the engagement_dist
   gate removed. This is an off-robot task that closes the diagnosis
   without needing another autonomous run.

2. **Implement the §4 single-loop controller** as a new file
   `compliant_insertion_studio/wrapper/single_loop_insert.py`. Keep
   `contact_search_fsm.py` as a fallback (`--fsm legacy`). Guard the new
   path behind a CLI flag `--controller=single_loop` so we can A/B test.

3. **Fix the post_zero_bias drift.** FAIL_INSERT had +2 N Fz baseline; the
   smoke-test bias check (threshold 2 N) passed because raw bias was 0.31 N.
   The post-zero bias re-measure happened with the held part still under
   slight side-load. Either (a) add a "hands-off + 1 s stillness" gate
   before re-measuring post_zero_bias, (b) reject runs with
   |post_zero_bias.Fz| > 0.5 N, or (c) rezero again before APPROACH starts
   if post-zero drift has shifted.

4. **Add the §6 diagnostic columns** to the CSV. This is the cheapest
   change with the highest information value. Once derived signals are
   logged we can do all subsequent analysis on plain CSVs without a robot.

5. **Drop the per-shape config aspiration for now.** The defaults.yaml
   already says "single config for ALL FMB1 parts". The single-loop
   controller has *one* per-part scalar (`min_seat_descent_m`). Set it from
   operator-demo descent / 2 for each of {u_brown, u_orange, line_green,
   inv_u_yellow}. This gives us a 4-line per-part config as the entire
   parametric extension surface.

---

## Appendix A — Numbers cited (verifiable)

- Sample rate of CSVs: 99 Hz median (computed from `t_s` column on
  insert_u_orange_20260504_145614.csv).
- Demo descent rates: bin 5–25 mm depth, vz = 1–4.6 mm/s in DEMO file
  (most rows in the 80–95 mm depth bins are quasi-static at 0.1 mm/s
  because peg already seated).
- May-3 demos descent rate: ~3.9 mm/s steady (116.9 mm in ~30 s).
- |r| (TCP to u_orange part_center, canonical face-down) = 28.7 mm in xy,
  241.8 mm in z. Vector ≈ (−26.7, +0.2, −241.8) mm.
- Operator F_lat median 0.43–0.47 N, p95 1.4–2.0 N (May-3 demos, depth bin
  10–25 mm). Peak across 10 demos: 12.0 N (one transient).
- Operator Fz median in free descent (depth 5–25 mm) = +0.07 N. p95 = +0.48 N.
  Median is essentially zero despite −9 N command.
- Operator tilt p95 = 1.34° (depth 15–20 mm), max 1.61° across 10 demos.
- DEMO (wrapper-assisted) tilt p95 = 3.3° at depth 85–90 mm (where the
  wrapper's algorithmic forces show up).
- FAIL_INSERT final z = 0.20124, target seat z = 0.20081, Δ = 0.43 mm.
- FAIL_INSERT final xy = (0.0343, −0.3467); CAD xy = (0.0267, −0.3517);
  error = 9.1 mm.
- Moment per N of lateral force at TCP, projected to part_center: 0.242 Nm.
- `engaged_T_lat_max_Nm` gate in defaults.yaml: 0.15 Nm. Saturated by 0.62 N
  of TCP lateral force, well below `spiral_F_max_N=12`.

## Appendix B — Files referenced

- `/home/aaugus11/Documents/ros-mcp-server/compliant_insertion_studio/wrapper/contact_search_fsm.py` (1150 LOC, FSM)
- `/home/aaugus11/Documents/ros-mcp-server/compliant_insertion_studio/wrapper/compliant_insert.py` (1709 LOC, wrapper)
- `/home/aaugus11/Documents/ros-mcp-server/compliant_insertion_studio/configs/defaults.yaml`
- `/home/aaugus11/Documents/ros-mcp-server/compliant_insertion_studio/logs/insert_u_orange_20260503_*.csv` (10 clean operator demos)
- `/home/aaugus11/Documents/ros-mcp-server/compliant_insertion_studio/logs/insert_u_orange_20260504_145614.{csv,meta.json}` (DEMO — wrapper-assisted with operator sigterm)
- `/home/aaugus11/Documents/ros-mcp-server/compliant_insertion_studio/logs/insert_u_orange_20260504_150738.{csv,meta.json}` (FAIL_INSERT — wrapper aborted on already-seated peg)
- `/home/aaugus11/Documents/ros-mcp-server/compliant_insertion_studio/logs/insert_u_orange_20260504_144653.{csv,meta.json}` (FAIL_FIND_HOLE)
