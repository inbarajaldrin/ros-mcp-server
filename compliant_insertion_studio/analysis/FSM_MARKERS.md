# FSM markers catalog

The wrapper's FSM (`compliant_insertion_studio/wrapper/contact_search_fsm.py`) emits an event each time it transitions between states. Each transition has a **predicate** evaluated against current signals at every tick. The predicate fires → transition fires → the FSM emits a label that ends up in the meta JSON's `outcome` / `outcome_reason` / `transition_msg` fields.

**Hard rule:** these labels are FSM CLAIMS, not physical truth. They were tuned via guessed thresholds across iterations. Phys-truth requires re-derivation from raw CSV data. This document catalogs every marker, its current threshold(s), and what raw-data signal it should be tuned against.

---

## Table of markers

| # | Transition | Predicate (current) | Threshold params | Signal source | Tunable from |
|---|---|---|---|---|---|
| 1 | APPROACH→FIND_HOLE (first-contact) | `fz_smoothed > T1` for `T2` window | `contact_threshold_N=3.0`, `contact_sustain_s=0.1`, `fz_smooth_window_s=0.1` | sensed `fz` (tool frame) | GOLD: known seat → trace fz back to threshold crossing point. Compare fz at first sustained rise vs current threshold. |
| 2 | FIND_HOLE→ENTRY_SETTLE (chamfer-engaged candidate, z-drop) | `z_drop > T1` in time window | `find_hole_drop_thresh_m=0.0008` (0.8 mm) | TCP z descent post-contact | GOLD: locate the z-drop event when peg falls into chamfer; measure rate + magnitude. |
| 3 | FIND_HOLE→ENTRY_SETTLE (chamfer-engaged candidate, tilt-relaxation) | tilt rose above tolerance, then dropped from peak by ≥ `T1`, sustained `T2` | `find_hole_tilt_tolerance_deg=3.33`, `find_hole_tilt_relax_min_deg=0.40`, `find_hole_tilt_relax_sustain_s=0.30` | EE Z-axis vs world -Z | GOLD: peg-tilt curve; measure peak tilt at rim, tilt-relax magnitude when peg falls into chamfer. |
| 4 | ENTRY_SETTLE→INSERT (engagement verified) | F_lat low + T_lat low + v_xy static, sustained | `engaged_F_lat_max_N=3.0`, `engaged_T_lat_max_Nm_partcenter=0.50`, `engaged_v_xy_max_m_s=0.001`, `engaged_sustain_s=0.5` | F_lat = √(fx²+fy²); T_lat about part-center; TCP xy velocity | GOLD: signature of "peg in slot, settling": measure F_lat, T_lat, v_xy 0.5–2s after chamfer-engagement event |
| 5 | ENTRY_SETTLE→FIND_HOLE (timeout, engagement NOT verified) | `t_in_state > entry_settle_max_s` AND failed | `entry_settle_max_s=1.5` | timer | Pure algorithmic; tunes when verifier should give up. |
| 6 | INSERT→DONE (seated) | `descent_post_contact > T1` AND `tcp_z` near `predicted_seat_z` AND `motion_stopped` for `T2` | `insert_min_descent_m=0.005`, `insert_motion_thresh_m_s=0.0005`, `insert_motion_window_s=1.5` | TCP z + cad_prediction.predicted_tcp_at_seat.z; TCP velocity | GOLD: descent magnitude (~31 mm); time to settle; final TCP-z relative to predicted. |
| 7 | APPROACH→ABORT (descent timeout) | `t_in_state > approach_max_duration_s` | `approach_max_duration_s=30.0` | timer | Tune to `(hover_z - predicted_seat_z) / typical_descent_rate × 1.5` safety margin. |
| 8 | FIND_HOLE→ABORT (STUCK) | `z_drop < T1` after `T2` window, with optional proximity-based extended window | `find_hole_stuck_z_drop_m=0.001`, `find_hole_stuck_window_s=15.0` (extends to 30s if proximity to predicted xy < 6mm) | TCP z descent + xy-distance to predicted | Tune from GOLD: how long does GOLD spend in FIND_HOLE before progress? |
| 9 | INSERT→ABORT (insert timeout, never seated) | `t_in_state > insert_max_duration_s` | `insert_max_duration_s=30.0` | timer | Tune from GOLD seat-time distribution. |
| 10 | INSERT→ABORT (tilt blew up) | `tilt > T1` | `insert_tilt_abort_deg=5.0` | EE tilt | GOLD: max tilt during INSERT phase. |
| 11 | * → ABORT (lateral force overload) | `\|F_lat\| > T1` for `T2` window | `abort_F_lat_N=30.0`, `abort_F_lat_window_s=0.10` | sensed F_lat | Safety threshold; tune so it doesn't trigger on normal contact (~5–10 N) but does catch crashes (>30 N) |

Plus the WEDGE_RECOVERY sub-machine (markers 12+ that fire when peg is wedged on rim and need recovery — mostly algorithmic conveniences; not derivable from operator data because operator never wedges).

---

## What's actually a physical event vs an algorithmic convenience

### PHYSICAL events (extractable from raw operator-demo data)

These have a real signature in any successful insertion. The marker should fire when the signature is present, not based on a guessed threshold:

- **First contact** (#1): peg-bottom touches a surface → fz transient from baseline (~0 N) up to ≥ load magnitude. *Tunable from any GOLD demo: locate first sustained fz > N for ≥ X ms.*
- **Chamfer engagement** (#2 OR #3): peg breaks past rim into chamfer → simultaneous z-drop + tilt-relax + lateral-force-collapse. *Tunable from GOLD: find the moment z-velocity becomes positive after rim-only contact.*
- **In-slot** (#4): peg fully past chamfer, walls guide descent → F_lat → 0, T_lat → 0, v_xy → 0, dz/dt sustained > 0. *Tunable from GOLD: the post-chamfer steady descent regime.*
- **Seated** (#6): peg at slot bottom or stopped by feature → dz/dt = 0 sustained, tcp_z near predicted seat z. *Tunable from GOLD: terminal regime.*

### ALGORITHMIC CONVENIENCES (no operator-demo signature)

These are FSM internal book-keeping. They don't map to a physical event the operator's hand "fires":

- **APPROACH timeout** (#7), **FIND_HOLE STUCK** (#8), **INSERT timeout** (#9): pure timers. Tune from operator-demo *distribution* (e.g. p99 of GOLD time-in-state × 1.5 safety). Not a marker per se.
- **ENTRY_SETTLE timeout** (#5): pure verifier give-up. Operator never visits ENTRY_SETTLE (they go directly chamfer→seated). Algorithmic only.
- **WEDGE_RECOVERY sub-states**: operator never wedges; pure algorithmic recovery.
- **Tilt-blow-up abort** (#10), **F_lat overload** (#11): safety thresholds. Tune to NOT fire on normal contact.

---

## Reframe: what events SHOULD exist

The current FSM has **7 transitions** (markers 1–7 above, plus WEDGE_RECOVERY). The operator-demo data only generates **4 distinct regime transitions**:

```
APPROACH → on first contact → CONTACT_REGIME (= rim or chamfer)
CONTACT_REGIME → on z-drop+tilt-relax+F_lat-collapse → IN_SLOT
IN_SLOT → on dz/dt → 0 sustained at predicted seat z → SEATED
[Anywhere] → on F_lat overload OR tilt blow-up → ABORT
```

Mapping current FSM states:
- APPROACH = APPROACH ✓
- FIND_HOLE + ENTRY_SETTLE = CONTACT_REGIME (these are split but operator-data shows they're one regime: peg in contact, working its way to chamfer/slot)
- INSERT = IN_SLOT ✓
- DONE = SEATED ✓

**Recommendation post data-collection:** consolidate FIND_HOLE + ENTRY_SETTLE. The current split exists because the FSM tries to *verify* engagement before committing, but operator data shows engagement is unambiguous from raw signal (z-drop > X mm + tilt-relax > Y° + F_lat-collapse < Z N — all simultaneous). One marker, one transition.

---

## Tuning plan (after data collection)

For each marker `M_i` in markers #1–4 + #6:

1. **Signature derivation** — segment the GOLD trajectory using `analysis/scripts/30_segment_regimes.py`. Locate the sample where regime changes (e.g. RIM → IN_SLOT_DESCENT). Read the raw signals at that sample.
2. **Threshold extraction** — across all 15+ GOLD demos, compute the distribution of the signal-at-transition (median, p5, p95). Pick threshold = median - 1σ (sensitive enough to catch all real events) or median (balanced).
3. **Check** — re-run segmenter with the new threshold; verify it still identifies the same transition moment within ±100 ms.
4. **Update `defaults.yaml`** with the data-derived threshold + a comment recording the GOLD demos it was derived from.

Markers #5, #7, #8, #9 (timers): compute `p99 × 1.5` from GOLD time-in-state distribution.

Markers #10, #11 (safety): compute `max + 5×stddev` from GOLD; set threshold above that floor with operator confirmation.

---

## Single-demo tuning preview (what we have today)

From the one v1.2 GOLD demo (`insert_u_orange_20260504_113809`), we already saw:

| Marker | GOLD signature observed | Current threshold | Status |
|---|---|---|---|
| #1 first-contact | fz crosses 1 N at t = 19.66s, 6 N at 19.97s | `3 N + 0.1s` (already tuned 2026-05-06) | ✓ matches single GOLD point |
| #6 seated descent magnitude | 31.5 mm post-contact | `5 mm` (insert_min_descent_m) | ✗ way too loose (should be 25 mm) |
| #6 seated TCP-xy proximity | xy drift contact→end = -5.7, -7.6 mm (10 mm total) | not currently a marker | should add: xy-near-predicted-seat |
| #8 FIND_HOLE STUCK timeout | GOLD spent 15.4 s in RIM regime | `15 s` window | ✓ matches but need distribution |

The seat-descent threshold `insert_min_descent_m=5mm` is the marker that let the false-seat slip through earlier today: peg "descended" 63 mm through air (no real contact) and the FSM was happy because that's > 5 mm. After data collection, this should become `25 mm` minimum (with a strict upper bound of `45 mm` to catch peg-fell-through cases).

---

## Files to update post data-collection

| File | What changes |
|---|---|
| `compliant_insertion_studio/configs/defaults.yaml` | Per-marker threshold overrides with GOLD-derivation comment |
| `compliant_insertion_studio/wrapper/contact_search_fsm.py` | Possibly consolidate FIND_HOLE + ENTRY_SETTLE per recommendation above |
| `compliant_insertion_studio/scripts/collect_regime_data.py` | Already has `_verify_physical_seat()` with descent 25–45 mm + xy drift <15 mm. Add per-marker verification too (e.g. record marker-event timestamps from raw, compare to FSM-claimed timestamps from meta) |
| `compliant_insertion_studio/analysis/REGIME_DECODING.md` | Cross-reference this catalog from the regime hypotheses |
