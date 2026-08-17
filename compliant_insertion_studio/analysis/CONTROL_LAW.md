# Found Hole Control Law — Verified Predicate v4

**Status:** validated 10/10 across 3 directions (A_pos_x, C_pos_y, D_neg_y), robust across 94/96 threshold combinations, 2026-05-06.

**Scope:** decode "peg is above the hole" from local F/T signals during the GUIDED phase of u_orange / base1 / grasp_id=1 insertion. Output: an autonomous, direction-invariant predicate that replaces the operator's manual SIGUSR1 trigger.

---

## TL;DR

The Found Hole event is a **rim-contact state transition**, detectable from `|fz_smooth|` alone:

```
ON_RIM   ⇔ |fz_smooth| > 4 N
OFF_RIM  ⇔ |fz_smooth| < 3 N         (hysteresis)
FIRE when:
  - state has been OFF_RIM continuously for ≥ 0.3 s, AND
  - state was ON_RIM at any point in the previous 2.5 s
```

That's it. No tilt, no F_lat, no r_cop, no drag-direction analysis, no spatial integration.

At fire time on 10/10 clean demos:
- Peg's xy is **0.30–3.36 mm from the eventual hole** (max-5 mm gate passed by every demo)
- Lead vs operator's SIGUSR1: **+2.05 to +5.74 s** (mean 3.4 s)
- Variation breakdown: A_pos_x 3/3, C_pos_y 3/3, D_neg_y 4/4

The autonomous controller can therefore trigger INSERT_DESCENT ~3 s earlier than a human operator would, with the peg already in the hole region.

---

## What "above the hole" means in this dataset

The operator's mental model and the physics differ. There are **two distinct events** in every GUIDED segment:

1. **Rim-cross / chamfer-encounter** (the local geometric event):
   - Peg slides off the rim, contact area collapses, vertical load `|fz|` falls from 13–22 N to <3 N within ~250 ms.
   - This is the moment the peg first reaches the hole region.

2. **Operator's SIGUSR1** (the manual mark):
   - 2–6 s LATER. The operator continues to fine-tune the peg's xy position over the chamfer before triggering autonomous descent.
   - Median: peg at SIGUSR1 is dead-center over the hole (`dist_to_hole = 0.0 mm`), but it was already within ~3 mm at the rim-cross.

The user's request — *"detect when the object is above the actual hole"* — corresponds to event #1, not event #2. SIGUSR1 was operator caution / extra centering, not the moment the peg arrived.

The predicate detects event #1.

---

## How the predicate was derived (data-driven, not guessed)

### Dataset

10 clean GUIDED demos (`outcome=success`, `csv_final=DONE`, `source=fsm_guided_sigusr1`):

| Variation | n | Demos |
|---|---|---|
| A_pos_x_10mm | 3 | 043426, 043529, 043633 |
| C_pos_y_10mm | 3 | 040446, 040628, 042120 |
| D_neg_y_10mm | 4 | 042232, 043107, 043221, 043324 |

(8 additional demos were discarded — `csv_final=ABORT` with `source=fsm_guided_sigusr1_backfill_from_csv`, indicating mid-record cancellation.)

### Step 1 — exhaustive tool-frame feature catalog

For each demo, extract the GUIDED segment `[t_contact, t_sigusr1]` and the INSERT_DESCENT segment `[t_sigusr1, t_end]`. Computed at every 10 ms tick:

| Signal | Meaning | AUC at SIGUSR1 vs rolling baseline |
|---|---|---|
| `tilt_deg` (EE-Z vs world-Z magnitude) | Peg orientation deviation | **0.51** (random) |
| `F_lat_tool = √(fx²+fy²)` | Lateral wrench in tool frame | **0.57** (low signal) |
| `\|r_cop\| = \|(-Ty, Tx)/Fz\|` | Center-of-pressure radius | artifact (Fz≈0 → divergence) |
| `fz_smooth` (instantaneous) | Normal load | **0.74** (lower-is-event) |
| `vz` (instantaneous) | Vertical velocity | **0.90** (lower-is-event) |
| `vz_rise_from_min_1s` | vz at its 1-s window minimum | **0.92** |
| `fz_drop_from_peak_1s` | fz drop magnitude over 1 s | **0.82** |

Tilt is a non-signal in this dataset: the peg stays face-down to within 0.01° throughout. **Predicates that depend on tilt-relax (the original hypothesis) cannot be built from this data.**

> **CORRECTED 2026-08-16.** The original wording of this paragraph attributed the 0.01° figure to
> "the FSM's full 6-DOF compliance." Both halves of that were wrong.
>
> 1. **The FSM is not 6-DOF compliant.** Every `selection_vector` in `contact_search_fsm.py` is
>    `(True, True, True, False, False, False)` — translation compliant, **rotation LOCKED** —
>    and has been since 2026-05-06.
> 2. **The 0.01° is therefore a consequence of that clamp**, not a measurement of contact
>    physics. It says nothing about whether tilt *would* carry signal under rotational
>    compliance, and must not be cited as evidence that tilt steering is impossible.
>
> Unlocking rotation on 2026-08-16 did make tilt responsive (excursion 0.004° → 0.25°, peak
> 0.59°) — still far short of the ~3° GOLD shows at chamfer engagement, and it did not change the
> outcome. It *did* break the insert for five consecutive real-arm runs, because lateral force
> then applies a moment about the grasp point: the part pivots in the jaws while the gripper
> translates, so TCP displacement stops being peg displacement. **Rotation stays locked in
> SEARCH.** See `SKILL.md` §10 and `docs/HANDOFF_NEXT_AGENT.md` anti-patterns.

### Step 2 — first attempt, instantaneous-threshold predicate (failed)

```
v1: vz ≤ -0.3 mm/s AND F_lat ≤ 2 N AND |fz| ≤ 1 N, sustained 100 ms
```

Result: **0/10 fired within ±300 ms of SIGUSR1**, 5/10 fired 2–3 s early. Many false fires (16–79 firings/demo).

This led to the realization that:
- The "early fires" were not false — they were correctly detecting the rim-cross.
- The acceptance criterion (fire within ±300 ms of SIGUSR1) was the wrong target.

### Step 3 — reframed target, transient detector (partial)

```
v3: fz dropped ≥ 5 N in 300 ms AND F_lat ≤ 2 N AND |fz_post| ≤ 2 N, sustained 100 ms
```

Result: **2/10 fired**. The simultaneity requirement was too strict — F_lat lags fz collapse by ~250 ms because the operator is still actively dragging the peg as it crosses off the rim.

### Step 4 — state-transition predicate (final)

Drop F_lat from the predicate entirely; rim-cross is purely an `|fz|` event:

```
v4: ON_RIM (|fz|>4) → OFF_RIM (|fz|<3), 0.3 s sustain, recent ON_RIM in 2.5 s
```

Result: **10/10 fire**, peg within 5 mm of hole at every fire (0.30–3.36 mm), 2.05–5.74 s lead vs SIGUSR1.

### Step 5 — robustness sweep

96 threshold combinations spanning `rim_high ∈ {3,4,5,6}`, `rim_low ∈ {2,3,4}`, `off_sustain ∈ {0.2, 0.3, 0.5, 0.75} s`, `recent_window ∈ {1.5, 2.0, 2.5, 3.0} s`. **94/96 pass 10/10 on both gates.** The two failures are obvious edge cases (too-tight recent window + too-long sustain). The signature is a wide plateau in parameter space, not a tuned point.

---

## Direction invariance — argument

The predicate uses only `|fz_smooth|`, the magnitude of the tool-frame Z component. By construction:

1. `fz` is in `tool0_controller` frame (verified in CSV `wrench_frame_id` column).
2. Peg axis is parallel to tool Z (peg held axially in the gripper).
3. `|fz|` is the normal load on the peg, which depends only on the local geometric contact (peg pressing into rim or floating off it), not on the world-frame direction of the rim approach.
4. Therefore the predicate is identical for any approach direction — confirmed empirically across +X, +Y, −Y variations (10/10).

The predicate does **not** observe `fx`, `fy`, `Tx`, `Ty`, `tcp_x`, `tcp_y`, `vx`, `vy`, or any quaternion component. There is no surface for direction-bias to enter.

The unobserved variations (B_neg_x, E_diag) are predicted to behave identically by this argument. The argument is testable — collecting 3 B_neg_x demos and 3 E_diag demos and re-running this analysis would falsify the prediction if the predicate fails on them.

---

## Per-demo results (predicate v4, default thresholds)

```
demo            var    sigusr1_t   fire_t  lead_s  |fz|@fire  Flat@fire   dist→hole   G1   G2
20260506_040446 C        34.81     32.18   +2.63       2.90       2.38       2.71mm  YES  YES
20260506_040628 C        34.93     31.60   +3.33       2.31       5.68       2.39mm  YES  YES
20260506_042120 C        38.14     32.56   +5.57       2.09       4.67       2.68mm  YES  YES
20260506_042232 D        36.49     30.75   +5.74       2.67       5.48       0.35mm  YES  YES
20260506_043107 D        38.67     35.44   +3.23       2.72       3.97       1.36mm  YES  YES
20260506_043221 D        32.92     30.56   +2.36       2.82       5.32       1.60mm  YES  YES
20260506_043324 D        33.09     31.04   +2.05       2.51       3.71       0.30mm  YES  YES
20260506_043426 A        34.68     31.76   +2.92       2.45       9.50       1.86mm  YES  YES
20260506_043529 A        35.54     33.08   +2.46       2.52       3.00       0.37mm  YES  YES
20260506_043633 A        35.62     32.23   +3.39       2.15       5.15       3.36mm  YES  YES

G1 (fires before SIGUSR1):           10/10
G2 (peg within 5mm of hole at fire): 10/10
```

`F_lat_at_fire` ranges 2.4–9.5 N (highly variable) — confirms F_lat is **not** a usable signal at the rim-cross moment. The operator is still actively dragging when the peg crosses off the rim; F_lat collapses 200–500 ms later.

---

## What this predicate does NOT do

This is the marker only. It does not:

- **Direct the peg toward the hole.** That's the autonomous director's job, supplied by the CAD-predicted slot xy refined by base calibration (Phase 2). The operator's drag in our data was labeling, not training data for a search policy.
- **Confirm the peg has fully seated.** That's the At Target marker (`|tcp_z − predicted_tcp_z| < 5 mm` + motion-stopped, sustained 1 s) — already finalized in `FSM_MARKERS.md`.
- **Identify which side of the rim the peg crossed from.** Direction-invariant by design — the predicate sees only normal-load magnitude, not direction.

---

## FSM integration sketch

In `compliant_insertion_studio/wrapper/contact_search_fsm.py`:

```python
# Replace the GUIDED state's reliance on operator SIGUSR1 with this predicate.
# Runs in tick(), real-time-safe, no history beyond a 2.5 s rolling buffer.

class FoundHoleDetector:
    def __init__(self, dt_s, rim_high=4.0, rim_low=3.0,
                 off_sustain_s=0.30, recent_window_s=2.5):
        self.dt = dt_s
        self.rim_high = rim_high
        self.rim_low = rim_low
        self.n_off_sustain = max(1, int(round(off_sustain_s / dt_s)))
        self.n_recent = max(1, int(round(recent_window_s / dt_s)))
        self._on_rim_history = collections.deque(maxlen=self.n_recent)
        self._off_run = 0
        self._fired = False

    def update(self, fz_smoothed):
        abs_fz = abs(fz_smoothed)
        on_rim_now = abs_fz > self.rim_high
        off_rim_now = abs_fz < self.rim_low
        self._on_rim_history.append(on_rim_now)
        if off_rim_now:
            self._off_run += 1
        else:
            self._off_run = 0
        if (not self._fired
            and self._off_run >= self.n_off_sustain
            and any(self._on_rim_history)):
            self._fired = True
            return True
        return False
```

Place in the GUIDED state's tick loop (or in FIND_HOLE if eliminating GUIDED entirely for autonomous mode). When `update()` returns True, transition GUIDED → INSERT_DESCENT (capture current `tcp_xy` as `hole_observed`) — exactly the same downstream behavior as the operator's SIGUSR1.

---

## Open questions / future work

1. **Will the predicate generalize to the unobserved B_neg_x / E_diag variations?** Argued yes by direction-invariance. Falsifiable by 3 demos in each, 30 min of collection.
2. **Will it generalize to other FMB1 parts** (u_brown, line_green, inverted_u_yellow)? The threshold magnitudes (4 N / 3 N) depend on the operator's drag pressure during search and on the part-specific rim height vs chamfer depth. Re-validate per part with 3–5 GUIDED demos each. The structure of the predicate (state-transition on |fz|) should hold; the thresholds may need per-part scaling.
3. **At the rim-cross fire moment, peg-xy is up to 3.36 mm from the eventual hole center.** INSERT_DESCENT must tolerate 3–5 mm xy error at descent start. Empirically (chamfer width ≈ ±5 mm for u_orange) this is well within the chamfer's self-aligning region. If a part has a tighter chamfer, the predicate may need a "settle" delay before triggering descent (e.g., wait for peg-xy velocity to drop below 1 mm/s after rim-cross, allowing the chamfer to pull peg toward center).
4. **Re-armable predicate.** v4 fires once per scan. If the operator wobbles the peg back onto the rim mid-search, the current implementation needs a `reset()` between attempts. The full FSM integration should handle this by re-arming on `state_change` events.

---

## Files referenced

- `analysis/scripts/31_decode_operator_action.py` — feature extractor (Phase B)
- `analysis/scripts/32_found_hole_signature.py` — AUC ranking (Phase C)
- `analysis/scripts/34_validate_law.py` — first-attempt instantaneous predicate (failed; kept for record)
- `analysis/scripts/35_validate_rim_cross.py` — transient predicate (partial; kept for record)
- `analysis/scripts/36_validate_rim_offrim_state.py` — state-transition predicate v4 (final, validated)
- `analysis/data/guided_features/*.features.json` — per-demo extracted features
- `analysis/data/found_hole_signature.json` — AUC analysis output
- `analysis/data/found_hole_v4_validation.json` — final validation scorecard
