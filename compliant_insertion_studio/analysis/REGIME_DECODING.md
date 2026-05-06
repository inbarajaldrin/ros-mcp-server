# Regime-Conditional Control Law Decoding

**Goal.** Decode the operator's control law from GOLD demo telemetry as a function:

```
operator_action(sensor_state) → (commanded_xy_direction, commanded_fz, when_to_change_action)
```

The law is conditional on *contact regime*. The operator does fundamentally different things when the peg is sitting on the rim vs. engaging the chamfer vs. descending in the slot. Without regime separation, aggregate stats over the whole contact span average across heterogeneous behaviors and tell us nothing actionable.

This document defines the regimes, the detectors, the per-regime action decoders, the transition-trigger decoders, and what additional data is needed to validate the result.

---

## 1. Regime hypotheses (physics-grounded, not data-derived)

The contact geometry of a prismatic peg over a slotted base produces four physically distinct regimes. Each has a distinct expected wrench + motion signature derivable from rigid-body contact mechanics — *not* from staring at the GOLD CSV.

### 1.1 RIM regime

**Geometry.** Peg-bottom resting on the flat top of the base, displaced laterally from the slot opening by some xy offset. Contact is a small surface patch (gripper-flat-bottom on base-flat-top).

**Expected signature:**
| Signal | Expected value |
|---|---|
| `dz/dt` | ≈ 0 (peg cannot descend, surface stops it) |
| `Fz_sensed` | ≈ commanded Fz, modest (operator pushes gently to slide rather than dig in) |
| `T_lat` (Tx, Ty) | small magnitude, roughly proportional to the lever arm of the peg-bottom contact patch from the TCP |
| `tilt_deg` | small (peg sits flat on flat surface) |
| `F_lat` reaction | exists if operator pushes laterally — magnitude ≤ μ × Fz (Coulomb friction limit before peg slides) |
| `dx/dt, dy/dt` | nonzero in operator's chosen direction |

**Operator's strategy in this regime:** push the peg laterally toward the slot opening with sustained directional force, low-magnitude commanded Fz so peg can slide rather than wedge.

### 1.2 EDGE_OF_SLOT regime

**Geometry.** Peg-bottom is partially over the slot opening — one edge of the peg-bottom is past the rim edge, the rest is still on the rim. Contact is a *line* (peg-bottom edge across the rim edge).

**Expected signature:**
| Signal | Expected value |
|---|---|
| `dz/dt` | small but nonzero (peg starts to tip into slot as it shifts) |
| `Fz_sensed` | comparable to RIM, may briefly drop as peg-bottom partially loses contact |
| `T_lat` | rises sharply — line contact at rim edge has a moment arm in one direction relative to peg-axis-through-TCP |
| `tilt_deg` | rises (peg leans toward slot side) |
| `F_lat` reaction | rim edge resists motion in the direction perpendicular to the rim edge |
| `dx/dt, dy/dt` | direction is set by where the operator is pushing AND by the direction of the rim edge |

**Operator's strategy:** continue pushing in the direction that gets the peg over the slot opening. As the peg starts to tip in (tilt rises), the operator may *reduce* lateral force and increase Fz to let gravity + Fz drive the peg through the chamfer.

### 1.3 CHAMFER_TRANSIT regime

**Geometry.** Peg-bottom is below the rim level, riding down the chamfer (the angled lead-in into the slot). Contact is the chamfer-side surface against the peg's bottom edge.

**Expected signature:**
| Signal | Expected value |
|---|---|
| `dz/dt` | clearly positive (peg descending) |
| `Fz_sensed` | rises briefly (chamfer pushes back as peg rotates around chamfer edge) then drops as peg becomes vertical in slot |
| `T_lat` | high, then rapidly drops (tilt is being undone as peg aligns with slot axis) |
| `tilt_deg` | peaks then drops back to baseline |
| `F_lat` reaction | from chamfer normal force (perpendicular to chamfer surface) |
| `dx/dt, dy/dt` | small — peg motion is dominated by descent |

**Operator's strategy:** stop pushing laterally (or release force entirely), let geometry guide peg through chamfer. Optionally apply Fz to ensure descent through chamfer friction.

### 1.4 IN_SLOT_DESCENT regime

**Geometry.** Peg fully inside the slot, walls of slot guide descent. Slot clearance is small (sub-mm typically); peg slides down with wall-contact possible on any side.

**Expected signature:**
| Signal | Expected value |
|---|---|
| `dz/dt` | sustained positive (peg falls / is pushed in) |
| `Fz_sensed` | low (matches commanded Fz; no rim resistance) |
| `T_lat` | low (slot walls can produce mild moments, but no tilt-inducing geometry) |
| `tilt_deg` | low and steady (back to baseline) |
| `F_lat` reaction | only if peg contacts wall — typically transient |
| `dx/dt, dy/dt` | ≈ 0 (slot constrains lateral motion) |

**Operator's strategy:** maintain Fz, let descent proceed.

### 1.5 SEATED regime (terminal)

**Geometry.** Peg fully seated, base of peg at slot bottom or stopped by feature.

**Expected signature:**
| Signal | Expected value |
|---|---|
| `dz/dt` | 0 (sustained) |
| `Fz_sensed` | matches commanded Fz, no descent → equilibrium |
| `T_lat` | low |
| `tilt_deg` | low |
| `F_lat` reaction | low |
| `dx/dt, dy/dt` | 0 |
| `tcp_z` | matches `predicted_tcp_at_seat.z` ± few mm |

**Operator's strategy:** stop applying force; insertion complete.

---

## 2. Regime detectors

Each detector is a function `(window of raw signals) → bool` indicating whether the peg is currently in that regime. Detectors are designed to be **mutually exclusive at any instant** (peg is in exactly one regime), and the segmentation is the timeline of regime memberships.

### 2.1 Detector definitions

All detectors use windowed signals (typical window: 0.5s @ 100Hz = 50 samples) and require sustained satisfaction (e.g., 0.3s of continuous truth) to debounce contact bounce.

```python
# Pseudocode; each detector returns bool for an instant given a centered window

def is_RIM(w):
    return (
        abs(w.dz_dt_median) < 0.5e-3                     # < 0.5 mm/s
        and 3.0 < abs(w.fz_median) < 8.0                  # gentle contact
        and w.tilt_deg_median < 1.5                       # peg flat on flat
        and abs(w.T_lat_median) < 0.3                     # small moment
    )

def is_EDGE_OF_SLOT(w):
    return (
        0.5e-3 <= abs(w.dz_dt_median) < 3.0e-3            # small descent starting
        and abs(w.fz_median) > 3.0
        and (w.tilt_deg_median > 1.5 or abs(w.T_lat_median) > 0.4)  # peg starting to tip
        and not is_CHAMFER_TRANSIT(w)
    )

def is_CHAMFER_TRANSIT(w):
    return (
        abs(w.dz_dt_median) >= 3.0e-3                      # > 3 mm/s
        and (w.tilt_deg_max - w.tilt_deg_min) > 1.0        # tilt is changing (rising or falling)
        and abs(w.fz_median) > 2.0                          # still in contact with something
    )

def is_IN_SLOT_DESCENT(w):
    return (
        abs(w.dz_dt_median) >= 1.0e-3
        and w.tilt_deg_median < 1.0
        and abs(w.T_lat_median) < 0.2
        and abs(w.fz_median) < 5.0                         # walls don't resist as hard as rim
        and (w.tilt_deg_max - w.tilt_deg_min) < 0.5        # tilt steady, not transient
    )

def is_SEATED(w):
    return (
        abs(w.dz_dt_median) < 0.3e-3                       # ≤ 0.3 mm/s
        and (w.tcp_z_median - w.predicted_seat_z) < 5e-3   # within 5mm of CAD seat
        and w.tilt_deg_median < 1.0
        and (w.duration_in_state >= 1.0)                   # 1s sustained
    )
```

**Threshold provenance.** All thresholds above are *initial estimates* informed by physics intuition — NOT derived from data yet. The validation step (Section 6) tunes these against GOLD data to maximize segmentation consistency. Each threshold should be data-derived before the law is locked in.

### 2.2 Segmentation algorithm

```python
def segment(csv_path) -> list[(t_start, t_end, regime)]:
    rows = load_active_phase(csv_path)
    timeline = []
    current_regime = None
    current_start = None
    for t in rows.timestamps:
        w = window_around(rows, t, half_width=0.25)  # 0.5s centered
        # Test in priority order — most specific first
        if is_SEATED(w):           regime = 'SEATED'
        elif is_CHAMFER_TRANSIT(w): regime = 'CHAMFER_TRANSIT'
        elif is_EDGE_OF_SLOT(w):    regime = 'EDGE_OF_SLOT'
        elif is_IN_SLOT_DESCENT(w): regime = 'IN_SLOT_DESCENT'
        elif is_RIM(w):             regime = 'RIM'
        else:                        regime = 'UNKNOWN'
        # Debounce: require 0.3s sustained to commit transition
        if regime != current_regime:
            if has_been_stable_for(rows, t, regime, 0.3):
                if current_regime is not None:
                    timeline.append((current_start, t, current_regime))
                current_regime = regime
                current_start = t
    if current_regime is not None:
        timeline.append((current_start, rows.timestamps[-1], current_regime))
    return timeline
```

---

## 3. Per-regime operator direction decoder

For each segment in the timeline, derive the operator's action:

### 3.1 Lateral direction (xy)

Within a regime segment, fit a unit vector to TCP xy motion:
```python
def operator_direction_xy(segment_rows):
    dx = segment_rows.tcp_x[-1] - segment_rows.tcp_x[0]
    dy = segment_rows.tcp_y[-1] - segment_rows.tcp_y[0]
    travel_mm = sqrt(dx**2 + dy**2) * 1000
    if travel_mm < 0.5:
        return None  # operator was not pushing laterally
    return (dx / travel_mm, dy / travel_mm), travel_mm  # unit + magnitude
```

### 3.2 Z command

Sensed Fz approximates commanded Fz under contact at low velocity. For RIM regime where peg is stationary in z, sensed Fz ≈ commanded Fz. For other regimes, infer commanded Fz from the cmd_wrench_raw sidecar if available.
```python
def operator_fz(segment_rows):
    return median(segment_rows.fz)
```

### 3.3 Direction commitment time

How long does the operator hold the same direction before changing?
```python
def commitment_duration_s(segment_rows):
    return segment_rows.t[-1] - segment_rows.t[0]
```

### 3.4 Output per-regime profile

```python
{
    'regime': 'RIM',
    'direction_xy': (-0.6, -0.8),       # unit vector toward slot
    'lateral_travel_mm': 9.4,            # distance moved
    'commanded_fz_N': 4.7,               # operator's Fz
    'commitment_s': 5.1,                 # time held this direction
}
```

---

## 4. Regime-transition trigger decoder

The interesting thing is *what sensor signature was present* at the moment the operator decided to change regime.

For each transition `regime_A → regime_B` in the timeline:
1. Capture sensor window (0.5s before through 0.5s after the transition timestamp).
2. Identify which signals crossed which thresholds in that window.
3. Output the trigger as a predicate.

```python
def decode_transition_trigger(rows, t_transition, regime_before, regime_after):
    pre = window(rows, t_transition - 0.5, t_transition)
    post = window(rows, t_transition, t_transition + 0.5)
    triggers = []
    for sig in ['fz', 'T_lat', 'tilt_deg', 'dz_dt', 'F_lat']:
        if pre.signal[sig].max() < threshold[sig] <= post.signal[sig].max():
            triggers.append(f"{sig} crossed {threshold[sig]} upward")
        elif pre.signal[sig].min() > threshold[sig] >= post.signal[sig].min():
            triggers.append(f"{sig} crossed {threshold[sig]} downward")
    return triggers
```

The trigger predicate becomes the regime-transition rule in the synthesized law.

---

## 5. Synthesized control law template

Produced by combining outputs of sections 2-4:

```yaml
control_law:
  initial_regime: RIM   # entered at first contact

  RIM:
    detector: |
      |dz/dt| < 0.5mm/s AND fz in [3,8]N AND tilt < 1.5° AND |T_lat| < 0.3Nm
    action:
      direction_xy_mode: COMMITTED   # hold same direction throughout regime
      direction_xy_source: predicted_seat_xy_minus_tcp_xy_unit_vector
      F_lat_magnitude_N: <derived from operator data, e.g. 1.0>
      Fz_N: <derived, e.g. 4.5>
    transitions:
      to_EDGE_OF_SLOT:
        when: tilt > 1.5° OR |T_lat| > 0.4Nm OR dz/dt > 0.5mm/s
      to_RIM_SEARCH_RESET:   # if no progress
        when: lateral_travel < 0.2mm/s sustained for 5s

  EDGE_OF_SLOT:
    detector: <as Section 2.1>
    action:
      direction_xy_mode: HOLD   # do not change xy direction further
      F_lat_magnitude_N: <reduced — derived>
      Fz_N: <may be elevated to push through chamfer>
    transitions:
      to_CHAMFER_TRANSIT: when: dz/dt > 3mm/s
      to_RIM: when: tilt drops AND no descent

  CHAMFER_TRANSIT:
    detector: <as Section 2.1>
    action:
      direction_xy_mode: RELEASE   # zero lateral force
      Fz_N: <maintain>
    transitions:
      to_IN_SLOT_DESCENT: when: tilt < 1° AND |T_lat| < 0.2Nm sustained 0.3s

  IN_SLOT_DESCENT:
    detector: <as Section 2.1>
    action:
      direction_xy_mode: NEUTRAL
      Fz_N: <maintain>
    transitions:
      to_SEATED: when: dz/dt = 0 sustained AND tcp_z near predicted

  SEATED:
    detector: <as Section 2.1>
    action: TERMINATE
```

The values `<derived>` are filled in by running the analysis pipeline on GOLD data.

---

## 6. Validation against GOLD

Once the law is filled in, validate it against the GOLD trajectory it was derived from (and other GOLDs):

1. Run the segmenter on GOLD CSV → timeline.
2. Run the per-regime direction decoder → action profile.
3. Run the transition trigger decoder → trigger predicates.
4. **Self-consistency check:** does the synthesized law, if executed, reproduce the GOLD timeline? (Forward-simulate the law on the GOLD initial conditions and check whether it produces the same regime sequence.) If not, the law is incomplete.

5. **Threshold robustness:** sweep each detector threshold ±20% and verify segmentation is stable. If small threshold change flips segments, the threshold is data-fragile.

---

## 7. Required data collection (the spec for what GOLDs are needed)

Single-trajectory derivation overfits. To derive a robust law, collect more GOLDs spanning the regime variation space. **Minimum complete dataset:**

### 7.1 Schema requirement

Every collected demo must be **schema v1.2** (5-file sidecar bundle). Don't collect more v1.1 demos — they don't have native-rate wrench, joints, or cmd_wrench data needed for tight regime detection.

### 7.2 Variation axes — what to vary and why

**Axis that matters most: starting peg position relative to slot (which side, which quadrant).** This exercises the operator's *direction-of-correction* selection. Without it, a single-side dataset would let the decoder hard-code "operator always pushes -X" because every demo did. We need samples from all sides to confirm that the per-regime direction rule is `unit(slot_xy − peg_xy)` (or whatever it actually is).

**Axis that matters less: grasp variation.** Don't deliberately introduce grasp offsets — natural perception variation produces enough already. The held_quat after rotate_object normalizes most of it via fold-symmetry-snap, so the peg orientation is canonical regardless. Adding grasp variation just adds noise without exercising new operator behavior.

**Axis that matters: RIM-regime length (slide distance).** Short (~5mm) vs long (~15mm) slides reveal whether operator's commitment duration scales with distance — informs the FSM's "when to release lateral force" trigger.

### 7.3 Initial-condition variation (per object)

For u_orange, base1, grasp_id=1, collect demos at varied initial xy offsets so the RIM regime gets exercised at different lengths (some demos enter chamfer immediately, some require 10mm of sliding). Suggested grid:

| Initial peg xy offset from slot | N demos |
|---|---|
| 0–2mm (peg lands on chamfer directly) | 5 |
| 2–5mm (short rim slide) | 5 |
| 5–10mm (medium rim slide) | 10 |
| 10–15mm (long rim slide) | 5 |
| **Total per object** | **25** |

### 7.4 Object/grasp variation

Each of {u_brown, u_orange, line_green, inverted_u_yellow} × at least 2 grasp_ids = 8 conditions. 25 demos per condition = **200 demos total** for full coverage. Given the operator's time budget (~60 min/session, ~30 demos/hour at ~2 min each), this is ~6-7 hours of collection time.

A meaningful first-pass dataset can be **30 u_orange demos** at varied xy offsets (2 hours), enough to derive the law for one object and validate the methodology.

### 7.5 Collection CLI

```bash
# Per-demo, with the part placed manually at varied xy offsets:
python3 -m compliant_insertion_studio.scripts.run_assembly_step \
  --object-name u_orange --base-name base1 --grasp-id 1 \
  --grasp-width 35 --mode real
```

Each demo writes 5 files to `compliant_insertion_studio/logs/` with timestamp basename.

### 7.6 Per-demo metadata to record (operator note)

For each demo, the operator should note (in a per-session log file or dashboard prompt):
- approximate initial peg xy offset from slot center (visual estimate, mm)
- subjective difficulty (1-5)
- any unusual behavior (peg got stuck, multiple corrections, etc.)
- whether seat completed cleanly

This metadata isn't sensor data but is essential for stratifying the regression in section 3.

---

## 8. Out-of-scope (deliberately)

This document does NOT:
- Propose a finalized control law (the law comes out of running the pipeline on data).
- Specify FSM code changes (those follow from the synthesized law, after validation).
- Address insert termination (`SEATED` detection is well-handled by the existing FSM; this work is about FIND_HOLE → ENTRY_SETTLE).
- Address operator-perception of slot location (assumed to be the same uncertain CAD prior we've always had).

---

## 9. Files this work produces

| File | Purpose |
|---|---|
| `analysis/scripts/30_segment_regimes.py` | Implements section 2's detectors + segmentation algorithm |
| `analysis/scripts/31_decode_operator_action.py` | Implements section 3's direction decoder |
| `analysis/scripts/32_decode_transition_triggers.py` | Implements section 4's transition trigger decoder |
| `analysis/scripts/33_synthesize_law.py` | Combines outputs into section 5's YAML |
| `analysis/scripts/34_validate_law.py` | Implements section 6's self-consistency check |
| `analysis/REGIMES.md` | This document |
| `analysis/derived_law.yaml` | Output of step 5 once data exists |

These are written ONCE, then re-run as more GOLD data is collected to refine thresholds and tighten the law.
