# Pitfalls Research

**Domain:** Force-compliant peg-in-hole assembly with kinesthetic demonstrations and offline policy synthesis on UR5e + OnRobot RG2
**Researched:** 2026-05-01
**Confidence:** MEDIUM-HIGH (UR + strain-gauge F/T sensor specifics are HIGH from official UR docs and forum reports; the Robotiq FT-300 sensor manual cited below is a general strain-gauge reference, not the project's hardware — UR5e built-in F/T is the actual sensor; small-data classifier guidance MEDIUM; some operator-process pitfalls drawn from LfD survey literature MEDIUM)

This file builds on six pitfalls the operator has already characterized:

1. **F/T zero is gripper-orientation-dependent** — gravity bias re-emerges after pose changes
2. **Controller switches cause force spikes** — need ~1 s settle before zero
3. **AprilTag occluded by gripper during ACTIVE phase** — pose lookup unreliable
4. **`move_home` from inserted pose plans straight-line trajectories** that ignore inserted base
5. **IK in real-mode rotate can yield "wrist-tucked" postures** (already patched at `rotate_object.py:622`)
6. **Operator hand on the part adds to F/T reading** — hard to separate operator vs contact force

The pitfalls below are the *additional* ones found in research and from related forum / paper reports, organized by the dimensions in the question.

---

## Critical Pitfalls

### Pitfall 1: F/T zero captures the operator's hand load as "gravity bias"

**What goes wrong:**
ZERO phase runs while the operator is still resting a finger on the part (e.g., from staging the demo, or because they're physically supporting the gripper to position it). `zero_ftsensor` then subtracts the operator force as if it were gravity. After the operator releases, the wrench reads a *negative* of what they were applying, and the robot drifts in the wrong direction.

**Why it happens:**
`zero_ftsensor()` is documented as "subtracting the current measurement from subsequent readings" — it has no notion of "what's gravity vs what's contact." The 0.5 s settle after zero only catches noise, not a *constant* operator load.

**How to avoid:**
- Add a pre-zero gate: require operator to send a "ready, hands off" SIGUSR signal **before** ZERO starts; print "RELEASE PART NOW" to stderr and wait 2 s before zeroing.
- Sample wrench for ~0.5 s at three points: pre-zero, post-zero, +1 s after zero. If the post-zero residual drifts more than ~1 N within the first second, abort and re-zero (operator was still touching it).
- Log all three samples to the meta JSON as `pre_zero_wrench`, `post_zero_wrench`, `post_zero_drift_1s`.

**Warning signs:**
- ACTIVE-phase Fz starts non-zero in CSV before any contact
- Robot drifts laterally during the first 200 ms of ACTIVE with no commanded x/y compliance force
- Post-zero residual `|F|` looks fine (< 2 N) but `post_zero_drift_1s` > 1 N

**Phase to address:**
Episode wrapper / lifecycle phase (the phase that builds the PRE → HOVER → ZERO → ACTIVE state machine).

---

### Pitfall 2: F/T sensor temperature drift over a session invalidates cross-episode comparison

**What goes wrong:**
The internal UR F/T sensor warms up over the first 20–30 minutes of operation. Force readings drift by 1–3 N during this window. Episodes recorded in the first 10 minutes of a session use a different "baseline" than episodes recorded an hour later, so cross-episode signature comparison ("median Fz at success") is contaminated by drift, not real part-to-part variation.

**Why it happens:**
Documented behavior of strain-gauge based F/T sensors. The Robotiq FT-300 manual (cited as a general strain-gauge sensor reference, not our hardware) explicitly mentions temperature sensitivity. UR's internal F/T sensor — the actual sensor on this project — is no exception.

**How to avoid:**
- Warm-up procedure: run the arm for 10 minutes (idle, gripper closed, in HOVER pose) before recording the first episode. Document this as part of the "operator workflow" SOP.
- Record the **per-episode post-zero bias** (which the wrapper already does) and ALSO log ambient + sensor uptime in `.meta.json`. In the dashboard, gray out cross-episode overlays where bias differs by more than ~0.5 N.
- Re-zero between every episode (already planned). Do not assume one zero is good for the whole session.

**Warning signs:**
- `post_zero_bias` in `.meta.json` trends monotonically across a session's episodes (early ones have large residual, later ones small)
- Median Fz at "success" varies by > 1 N within a single object across the same session

**Phase to address:**
Episode wrapper phase (warm-up SOP + per-episode bias logging) and Dashboard phase (visualizing bias vs episode time).

---

### Pitfall 3: Controller switch races leave robot in undefined state

**What goes wrong:**
The wrapper sequence is `switch_controller(scaled_joint → force_mode) → wait → start_force_mode → ACTIVE → stop_force_mode → switch_controller(force_mode → scaled_joint)`. Each is an async ROS2 service. If `switch_controller` returns success but the controller hasn't actually transitioned its internal state (race with the controller manager's update loop), `start_force_mode` can fire against an inactive controller, return success, and the robot goes nowhere — but the wrapper thinks it's in ACTIVE. Worse: on shutdown, if `stop_force_mode` fires while the controller is mid-deactivation from a SIGTERM, the robot can be left in force mode with the position controller "active" — causing erratic behavior on the next primitive.

**Why it happens:**
`switch_controller` claims/releases hardware interfaces, which the hardware interface processes on its own update loop tick. Service success means "request accepted," not "transition complete." This race is documented in the ROS2 control / MoveIt2 issue history (issue #450 et al.).

**How to avoid:**
- After every `switch_controller` call, **poll** `/controller_manager/list_controllers` until the target controller's state is `active` (and the source controller is `inactive`). 100 ms poll, 2 s timeout, abort on timeout.
- After `stop_force_mode`, verify with the controller's published state topic that force mode is actually off before returning. Don't trust the Trigger response alone.
- Cleanup handler (SIGTERM trap) must be **idempotent**: call `stop_force_mode` regardless of perceived state, then call `switch_controller(force_mode → scaled_joint)` regardless, then verify both. Belt + suspenders.

**Warning signs:**
- Robot does not move during ACTIVE despite force mode appearing to start
- Next primitive after the wrapper exits behaves strangely (drifts, refuses commands)
- Logs show `start_force_mode succeeded` but no wrench commands appear in the controller's command topic

**Phase to address:**
Episode wrapper phase. Build the "controller state verifier" helper as part of the lifecycle infrastructure, not as an afterthought.

---

### Pitfall 4: `selection_vector` confusion — type=1 vs 2 vs 3 silently change the meaning of "compliant"

**What goes wrong:**
UR's force mode `type` parameter has three values:
- **1**: Force frame is the transformation between base and "feature pose" (your specified frame)
- **2**: Compliant axes are aligned with the feature frame (most common — "do force mode in this frame")
- **3**: Compliant axes are aligned with the feature frame **and the wrench is held constant** even when the robot rotates

Operators (and copy-pasted examples) routinely use type=2 by reflex. For peg-in-hole where the peg orientation can drift due to compliance in pitch/roll, type=2 means a commanded `Fz = -3 N in base_link` will start to push *sideways* relative to the part as the part tilts. Type=3 keeps the wrench locked to the *initial* feature frame regardless of robot rotation.

For full 6-DOF compliant insertion in `base_link` with the part nominally upright, type=2 is fine. But once the part can tilt > ~10°, the commanded Fz vector is no longer aligned with the hole, and the robot can push the part *across* the hole opening. This is a silent failure: no error, just wrong direction.

**Why it happens:**
The `type` semantics are buried in the URScript documentation. The ROS2 `SetForceMode` srv accepts an integer with no enum names. Tutorials don't differentiate.

**How to avoid:**
- Document the choice in the per-object YAML config (`force_mode_type: 2` with a comment) so it's an explicit decision, not a default.
- For peg-in-hole in `base_link` with parts that tilt, default to `type=2` only when the *operator selected base_link* as the feature frame. If the part is tilting more than ~5°, switch to a TCP-aligned feature frame so "compliant Z" follows the part.
- Validate in unit tests: with the part tilted 30°, commanded Fz should produce vertical motion if the operator's intent was "push toward the hole" — measure this in dry-run.

**Warning signs:**
- During ACTIVE, robot drifts laterally even though only Fz is commanded
- Insertion success rate drops off a cliff for parts that need pitch/roll compliance (e.g., `inverted_u_yellow` if it tilts during approach)
- F-vs-Z phase plot shows large Fx/Fy growing alongside Fz — the part is being pushed across the hole

**Phase to address:**
Parametric algorithm phase (when defining the per-object YAML schema). Also relevant in initial wrapper design — pick the default frame deliberately.

---

### Pitfall 5: `gain_scaling > 1` causes oscillation against hard surfaces

**What goes wrong:**
Force mode's `gain_scaling` factor multiplies the controller's response. UR documentation explicitly warns: values > 1 can make force mode unstable when the robot pushes against hard surfaces. For peg-in-hole, the moment of "peg tip touches base" is exactly that: a hard contact. Gain > 1 produces a high-frequency oscillation, the contact spikes 20+ N peak even with a 3 N command, and the controller can trigger a protective stop.

**Why it happens:**
Higher gain means faster correction of force error. With a stiff environment (metal-on-metal, thin tolerance), the system has insufficient damping margin. Recent published research (MDPI, "Experimental Evaluation of UR5e Collaborative Robot Force Control in Low-Force Applications") recommends gain=0.5 for the 1–4 N range and gain=1.0 only for 5–7 N.

**How to avoid:**
- Default `gain_scaling` to 0.5 in the universal config.
- Per-object override allowed but must be justified in a config comment (e.g., "1.0 used because the snap-fit needs faster transient response at the bend point").
- Pair gain choices with explicit `damping_factor`. Documented stable pairs: gain=0.5 / damping=0.7 (low force), gain=1.0 / damping=0.8 (mid force).
- Never expose gain to per-episode tuning — it's a system parameter, not a per-demo parameter, and tweaking it episode-to-episode contaminates the dataset.

**Warning signs:**
- High-frequency (>10 Hz) Fz oscillation visible in CSV at the moment of contact
- Protective stops triggered specifically at the descent-into-contact moment
- Peak Fz during contact transient is >3× the commanded value

**Phase to address:**
Universal algorithm config defaults phase (set conservative gain in `defaults.yaml`).

---

### Pitfall 6: `damping_factor` extremes — too low drifts forever, too high masks contact

**What goes wrong:**
`damping_factor` ∈ [0, 1], default 0.025 in URScript docs. Default 0.025 is *very low* damping — the robot maintains commanded velocity even after the wrench equalizes. For peg-in-hole this means: the peg seats, force balances, but the robot keeps trying to creep down. This produces low-amplitude vertical hunting that contaminates the "motion-stopped" termination criterion (motion never actually stops).

The opposite mistake: setting damping=1.0 to "be safe" makes the robot decelerate so aggressively it won't follow the operator's gentle push during demos — operator perceives robot as "stuck," pushes harder, F/T reading is now contaminated by operator force.

**Why it happens:**
The default value reflects URScript's behavior with no UR Polyscope GUI active; in practice, "useful" damping for low-force assembly is 0.5–0.8. Few tutorials state this.

**How to avoid:**
- Default `damping_factor` to 0.7 in `defaults.yaml`.
- Document explicitly: "Damping is a *settling* parameter, not a safety parameter — it controls how aggressively the robot decelerates when wrench equalizes."
- Add a "termination compatibility" check in the dashboard: if the chosen termination criterion is "motion-stopped," verify damping ≥ 0.5 in the config or warn that motion-stopped will never trigger reliably.

**Warning signs:**
- TCP Z-velocity in the CSV oscillates around zero rather than decaying to zero post-contact
- Operator complaint: "the robot won't let me push it"
- Termination criterion "motion-stopped" never fires within the timeout

**Phase to address:**
Universal algorithm config defaults phase + Dashboard phase (compatibility warning).

---

### Pitfall 7: Termination on Z-reached false-positives at the *first* contact (peg tip on chamfer, not seated)

**What goes wrong:**
"Z-reached" termination compares current TCP Z to a target Z. If the target is set to the *fully seated* Z but the robot stops at the *chamfer-touch* Z (because the operator stopped pushing, or because compliance absorbed the force), Z-reached fires falsely — the dashboard logs "success" but the part is sitting 5 mm proud.

For multi-peg parts (`fork`, `bracket`), this is even worse: one peg can be seated and the other not, but the TCP Z averages out to "close enough" — false success.

**Why it happens:**
Z-reached is the laziest termination criterion to implement and the most attractive to derive from a single demo. It assumes the demo's final Z is a reliable feature, but final Z is an *outcome* of the operator's behavior, not a property of the seated state.

**How to avoid:**
- **Never use Z-reached alone.** Always combine with one of: force-absorbed (Fz settled at commanded value for ≥ 0.3 s) OR motion-stopped (|TCP velocity| < threshold for ≥ 0.3 s).
- Per-object YAML schema requires `termination: { primary, secondary, must_agree: bool }` — at least two signals must agree, "must_agree=true" by default.
- For multi-peg parts (`fork`, `bracket`), require Tx/Ty within tolerance band as a third signal — uneven seating produces a torque, even if Z and Fz both look fine.

**Warning signs:**
- Cross-episode overlay shows "success" episodes with bimodal final-Z distribution (some at chamfer, some at seated)
- Per-episode user notes contradict outcome ("looks proud but logged as success")
- For multi-peg parts, post-success TCP roll/pitch differs by > 2° from start orientation

**Phase to address:**
Termination criterion derivation phase (the project deliverable explicitly called out in PROJECT.md).

---

### Pitfall 8: Operator hand-on-part contaminates the "what does success look like" derivation

**What goes wrong:**
The operator's job during a guided demo is to help the part find the hole. Their hand applies force throughout. The CSV records the total wrench at the F/T sensor, which is `contact + operator + gravity_residual`. If the dashboard derives "median Fz at success = 4 N" from these episodes, the parameter is wrong — autonomous insertion has no operator hand, so the algorithm will overshoot.

**Why it happens:**
F/T sensors measure *total* force; they cannot distinguish the source. Even with bookend "I'm pushing now" / "I let go" SIGUSR markers (already designed), the operator might push intermittently or unconsciously brace.

**How to avoid:**
- **Bookend protocol**: enforce a strict "hands-off final ≥ 2 s" window at the end of each demo. The wrapper should print "STEP BACK — recording final state for 2 s" before the operator can SIGTERM the episode. This window is the *only* part of the wrench the parameter-derivation analysis should use.
- Sidecar JSON records `hands_off_window: [start_t, end_t]`. Dashboard's "median Fz at success" feature uses *only* samples in this window, never the full episode.
- Add an `event_marker` taxonomy: "operator pushing," "operator guiding (light contact)," "operator clear." Three states, not two.

**Warning signs:**
- Median Fz computed from full episode disagrees significantly (> 1 N) with median Fz computed from hands-off window
- Operator's user notes mention guiding the part, but the event_marker shows "clear" the whole time (operator forgot to mark it)
- Autonomous reproduction overshoots the demo's commanded Fz

**Phase to address:**
Episode wrapper phase (bookend protocol) + Dashboard phase (window-based stats) + Termination derivation phase (use only clean windows).

---

### Pitfall 9: Demo selection bias — operator records 5 "good" demos, ignores the half they aborted

**What goes wrong:**
Operators naturally curate. They re-record episodes that "felt off," only saving the clean ones. The resulting 20-episode dataset is the *easy half* of reality. The derived parameters work for the easy cases and fail for the misalignments, off-axis approaches, slight gripper-grip variations that the operator subconsciously avoided.

**Why it happens:**
LfD literature (MDPI survey on LfD in human-robot collaboration) explicitly identifies this as a top-three failure mode. PROJECT.md already notes "Mix of clean inserts and intentional misalignments to populate failure-mode library" — but this needs an enforcement mechanism, not a hope.

**How to avoid:**
- **Save aborted episodes too.** Do NOT delete on abort. Tag outcome as `aborted_by_operator` with a free-text reason in user notes.
- Mandate a quota: of 5 demos per object, at least 1 must be tagged `intentional_misalignment` and at least 1 must be tagged `failure_mode_demo`. Wrapper warns at end-of-session if quotas unmet.
- Dashboard's per-object signature card explicitly shows the success/failure ratio. If it's 100% success, flag it: "selection bias likely — record a failure case."

**Warning signs:**
- Logs directory has fewer episodes than the operator started (deletion in progress)
- All episodes for an object have identical outcome
- Cross-object Fz signatures look implausibly tight (real operators are noisier than this)

**Phase to address:**
Data collection process phase (enforce protocol) + Episode wrapper phase (don't allow easy deletion).

---

### Pitfall 10: Per-object parameter coupling — tuning compliance mask, Fz, and speed_limit independently produces nonsense

**What goes wrong:**
The per-object config has multiple knobs: `selection_vector` (which axes are compliant), `commanded_wrench`, `speed_limit`, `gain_scaling`, `damping_factor`. These knobs **interact**:
- Increasing `commanded_Fz` while reducing `speed_limit` makes the robot slower but pushes harder — the *energy* per unit time is unchanged, but the contact stiffness changes.
- Adding pitch compliance to `selection_vector` while keeping `gain_scaling=1` lets the part swing freely — the operator's first demo might look great, the second might tip over.

If parameters are tuned one-at-a-time in isolation, you can find a "working" config for one knob that breaks when another is changed.

**Why it happens:**
Force-mode controllers are MIMO systems. The 6 DOF + wrench + scaling + damping space is high-dimensional and not separable.

**How to avoid:**
- **Tuning workflow as a fixed recipe**, not freeform: (1) fix gain and damping to the universal defaults, (2) choose selection_vector based on which axes the *demo data* shows operator using compliance on, (3) set commanded_Fz from the median of the hands-off window, (4) only then tweak speed_limit if needed.
- Document this as the "tuning a new part in 30 min" SOP. The order matters more than the values.
- Dashboard provides a "config diff" view: when changing a per-object config, show which other parameters might need to change.
- Never commit to a config without running 1 dry-run validation episode against it.

**Warning signs:**
- Operator changes one parameter, reports inserts now fail at a different point
- Two different configs for the same object both "work" but produce wildly different signatures
- Configs for similar parts (e.g., u_brown vs u_orange) differ in unrelated parameters

**Phase to address:**
Parametric algorithm phase + Generalization validation phase (the SOP is the deliverable).

---

### Pitfall 11: Auto-classifier overfit to 20-episode dataset, indistinguishable from heuristics

**What goes wrong:**
Operator decides to "try a decision tree" on the per-episode feature vectors. With 20 episodes, even a depth-3 tree can split perfectly on training data. Cross-validation looks great because of leave-one-out's high variance. The "learned" classifier is functionally identical to "if median_Fz > 3.5 then success" — which is a heuristic. Operator concludes ML "worked" and ships it. Next assembly has different scale, classifier breaks completely.

**Why it happens:**
Decision trees and small-k classifiers overfit aggressively on small N. Cross-validation on N=20 has wildly noisy fold scores. Without held-out *future* data, all reported accuracy is optimistic.

**How to avoid:**
- **Heuristics are the default**, classifier is the carve-out (PROJECT.md already specifies this).
- If a classifier *is* tried, the comparison rule is: classifier must beat the best heuristic by ≥ 15 percentage points on a held-out object (not held-out episodes within the same object). The held-out-object split forces the classifier to actually generalize.
- Forbid feature engineering that uses the outcome label (e.g., "max Fz before success" — `before success` is the leak).
- Compare only on episode-level metrics; never compare on sample-level (each episode contributes one prediction, not one per CSV row).

**Warning signs:**
- Reported classifier accuracy > 95% on N=20
- Classifier "feature importance" is dominated by 1–2 features that suspiciously look like the heuristic rule already in use
- Classifier fails on the second-assembly validation, heuristics don't

**Phase to address:**
Classification carve-out phase (the explicit research deliverable post-data-collection).

---

### Pitfall 12: Snap-fit force signature inverts at the seat point — termination fires *before* the snap

**What goes wrong:**
Snap-fit and interference-fit parts have a non-monotonic force signature: rising force as the cantilever deflects, *peak* at the moment of maximum interference, then a sharp *drop* as the snap clears. Force-absorbed termination keyed to the peak fires at the deflection moment, not the seated moment. The robot stops mid-snap. Manual force completion required.

**Why it happens:**
Standard peg-in-hole termination heuristics assume monotonic force-vs-depth. Snap-fit physics violates this assumption (literature: insertion force is a peak, not a settling value; retention force is 1.5–3× insertion force).

**How to avoid:**
- Per-object YAML must declare `signature_type: monotonic | snap_fit | interference_fit`.
- For snap-fit: termination requires `Fz_peak_then_drop` event (force rose, then dropped by ≥ X% within Y ms) AND Z-reached AND motion-stopped. Three-signal AND.
- Dashboard auto-detects signature type from cross-episode overlays: if Fz traces are unimodal-rising, monotonic; if they show a peak with a trailing drop, snap-fit. Warn if the operator's config disagrees with the detected type.
- For FMB1's parts: most likely all monotonic, but `inverted_u_yellow` could be snap-fit-like depending on tolerance — measure, don't assume.

**Warning signs:**
- Cross-episode F-vs-Z phase plot shows a *loop*, not a curve (force drops below the rising-curve value at the same Z)
- Operator notes mention "I had to give it a final push"
- Part is repeatably proud by ~1–2 mm in autonomous mode, hand-completable

**Phase to address:**
Termination criterion derivation phase + Per-object config schema phase. Detect the signature type from the data, don't pre-specify.

---

### Pitfall 13: Multi-peg parts engage one peg first — single-peg termination triggers false success

**What goes wrong:**
For `fork` or `bracket` parts (multiple pegs), one peg almost always touches first due to manufacturing tolerance. The "leading peg" produces an Fz spike. If termination fires on first Fz spike, the second peg never seats. Worse: continued descent with one peg seated and one peg unseated produces a *torque* (Tx or Ty), not increased Fz — and torque-blind termination misses it entirely.

**Why it happens:**
Single-peg-style termination heuristics generalize poorly to multi-peg. Jamming analysis literature (dual peg-in-hole research) shows that two-point contact dynamics are fundamentally different from single-point.

**How to avoid:**
- For multi-peg parts, termination MUST include a torque-band check: `|Tx| < tx_tol AND |Ty| < ty_tol` for ≥ 0.3 s. Until torques settle near zero, the part is not evenly seated.
- Encode the part topology in the YAML: `peg_count: N`, `peg_layout: { single | linear_2 | triangular_3 | etc }`. Termination logic branches on this.
- During data collection, explicitly record the operator narrating "first peg engaged" and "fully seated" with separate event_markers — these become the ground truth for tuning the torque thresholds.

**Warning signs:**
- Terminal Tx or Ty in autonomous mode > 2× the typical value seen in successful demos
- Visible part tilt in operator post-mortem inspection
- Single-peg config copied for multi-peg part performs poorly

**Phase to address:**
Per-object config schema phase + Termination derivation phase (introduce torque-band requirement).

---

### Pitfall 14: HTML dashboard chokes at ~50 episodes due to in-browser CSV parsing memory

**What goes wrong:**
Static HTML + Plotly.js + FileReader, no server. Each episode CSV is 30 s × 100 Hz × ~20 columns = ~60k cells. At 50 episodes, the browser is holding ~3 million cells in memory plus all the rendered SVG paths. Plotly.js is documented as taking several GB to render ~40 traces at 1600 ticks (community report). Firefox in particular consumes ~2 GB of memory on dashboards regardless of complexity. At 100 episodes, the dashboard becomes unusable.

**Why it happens:**
Plotly.js renders all data as SVG by default. SVG DOM nodes don't garbage-collect cleanly when traces are added/removed. FileReader holds the full CSV string in memory until parsed. Browser tabs leak memory across reloads.

**How to avoid:**
- **Decimate on load**: don't plot at 100 Hz. Downsample to 20 Hz for overview plots, give a "zoom to see full resolution" mode that re-loads the segment from the file.
- Use Plotly.js's `scattergl` (WebGL) instead of `scatter` (SVG) for time-series traces. Significantly lower memory.
- Lazy-load: don't parse all 50 CSVs upfront. Parse a *summary* (computed feature vector + thumbnail trace) for the per-episode card, parse the full CSV only when an episode is opened.
- Cap the dashboard's "overlay all episodes" mode at 20 traces; provide filtering UI to subset.
- Test with synthetic data at 100 episodes before declaring the dashboard done.

**Warning signs:**
- Tab consumes > 2 GB RAM
- "Overlay all" view takes > 5 s to render
- Browser's Performance tab shows long-running scripts during plot updates
- Operator complains "dashboard is slow"

**Phase to address:**
Dashboard phase. Build with decimation and `scattergl` from day 1, not as a v2 optimization.

---

### Pitfall 15: Protective stop in Local pendant mode requires physical pendant interaction — wrapper must avoid triggering them

**What goes wrong:**
PROJECT.md notes the operator wants Local mode (manual control accessibility), so dashboard service `--recover` calls fail. Once a protective stop fires in Local mode, *only* the physical pendant can clear it. UR's documented behavior also adds a 5-second cooldown before unlock can be requested. So a wrong move during ACTIVE = session interrupted, operator walks to pendant, clears stop, re-homes, restarts. Dataset collection slows dramatically. Worse: a protective stop *during* an episode leaves the CSV truncated mid-record, and the wrapper may not detect the truncation.

**What triggers protective stops in force mode (specifically):**
- Commanded wrench exceeded by environment (e.g., crash into base): force-mode-induced stop
- TCP velocity exceeds limit: too-fast force-mode response (gain too high, damping too low)
- Joint position limit approached: wrapper navigated to a pose near a singularity, force mode amplified the issue
- Sudden controller switch with non-zero command queued

**How to avoid:**
- Conservative defaults: gain ≤ 0.5, damping ≥ 0.7, commanded wrench ≤ 5 N (already specified in constraints).
- HOVER pose must be in the *interior* of the joint range, not near limits. Check joint distances to limits before entering force mode; reject HOVER if any joint is within 10° of a limit.
- Episode wrapper traps the protective-stop signal (visible via robot state topic) and:
  1. Marks CSV outcome as `protective_stopped`
  2. Closes CSV cleanly with footer row indicating early termination
  3. Tells operator (stderr): "Protective stop. Clear at pendant, then SIGUSR2 to abort cleanly."
- Test the wrapper's protective-stop handling deliberately during integration phase — don't discover it the hard way.

**Warning signs:**
- Joint angle delta in CSV grows abnormally fast right before stop (controller response chasing instability)
- Multiple protective stops in one session at the same waypoint (HOVER pose near a limit)
- Operator session ends after 3 episodes and they don't come back today

**Phase to address:**
Episode wrapper phase (protective-stop handling) + Integration / validation phase (deliberately test it).

---

### Pitfall 16: F/T sensor "lies" after `set_payload` is called mid-session — readings look fine but are biased

**What goes wrong:**
URScript's `set_payload(mass, cog)` interacts with `zero_ftsensor()`. Forum reports (UR community) indicate that in some software versions (5.4.x), calling `set_payload` zeroes the RTDE force data. If the wrapper or any sibling primitive calls `set_payload` between zero and ACTIVE — for instance, a gripper close adjusts the apparent payload — the F/T data goes to zero or to a stale baseline. Operator sees Fz = 0.2 N during ACTIVE, thinks "great zero." Robot drifts.

**Why it happens:**
Software version differences in URScript behavior; underdocumented coupling between payload and FT zero.

**How to avoid:**
- Establish a single source of payload truth: set_payload is called *once* during robot bringup with the gripper's mass and CoG. Never re-call it during a session unless the gripper picks up a heavier-than-expected part.
- After every `zero_ftsensor`, validate by reading wrench for 0.5 s and comparing to previous post-zero reading. If they differ by > 1 N with no pose change, log a WARNING and re-zero.
- If gripping an object with significant mass (> ~50 g) introduces visible drift, add a per-grasp `set_payload` call followed by a re-zero — but document this explicitly in the YAML.

**Warning signs:**
- ACTIVE-phase Fz starts non-zero with no operator contact and no environment contact
- Post-grasp wrench changes by ~1 N from pre-grasp wrench when at the same pose
- Different operators on the same hardware get different zero behavior

**Phase to address:**
Episode wrapper phase (single-source payload management).

---

### Pitfall 17: The 1.0 s controller-switch settle is necessary but not sufficient — the controller's command queue can hold residuals

**What goes wrong:**
Operator already correctly identified the controller switch causes force spikes, requiring ~1 s settle before zero. Beyond that: the *previous* controller (`scaled_joint_trajectory_controller`) may have queued joint commands that don't get flushed at switch. After 1 s settle and zero, the first commanded force is applied while the joint trajectory's last unprocessed command is still being chased — producing a transient that contaminates the first ~200 ms of ACTIVE.

**Why it happens:**
ros2_control's hardware interface buffers commands at the controller-update rate. Switching deactivates the source controller but doesn't necessarily clear its command buffer.

**How to avoid:**
- Before `switch_controller`, send the source controller a *current-pose* hold command (i.e., "stay where you are"). This empties the trajectory queue with a stable target.
- After switch + 1 s settle, do *another* sanity check: read TCP velocity for 200 ms; if |v| > 5 mm/s, wait additional 0.5 s. Don't enter ACTIVE while still drifting.
- Add this verification to the wrapper as a non-negotiable gate.

**Warning signs:**
- TCP position changes during the supposedly-static settle window
- First 200 ms of ACTIVE shows wrench transient unrelated to commanded force
- Behavior differs depending on whether the previous primitive ended with a fast or slow trajectory

**Phase to address:**
Episode wrapper phase. Add the "queue-flush + velocity verify" step to the lifecycle.

---

### Pitfall 18: Re-zero mid-episode (SIGUSR2) silently changes the meaning of subsequent samples

**What goes wrong:**
PROJECT.md specifies SIGUSR2 re-zeros F/T mid-episode. Useful for long demos where drift accumulates. But if the dashboard's stat computation ("median Fz at success") averages across all samples, samples before and after a re-zero are on different baselines. The "median" is meaningless.

**Why it happens:**
Re-zero is a discontinuity in the wrench signal. Naive statistics ignore it.

**How to avoid:**
- Log a `zero_event` row in the CSV (or a separate `zero_events` array in the meta JSON) at every zero call. Each row tagged with sample index.
- Dashboard stats partition on zero events: stats are computed *per zero-segment*, the latest segment is what the parameter derivation uses.
- Operator gets a UI warning if they re-zero more than once per episode ("re-zero significantly affects analysis quality").

**Warning signs:**
- Median Fz computed across full episode is suspiciously close to zero (averaged across re-zeros)
- F-vs-time trace shows a vertical step (the re-zero discontinuity)
- Episode notes mention re-zero but stats look unaffected

**Phase to address:**
Episode wrapper phase (CSV schema) + Dashboard phase (segmented stats).

---

### Pitfall 19: AprilTag re-emerges mid-insert, vision-aware code "helpfully" overrides force decisions

**What goes wrong:**
Operator already noted AprilTag is occluded during ACTIVE. But: the part can shift during compliance, the gripper can rotate, and the tag may briefly become visible again. If any code in the pipeline uses the most-recent vision pose without checking timestamp/staleness, it can overwrite the force-mode-driven trajectory with a vision correction based on a noisy reappearance.

**Why it happens:**
The MCP server (`server.py`) auto-injects orientation/grasp_id between primitive calls. If this auto-injection runs during a force-mode lifecycle, it can do the wrong thing.

**How to avoid:**
- During ACTIVE phase, the wrapper sets a flag (or context) telling the MCP server: "do not override pose; force mode is in control."
- All vision pose subscribers in the wrapper must check timestamp; reject any pose older than ~200 ms.
- Explicit unit test: simulate mid-episode AprilTag reappearance, verify the wrapper doesn't react.

**Warning signs:**
- TCP pose jumps abruptly mid-ACTIVE (vision override kicked in)
- Logs show vision pose updates during ACTIVE phase
- Behavior differs between sessions where the part stays occluded vs. partially visible

**Phase to address:**
Episode wrapper phase (vision lockout) + Integration phase (verify with the existing MCP server).

---

### Pitfall 20: "It worked once" — single-success deployment without robustness measurement

**What goes wrong:**
Operator gets a config working for u_brown, demonstrates it once successfully, ships it. Two days later, the part is gripped 0.5 mm differently, the table has shifted slightly, the F/T sensor warmed up to a different temperature — config fails. The "validated requirement" was a lie because N=1.

**Why it happens:**
Robotics validation budget is tight, the operator wants to move on to the next part. Single demonstrations of success are convincing in the moment.

**How to avoid:**
- Per-object validation requires ≥ 5 consecutive autonomous successes (no operator intervention) before marking the config "validated" in the requirements doc.
- Log every validation attempt to a separate `logs/validation_*` directory. Validation failures are NOT silently retried.
- Dashboard shows per-object success rate over the most recent 10 attempts, not all-time. A regression is visible immediately.

**Warning signs:**
- Validated config in PROJECT.md but logs show ≤ 1 autonomous success
- Validation log has more failures than successes for a "validated" object
- Operator can demonstrate success but only when "things are right today"

**Phase to address:**
Generalization validation phase. Make the success-rate gate explicit in the milestone exit criteria.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcode termination as "Z reached" with no force check | Easy to implement, works on first 1-2 demos | False positives on chamfer-rest, multi-peg, snap-fit | **Never** — always combine signals |
| Skip post-zero residual verification | Faster ZERO phase | Bad zeros pass silently, wrong-direction drift | Never — the check is 0.5 s |
| Use the same `gain_scaling` for all objects | Simpler config schema | Snap-fit needs different gain than smooth peg; protective stops on stiff parts | Acceptable if all parts are similar mechanical class (FMB1 PoC scope) |
| Plot all CSV samples in dashboard at full rate | Highest fidelity view | Browser dies at 50 episodes | Acceptable for single-episode view; never for overlay view |
| Auto-delete aborted episodes | Cleaner logs directory | Lose failure-mode data, demo selection bias | **Never** — abort log is highest-value data |
| Trust `switch_controller` success response | Faster wrapper | Race conditions leave robot in undefined state | Never — always poll for actual transition |
| Re-zero on every ACTIVE entry but skip residual check | Convenient, "always fresh" | If operator's hand on part during zero, contamination is invisible | Only if a hands-off enforcement protocol is enforced |
| Single-demo config tuning | 30 min per part | Overfit to one operator behavior, fails on second part | Acceptable for first-cut config; never for validated config |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `force_mode_controller` start/stop | Trust service Trigger response | Verify via controller state topic before proceeding |
| `switch_controller` | Treat as synchronous | Poll `list_controllers` for actual transition (100 ms × 20) |
| `zero_ftsensor` | Call once per session | Re-call at every ZERO phase; verify residual; log post-zero bias |
| `set_payload` | Call from multiple places | Single source of truth at robot bringup; never mid-episode unless documented |
| `objects_poses_real` (vision) | Use unconditionally | Check timestamp staleness; lock out during ACTIVE |
| `gripper_command` (OnRobot RG2) | Assume close = grasp | Verify with `gripper_grasp_detected` AND width check |
| MCP server auto-injection | Allow pose override during ACTIVE | Set explicit lockout flag in wrapper context |
| SIGTERM cleanup | Direct switch back to position controller | Stop force mode FIRST, settle, THEN switch — or risk transient |
| CSV writing | Buffered with no flush | Flush every N rows; ensure crash-safety so truncated logs are still partially usable |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Plotly.js `scatter` (SVG) for time series | Browser slow above ~20 traces × 1000 points | Use `scattergl` (WebGL) | ~50 episodes |
| Full-resolution overlay of all episodes | Browser hangs, 2+ GB memory | Decimate to 20 Hz for overview, full-res only on zoom | ~30 episodes if not decimated |
| Loading all CSVs into memory upfront | Slow dashboard load, eventual crash | Lazy-load: parse summary first, full CSV on demand | ~50 episodes |
| ROS2 service call timeouts left at default | Dashboard's auto-discovery hangs on missing files | Explicit timeout per call (2 s for transitions, 5 s for force ops) | Anytime hardware misbehaves |
| Re-rendering Plotly on every state update | UI lag during cursor move | Use `Plotly.restyle` for partial updates, not `Plotly.react` for full re-render | ~10 traces |
| FileReader for each CSV synchronously | UI blocks on load | Use Web Workers for parsing | ~10 episodes |

## Security/Safety Mistakes

(Domain-specific — robot safety, not network security)

| Mistake | Risk | Prevention |
|---------|------|------------|
| Commanded wrench > 5 N default | Part / fixture damage; protective stops | Hard cap in code, not just in config (`max_command_n: 5.0`) |
| HOVER pose computed without joint-limit check | Force mode amplifies near-limit behavior, protective stop | Pre-check joint distances ≥ 10° from limits |
| SIGTERM handler that doesn't switch back to position controller | Robot left in force mode after wrapper exits | Idempotent cleanup: stop force mode + switch + verify, in that order |
| No cap on ACTIVE duration | Operator wanders off, robot pushes indefinitely | Hard timeout (60 s default), auto-abort with safe-height retreat |
| Test in real mode without sim verification | Crash on first run | Wrapper supports `--mode sim` for state-machine validation; gate real mode behind successful sim run |
| Re-engaging force mode after a protective stop without verifying part position | Part may have shifted; second engagement crashes | Re-verify HOVER pose with vision after any protective stop recovery |

## Operator UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| SIGUSR1 / SIGUSR2 / SIGTERM ambiguity | Operator forgets which signal does what mid-demo | Print a one-line cheatsheet to stderr at each phase entry |
| Silent ZERO phase | Operator doesn't know when to release | Print "RELEASE PART NOW" at zero start; print "ZEROED — push to demo" at zero complete |
| Aborts indistinguishable from successes in CSV | Operator has to remember which is which | Wrapper prompts for outcome tag at end; sidecar JSON makes it impossible to skip |
| "Did the demo record?" uncertainty | Operator re-runs unnecessarily | Wrapper prints CSV path + sample count at exit |
| Dashboard requires manual file selection | Operator has to remember directory layout | Dashboard auto-discovers `logs/insert_*.csv` + matching meta |
| Per-object config is YAML the operator must hand-edit | Typos, wrong indentation, schema drift | Provide a `make-config` script that interactively prompts; validates schema |
| Validation failures are invisible | Operator declares success on N=1 | Per-object validation requires ≥ 5 consecutive successes, displayed in dashboard |

## "Looks Done But Isn't" Checklist

- [ ] **Episode wrapper:** Often missing protective-stop signal handling — verify the wrapper closes the CSV cleanly with `protective_stopped` outcome when one fires
- [ ] **F/T zero:** Often missing post-zero drift check (only checks instantaneous residual) — verify it samples for 1 s after zero and warns on drift
- [ ] **Controller switch:** Often missing actual-transition poll (trusts service response) — verify by killing the controller manager and checking the wrapper detects the bad switch
- [ ] **Termination criterion:** Often single-signal — verify config schema requires ≥ 2 signals AND'd together
- [ ] **Per-object config:** Often missing `signature_type` (assumes monotonic) — verify snap-fit detection logic exists in dashboard and per-object override works
- [ ] **Multi-peg config:** Often missing torque-band termination — verify config supports `peg_count > 1` with torque thresholds
- [ ] **Dashboard:** Often built with SVG plots that die at scale — verify using synthetic 100-episode dataset before declaring done
- [ ] **Aborted episodes:** Often deleted — verify the wrapper writes them to disk with `aborted` outcome, no path in code can delete them
- [ ] **Operator hands-off window:** Often unenforced — verify the wrapper prints a "STEP BACK" prompt with a 2 s waiting period before allowing SIGTERM
- [ ] **Validation:** Often N=1 — verify per-object success requires ≥ 5 consecutive autonomous runs logged
- [ ] **Generalization:** Often "we ran it on one new part" — verify the new part was tuned via the documented 30-min SOP, not bespoke fiddling
- [ ] **`set_payload`:** Often called from multiple places — grep for it; verify a single source of truth
- [ ] **Vision lockout during ACTIVE:** Often missing — verify simulating an AprilTag reappearance mid-episode doesn't move the robot
- [ ] **CSV crash-safety:** Often buffered without flush — verify a `kill -9` (NOT during X11 use) on the wrapper still produces a partially-readable CSV up to the last flush
- [ ] **Cleanup idempotency:** Often relies on state — verify calling cleanup twice in a row produces no errors

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Bad F/T zero (operator hand on) | LOW | Re-zero with hands off, re-run episode; discard contaminated one |
| Controller switch race | MEDIUM | Manual `switch_controller` via CLI, restart wrapper, no data loss if pre-ACTIVE |
| Protective stop in ACTIVE | HIGH | Walk to pendant, clear, manually move robot to safe height, re-home, restart episode (lost demo) |
| Snap-fit termination misfire | LOW | Re-tune `signature_type` in YAML, add `Fz_peak_then_drop` requirement, re-validate |
| Multi-peg uneven seating | MEDIUM | Add torque-band requirement to termination, re-derive thresholds from existing data |
| Demo selection bias | MEDIUM | Mandate failure quotas, re-collect failure cases, re-derive parameters |
| Classifier overfit | LOW | Roll back to heuristic, document decision in PROJECT.md decisions table |
| Dashboard memory crash | MEDIUM | Switch to `scattergl`, add decimation, retest with synthetic data |
| Mid-session F/T temperature drift | LOW | Re-zero between episodes (already planned), gray out cross-session overlays in dashboard |
| Aborted-episode loss | HIGH | If episodes were deleted, irrecoverable — re-run data collection. Prevention is the only fix. |
| Vision override during ACTIVE | MEDIUM | Add lockout flag, verify with mocked vision update, re-validate |
| `set_payload` mid-session | LOW | Restrict to bringup, verify in code review |
| Inserted-base collision on home | MEDIUM | Always exit ACTIVE via safe-height-then-home (already noted) — if missed, manual retract via pendant |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1. Operator-hand-during-zero | Episode wrapper | Pre-zero gate signal + post-zero drift sample |
| 2. F/T temperature drift | Episode wrapper + Dashboard | Warm-up SOP + bias-vs-time visualization |
| 3. Controller switch races | Episode wrapper | Poll `list_controllers` after every switch |
| 4. `selection_vector` type 1/2/3 confusion | Parametric algorithm config schema | YAML `force_mode_type` explicit + tilt-aware default selection |
| 5. `gain_scaling > 1` instability | Universal config defaults | Default to 0.5; per-object override requires comment |
| 6. `damping_factor` extremes | Universal config defaults + Dashboard | Default to 0.7; dashboard warns if termination/damping incompatible |
| 7. Z-reached false-positives | Termination derivation | Schema requires ≥ 2 signals AND'd; never Z-alone |
| 8. Operator-force contamination | Episode wrapper + Dashboard + Termination derivation | Hands-off window enforced + window-restricted stats |
| 9. Demo selection bias | Data collection process + Episode wrapper | Save aborts; mandate failure quota; dashboard shows ratio |
| 10. Per-object parameter coupling | Parametric algorithm + Generalization SOP | Document tuning order; "30 min SOP" deliverable |
| 11. Classifier overfit | Classification carve-out | Held-out-object validation gate; ≥ 15 pp lift required |
| 12. Snap-fit signature inversion | Termination derivation + Per-object schema | `signature_type` field; auto-detect from F-vs-Z phase plot |
| 13. Multi-peg uneven seating | Per-object schema + Termination derivation | `peg_count` field; torque-band requirement when > 1 |
| 14. Dashboard memory bloat | Dashboard | `scattergl` + decimation + lazy load; test at 100 episodes |
| 15. Protective stops in Local mode | Episode wrapper + Integration | Joint-limit check; conservative defaults; deliberate test |
| 16. `set_payload` interaction | Episode wrapper | Single source of truth at bringup |
| 17. Trajectory queue residual | Episode wrapper | Pre-switch hold command + post-switch velocity verify |
| 18. Re-zero discontinuity | Episode wrapper + Dashboard | `zero_event` rows in CSV; segmented stats |
| 19. AprilTag re-emergence override | Episode wrapper + Integration | Vision lockout flag; timestamp staleness check |
| 20. Single-success deployment | Generalization validation | ≥ 5 consecutive successes per object; per-object success-rate dashboard |

## Sources

- [UR ROS2 Driver: ur_controllers documentation](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_controllers/doc/index.html) — `gain_scaling`, `damping_factor`, force-mode parameters and stability warnings (HIGH)
- [UR official: URScript Dynamic Force Control](https://www.universal-robots.com/articles/ur/programming/urscript-dynamic-force-control/) — `force_mode` type 1/2/3 semantics (HIGH)
- [UR e-Series User Manual UR5e](https://s3-eu-west-1.amazonaws.com/ur-support-site/40974/UR5e_User_Manual_en_US.pdf) — protective stop behavior, recovery procedure (HIGH)
- [UR Forum: zero_ftsensor() precision](https://forum.universal-robots.com/t/precision-of-zero-ftsensor/42537) — zero behavior, single-sample limitation (MEDIUM)
- [UR Forum: 5.4.x set_payload and zero_ftsensor](https://forum.universal-robots.com/t/5-4-x-rtde-set-payload-and-zero-ftsensor/5146) — payload-zero coupling bug (MEDIUM)
- [UR Forum: Force mode to Protective Stop](https://forum.universal-robots.com/t/force-mode-to-protective-stop/15664) — common protective stop triggers in force mode (MEDIUM)
- [UR Forum: Dashboard restart without teach pendant](https://forum.universal-robots.com/t/dashboard-restart-program-after-safety-mode-violation-without-teach-pendant/2038) — Local mode recovery limitations (MEDIUM)
- [UR Support PDF: Understanding Protective Stops](https://s3-eu-west-1.amazonaws.com/ur-support-site/76519/Understanding%20Protective%20Stops.pdf) — protective stop fundamentals + 5 s cooldown (HIGH)
- [Robotiq FT-300 Sensor Manual](https://assets.robotiq.com/website-assets/support_documents/document/FT_Sensor_Instruction_Manual_PDF_20181218.pdf) — calibration drift, temperature, install-induced bias (HIGH)
- [Robotiq Knowledge: Operating and calibrating FT-300S](https://blog.robotiq.com/knowledge/operation-and-calibration-of-the-ft-300s-5-1736280819067) — drift sources, signal drift consequences (HIGH)
- [MDPI Sensors 2026: Experimental Evaluation of UR5e Collaborative Robot Force Control in Low-Force Applications](https://www.mdpi.com/1424-8220/26/5/1709) — empirical gain/damping recommendations for low-force assembly (HIGH)
- [ROS2 Control: Controller Manager (Humble)](https://control.ros.org/humble/doc/ros2_control/controller_manager/doc/userdoc.html) — switch_controller semantics (HIGH)
- [MoveIt2 issue #450: race condition between launching ros2_control and Servo](https://github.com/moveit/moveit2/issues/450) — documented controller-switch race patterns (MEDIUM)
- [Plotly Community: Browser memory consumption advice](https://community.plotly.com/t/browser-memory-consumption-advice/83632) — Plotly.js memory at scale (HIGH)
- [Plotly.js issue #553: Performance with 180k+ datapoints](https://github.com/plotly/plotly.js/issues/553) — rendering bottleneck thresholds (HIGH)
- [Plotly.js issue #5790: Performance regression in 2.x](https://github.com/plotly/plotly.js/issues/5790) — version-specific regressions (MEDIUM)
- [MDPI Robotics: Learning from Demonstrations Survey](https://www.mdpi.com/2218-6581/11/6/126) — operator bias in LfD; demo selection bias as a known top-three failure mode (HIGH)
- [arXiv 2403.10140: Comparative Analysis of Programming by Demonstration Methods](https://arxiv.org/html/2403.10140v1) — kinesthetic teaching data quality issues, smoothness/noise (MEDIUM)
- [MDPI Robotics: A Practical Roadmap to Learning from Demonstration for Robotic Manipulators](https://www.mdpi.com/2218-6581/13/7/100) — small-dataset pitfalls and the case for heuristics (MEDIUM)
- [MDPI Sensors: Peg-in-Hole Two-phase Scheme F/T Sensor for Dual-arm](https://www.mdpi.com/1424-8220/17/9/2004) — F/T threshold tuning fragility, multi-peg analysis (HIGH)
- [Springer 2025: Advances in Robotic Peg-in-Hole Assembly Comprehensive Review](https://link.springer.com/article/10.1186/s10033-025-01349-w) — failure modes, jamming, wedging fundamentals (HIGH)
- [arXiv 2506.22766: Robust Peg-in-Hole Assembly under Uncertainties via Compliant and Interactive Contact-Rich Manipulation](https://arxiv.org/html/2506.22766v1) — compliance + uncertainty interaction (MEDIUM)
- [ScienceDirect: Robotic jamming-free assembly via multi-DOF parallel end-effector](https://www.sciencedirect.com/science/article/abs/pii/S0736584525001371) — two-point jamming → one-point contact strategy (MEDIUM)
- [Royal Society Open Science: Mechanics of rectangular peg-hole disassembly](https://royalsocietypublishing.org/rsos/article/11/11/240956/92409/Characterizing-the-mechanics-of-rectangular-peg) — extraction force, fit-class force signatures (MEDIUM)
- [Springer: Insertion Force in Snap-Fits Assembly Simulation Study](https://link.springer.com/chapter/10.1007/978-981-15-9505-9_87) — snap-fit force signature: peak then drop (HIGH)
- [Wikipedia: Snap-fit](https://en.wikipedia.org/wiki/Snap-fit) — insertion force vs retention force ratios (MEDIUM)
- [Analytics Vidhya: Data Leakage Effects on ML Model Performance](https://www.analyticsvidhya.com/blog/2021/07/data-leakage-and-its-effect-on-the-performance-of-an-ml-model/) — feature leakage patterns (HIGH)
- [Milvus AI Reference: Handling overfitting in small datasets](https://milvus.io/ai-quick-reference/how-do-you-handle-overfitting-in-small-datasets) — small-N overfit risks; held-out-set discipline (MEDIUM)

---
*Pitfalls research for: force-compliant peg-in-hole assembly with kinesthetic demonstrations on UR5e + OnRobot RG2*
*Researched: 2026-05-01*
