# Feature Research

**Domain:** Force-compliant peg-in-hole assembly with kinesthetic-demo data collection and analytical (rule-derived) policy synthesis
**Researched:** 2026-05-01
**Confidence:** MEDIUM-HIGH (table stakes drawn from NIST benchmark + Schimmels accommodation theory + Suarez-Ruiz / Pham fine-assembly framework + ROS demonstration-collection conventions; some operator-UX features are LOW confidence and flagged inline)

---

## Feature Landscape

### Table Stakes (Must Have for Safe + Repeatable Demo Collection)

These are non-negotiable. Missing any of these makes the system either unsafe (operator hand near robot, gear damage, protective stops) or scientifically useless (logs that can't be re-analyzed in 2 weeks). Every commercial and research system surveyed has all of these in some form.

#### Episode Lifecycle (safe demo collection)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Explicit phase state machine `PRE → HOVER → ZERO → ACTIVE → DONE/ABORT`** with phase tag in CSV | Every contact-rich-manipulation system in the literature (Suarez-Ruiz/Pham 2016, Chen 2016 AIM, Park "compliance-without-force-feedback") segments inserts into approach/search/insertion phases — without a phase tag, you can't slice F/T traces meaningfully later. | LOW | Already partially in compliant_insert.py. Phase enum + monotonic transitions + per-transition timestamps in sidecar. |
| **HOVER pre-contact pose** at base_xy + per-object_offset_xy with z = base_z + hover_offset, holding current EE orientation | Without a defined hover pose, ZERO happens at varying poses, biases vary, and trajectory replays can't be aligned cross-episode for overlay analysis. | LOW | Reuses `translate_for_target_real`. Already specified in PROJECT.md. |
| **F/T zero with measured residual bias verification** (1.0 s post-controller-switch settle, zero call, 0.5 s post-zero settle, |F| < 2 N gate) | Validated empirically this session — short settles produced 4 N biases that drove wrong-direction drift. UR force_mode integrates the bias as gravity-compensated wrench, so residual bias = constant drift. | LOW | Already verified working. Warn-don't-abort on > 2 N (operator may be touching robot during zero). |
| **Safe-height-then-home exit** on DONE/ABORT (NOT direct move_home) | Direct `move_home` from inserted pose plans straight-line joint trajectory through the inserted base — observed during this session. Without bookend, every successful insert risks crashing on exit. | LOW | Already specified. Critical safety feature. |
| **Clean SIGTERM cleanup** that always switches back to scaled_joint_trajectory_controller and stops force mode, even on Python exception | Operator's hand is near robot. If SIGTERM doesn't restore position control, the next operator command gets routed to a dead/wrong controller. Pendant Local-mode means no `--recover` rescue. | MEDIUM | `try/finally` with explicit StopForceMode + SwitchController. Test with exceptions injected (KeyboardInterrupt, ROS service timeout). |
| **Force-clamp ceiling** ≤ 5 N commanded Fz default (configurable per object) | Constraint from PROJECT.md — gear/part/fixture damage. Schimmels accommodation theory shows force assembly works at gentle commanded forces if compliance matrix is correct; high-force "pushing harder" is the wrong solution to misalignment. | LOW | Already in primitive. Per-object override via YAML. |
| **Wrench saturation / abort on |F| > N_max** (e.g., 25 N) | Hard safety: any unexpected contact spike (operator bumps robot, part wedges catastrophically) must abort to safe-height. NIST benchmark uses 17 N as the *success* threshold; abort threshold should be higher but bounded. | LOW | Sample F/T at controller rate, threshold check, raise abort signal. |
| **Manual abort signal** distinct from SIGTERM (SIGTERM = "I'm done, success"; abort = "stop now, mark fail") | Operator must be able to bail out of a clearly-failing insert without it getting tagged as success. | LOW | Custom signal (e.g., SIGUSR2 reused as abort + re-zero is overloading — recommend SIGINT for abort, SIGUSR1 for marker, SIGUSR2 for re-zero, SIGTERM for success-end). |

#### Telemetry (universally needed for retrospective analysis)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Wrench (Fxyz + Txyz) at controller rate** with monotonic timestamps | The fundamental signal. Every signature-extraction technique (EM-GMM contact-state classifier, feature-based compliance, hybrid-control papers) operates on this. | LOW | F/T topic publishes at 500 Hz on UR; downsample to 100 Hz for log size. |
| **Full TCP pose (xyz + quat) synchronized with wrench** | Required for F-vs-Z phase plots, for re-computing per-axis errors offline, and for 3D trajectory rendering. Missing pose = log is force-only and you've lost half the information. | LOW | tcp_pose_broadcaster ~125 Hz; interpolate to F/T rate or log both at native rates with shared monotonic timestamp. |
| **Phase tag column** | Every analysis filter ("show me ACTIVE phase only") requires this. Without it, you can't separate descent from contact from steady-state. | LOW | Already in PROJECT.md schema. |
| **Event marker column** (operator-driven via SIGUSR1) | Operator's annotations are the ground-truth labels for "this is the moment the part bottomed out." Without markers, you're inferring labels from F/T noise. | LOW | Toggle column 0/1 on signal; or auto-increment marker ID per press to distinguish events. |
| **Sidecar JSON metadata per episode** (object/base/grasp_id, assembly target, ISO timestamps, outcome, force-mode params, post-zero bias, free-text user_notes) | Logs without metadata are dead in 2 weeks. NIST benchmarking-protocols paper (Falco et al.) explicitly calls out metadata schemas as a top contributor to dataset reusability. user_notes captures the qualitative knowledge that doesn't fit in numeric columns. | LOW | JSON next to CSV. Prompt for user_notes at episode end (don't make it skippable — operator memory decays in minutes). |
| **Consistent path convention** `logs/insert_<object>_<YYYYMMDD_HHMMSS>.csv` + `.meta.json` | Dashboard auto-discovery requires a parseable convention. UTC timestamps to avoid DST bugs. | LOW | Use `datetime.now().strftime("%Y%m%d_%H%M%S")`. |
| **Commanded Fz logged separately from measured Fz** | Distinguishes "what I asked for" from "what happened." For derivation of the command-tracking gap (i.e., how much external force opposed the commanded descent), both are needed. | LOW | Add commanded_fz column; copy from latest force-mode service args. |
| **Per-axis error columns dx/dy/dz/droll/dpitch/dyaw** (recomputed per sample from target pose) | Recomputed offline-or-online: lets dashboard plot "approach error" without re-deriving from raw poses every load. Cheap to compute. | LOW | Already in PROJECT.md schema. |
| **Gripper width** logged | Detects gripper slip during insert (the part shifted in the jaws), which manifests as sudden force change unrelated to insert physics. Without width, you misattribute slip to insertion failure. | LOW | gripper_width topic at ~50 Hz; downsample-and-hold. |

#### Analyzer Dashboard (retrospective analysis)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Single-episode F vs t** (3 traces Fx/Fy/Fz on shared time axis) | The single most-used plot in every contact-manipulation paper. Bare minimum. | LOW | Plotly.js scatter, shared x-axis. |
| **Single-episode T vs t** (3 traces Tx/Ty/Tz) | Torques distinguish lateral misalignment (Tx/Ty spikes) from straight insertion. Required for any signature work. | LOW | Same as above. |
| **Z vs t with synced cursor** to F/T plots | Lets operator visually correlate "Fz spiked here" with "Z stopped descending here." Without it, eye has to count timestamps. | LOW | Plotly cross-plot hover. |
| **F-vs-Z phase plot** | Differentiator from typical sim viewers. Phase plots reveal signature shapes (contact-state literature: GMM contact identification operates on this). The shape of the F-vs-Z curve is the per-object fingerprint. | LOW | Just F[i] vs Z[i] scatter colored by t or by event_marker. **Strongly differentiates this dashboard from naive plotters.** |
| **Sidecar metadata panel** (object, target, outcome, user_notes shown alongside plots) | Without notes visible next to the plot, you're tab-switching constantly and losing context. | LOW | HTML table, no styling needed. |
| **Auto-discovery from `logs/` directory** | Manual file selection breaks the "drop CSVs and look" workflow described in PROJECT.md. | LOW | FileReader API + directory picker, or pre-built manifest at session end. |

#### Per-Object Configuration

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **YAML config schema** with axis-wise compliance flags, Fz target, hover offset z, termination thresholds, retry behavior, per-axis tolerance bands | All surveyed parametric systems (FANUC iRPickPRO teach pendant + insert macros, Schimmels admittance basis matrices, Suarez-Ruiz fine-assembly framework) externalize these knobs. Hardcoding = rewrite per part. | LOW | Already specified. Pydantic schema for validation. |
| **`defaults.yaml` + per-object override** (family/inheritance) | Without inheritance, adding a 5th object means cloning a 100-line YAML. With inheritance, it's a 5-line diff. NIST benchmarking suggests inheritance because most parts share most parameters. | LOW | Use deep-merge (e.g., `deepmerge` package or inline). Already specified. |
| **Per-object termination criterion expressed as combination logic** (force-absorbed AND z-reached, OR motion-stopped) | PROJECT.md flags this as a deliverable. Schimmels-style assembly uses force-absorbed; classical literature (Whitney) uses z-reached; modern compliant work (Park, Chen) uses combinations. The schema must permit all three and their AND/OR composition. | MEDIUM | Express as DSL or as predicate list with combination operator. Start with named primitives ("force_absorbed", "motion_stopped", "z_reached") + combinator ("all", "any"). |

---

### Differentiators (Improve Algorithm Derivation Quality / Speed Up Tuning)

These are where this project competes with off-the-shelf "just use admittance control" solutions. Aligned with the Core Value: making per-object tuning a one-config-file extension.

#### Analyzer Dashboard

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Cross-episode overlay view** (filter by object + outcome, overlay F/T traces) | This is where signatures *emerge*. A single episode is anecdote; 5 overlaid successful u_brown inserts reveal the canonical Fz-vs-Z shape. Without overlay, signature extraction is eyeballing one trace at a time. **Direct enabler of the rule-derivation deliverable.** | MEDIUM | Plotly multi-trace, time-align on first-contact event (not absolute t=0) so traces overlay despite different approach durations. **Time-alignment is the trick** — naive overlay on absolute time gives unaligned traces. |
| **Per-object signature card** (auto-computed: median Fz at success, |Tx|/|Ty| peak distributions, lateral travel during ACTIVE, descent duration) | These statistics are exactly the values that get copied into the per-object YAML. Auto-compute eliminates the "stare at plot, type number" loop. | MEDIUM | Pandas-equivalent reductions in JS, or pre-compute in Python and emit a stats JSON for the dashboard to load. |
| **3D trajectory rendering with target marker** | Catches "the EE drifted off to the side during ACTIVE" failures that don't show up clearly in 1D Z-vs-t plots. Especially useful for diagnosing line_green (asymmetric part) misalignments. | MEDIUM | Plotly 3D scatter + cone for target orientation. |
| **Event-marker vertical lines on time-axis plots** | Lines from event_marker column overlaid on F-vs-t. Shows "operator pushed at t=2.3s, Fz peak followed at t=2.4s" — links operator action to F/T response. | LOW | `shapes` array in Plotly layout. |
| **Failure-mode library view** (clusters failed episodes by feature similarity) | Once 5+ failures are collected, grouping them ("misaligned-x", "wedged", "gripper-slip") makes failure-recovery rules tractable. Without it, every failure looks unique. | HIGH | Defer until Phase that runs after data collection. Could be just k-means on stat-vector. |

#### Episode Lifecycle / Operator Interaction

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Mid-episode re-zero (SIGUSR2)** | Enables long-duration demos where F/T drift accumulates. Operator notices drift on terminal display, sends SIGUSR2 mid-air to re-baseline. | LOW | Already specified. Logs a re-zero event in sidecar. |
| **Live terminal display of |F| and |T| during ACTIVE** | Operator situational awareness — they see when force is climbing and can intervene. Without it, the only feedback channel is the robot's behavior, which lags. | LOW | Curses or just print-with-carriage-return at 5 Hz. |
| **Pause/resume during ACTIVE** (operator-controlled hold-position) | Lets operator pause to think mid-demo without ending the episode. Useful for "wait, let me try a different angle" without restarting. | MEDIUM | Force-mode parameters can be re-issued with zero target wrench → effectively pause. Not strictly needed for v1 if SIGTERM-and-restart is fast enough. **LOW confidence on value — may be unnecessary.** |
| **Free-text user_notes prompt at episode end** (with default suggestions like "clean", "wedged", "drifted-x") | Operator memory decays fast. Forced prompt captures the qualitative ground truth that no F/T sensor measures: "I had to push the back-left corner down." | LOW | `input()` after DONE phase before exit. Default suggestions reduce friction. |

#### Per-Object Configuration

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Manual-guidance fallback for unknown parts** (force-compliant, no autonomous termination, operator runs ~5 demos, system synthesizes starter config) | The "calibration ramp" from PROJECT.md. Avoids the two bad alternatives: refuse-to-run (frustrating) or default-config-gambling (dangerous). Differentiator vs commercial systems that require pre-tuned config. | MEDIUM | Mode switch: `--mode learn` runs ACTIVE phase forever, depends on operator SIGTERM, then runs the signature-card computation and emits a starter YAML. |
| **Termination criterion derived from data, not assumed** | PROJECT.md flags this as a project deliverable. Differentiator vs every commercial system (which hardcodes either force-threshold or position-threshold). The deliverable is the *answer to which works*. | HIGH (research) | Phase output is a chosen criterion + supporting evidence from the dataset. |
| **Statistical classifier carve-out** (k-means / decision tree on episode feature vectors) for outcome prediction | Backstop if hand-derived rules fail. NIST benchmarking literature notes that small-dataset classification (~20 episodes) often loses to hand-rules but is worth a 1-day spike. | MEDIUM | sklearn DecisionTreeClassifier on episode feature vector (median Fz, peak |T|, descent duration, lateral travel). Decision deferred per PROJECT.md. |

#### Failure Recovery

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Bounded retract+retry** (max 1–2 retries: retract 5 mm → re-approach → try again) | The Variable-Admittance literature consistently uses 1–2 retries; beyond 2, repeated wedging usually means worse alignment, not retryable contact. Bounded is the differentiator from "infinite retry until protective stop." | LOW | Counter in episode wrapper. Retract uses force-mode with positive Fz briefly, then position-mode jog up. |
| **Retract criterion = stagnation in ACTIVE phase** (descent rate < ε for > T seconds while |Fz| < target) | Distinguishes "stuck" (retry) from "bottomed out" (success). Without an explicit stagnation rule, "stuck" looks like "force-absorbed" and gets tagged success. | MEDIUM | Compute dz/dt over rolling window. Explicit rule makes the failure-vs-success boundary auditable. |

---

### Anti-Features (Deliberately NOT Building for This Scope)

Each of these is *commonly requested* in contact-rich manipulation research, *commonly built* in the literature, and *deliberately wrong* for this 20-episode, 4-part, single-operator, single-arm scope. Confirming PROJECT.md's "Out of Scope" with reasoning from the surveyed literature.

| Feature | Why Requested | Why Problematic Here | Alternative |
|---------|---------------|----------------------|-------------|
| **Vision-in-the-insert-loop** (AprilTag tracking during ACTIVE) | Modern peg-in-hole literature (Haugaard 2021 visual servoing, Wang 2021 combined perception, Zero-Shot Peg Insertion 2025) heavily favors vision+force fusion. Vision would seem to give "absolute" position correction. | The gripper occludes the AprilTag during ACTIVE phase (hardware-confirmed in PROJECT.md). Re-emergence mid-insert would inject a spurious pose update at exactly the wrong time. The sensor fusion would have to model occlusion explicitly = high complexity for low value. | Force-only control during ACTIVE (Suarez-Ruiz / Pham fine-assembly framework operates this way). |
| **Online learning / RL / policy-gradient updates at runtime** | TD3-on-admittance-params (MDPI 2025), reinforcement-learning compliance (Springer 2024), and DRL alignment (Wiley 2021) papers are abundant. RL appears to "tune itself." | RL needs O(1000s) of rollouts for convergence; this project has 20 rollouts/object. RL requires a reward signal that's exactly the unsolved problem (what's "success"?). The hand-derived rule-from-dashboard path actually *answers* the open question, RL kicks the can. | Offline rule derivation from logged demos. Statistical classifier as the only "learning" component, and only if rules fail. |
| **Multi-strategy retry chains / fallback ladders** | Hybrid-control + spiral-search + helical-search + tilt-and-retry compose into long fallback chains in commercial systems (FANUC search macros, KUKA WorkVisual macros). They appear robust. | Each new strategy adds N parameters that need tuning per object. With 4 objects + 1 generalization part, parameter explosion outpaces episode count. The 1–2 retry bound from PROJECT.md is correct: beyond 2, the right move is retract-to-safe-height-and-page-operator, not deeper rule cascade. | Bounded 1–2 retract+retry, then human escalation. |
| **Vision-language-model orchestration of the insert** | Recent VLM-for-robotics work (Zero-Shot Peg Insertion with VLMs, 2025) suggests LLMs can pick mating holes and parameters. The repo already runs an MCP server with LLM agents. | The agent calls `--insert` as a black-box primitive (per PROJECT.md). Putting an LLM *inside* the insert loop adds LLM latency (seconds) to a control loop that needs sub-100ms reactions. The LLM doesn't add information that F/T doesn't already provide. | Agent treats insert as atomic. LLM-in-loop is a separate research project. |
| **Dashboard styling polish / responsive design / theming** | Every dashboard demo screenshot shows polished UIs. Pull-quote-quality plots feel like a deliverable. | Sole user is the operator. Time spent on CSS is time not spent on signature extraction. PROJECT.md is explicit: "screenshots-into-paper quality is fine." | Functional Plotly defaults. Spend the design budget on *which plots to show*, not *how they look*. |
| **Server-backed dashboard / SQL-backed log store / log indexing service** | Industrial data-pipeline patterns (TimescaleDB, MLflow, Foxglove Studio) suggest server-backed log management. | Single operator, single machine, ~100 episodes total over project lifetime. Filesystem + JSON sidecar is sufficient and zero-operations. | Static HTML dashboard reading filesystem. PROJECT.md is explicit. |
| **Multi-operator concurrent demo collection** | Multi-arm parallel data collection is a common scaling pattern in modern imitation-learning datasets (RT-1, Open-X-Embodiment). | One physical robot, one operator at a time. The constraint is hardware, not software. | Sequential collection across sessions. Path convention supports this (timestamps disambiguate). |
| **Cross-robot portability layer** (Franka, KUKA, etc.) | Many published frameworks claim "robot-agnostic." | Algorithm uses UR `force_mode_controller` SetForceMode service surface specifically. Premature abstraction freezes API choices that aren't validated yet. | Document UR-specificity explicitly. Re-derive when a second robot actually exists. |
| **Real-time signature classification** (live "this episode is failing" detection) | Predictive failure detection is hot in contact-rich literature (FILIC 2025, feature-based compliance 2021). | Requires the trained classifier from offline analysis, which is itself a project deliverable. v1 = collect + analyze; v2 = could add live classification once rules exist. | Defer until rules are validated offline. Operator is the live failure detector. |
| **Rosbag-format logging** (instead of CSV+JSON) | Standard ROS practice. Tools like Foxglove, PlotJuggler, ROSAnnotator natively read rosbags. Better message-time fidelity. | Rosbags are opaque to the static-HTML dashboard (would require ros2 bag → CSV step or a wasm rosbag reader). Adds toolchain dependency for marginal benefit at this dataset size. | CSV + sidecar JSON. Could add rosbag-record as a parallel logger if needed for replay (cheap, additive). **NOTE: weak anti-feature — could legitimately be added as a parallel logger with no downside if Phase budget allows. LOW confidence in excluding it.** |

---

## Feature Dependencies

```
[Episode Lifecycle (PRE→HOVER→ZERO→ACTIVE→DONE)]
    └──requires──> [Phase tag in CSV schema]
                       └──enables──> [Cross-episode overlay] (filter by phase)
                       └──enables──> [Per-object signature card] (stats over ACTIVE only)

[Enriched CSV Schema (pose + wrench + phase + event_marker + commanded_fz)]
    └──requires──> [Synchronized timestamping at controller rate]
    └──enables──> [F-vs-Z phase plot]
    └──enables──> [Per-axis error overlay]
    └──enables──> [Signature extraction]

[Sidecar JSON Metadata]
    └──requires──> [Free-text user_notes prompt at episode end]
    └──enables──> [Dashboard metadata panel]
    └──enables──> [Failure-mode library view] (qualitative ground-truth labels)

[F-vs-Z Phase Plot] ──enhances──> [Per-object signature card]
                        (signature shapes are the visual companion to numerical stats)

[Cross-episode Overlay] ──requires──> [Time-alignment on first-contact event]
                            (without alignment, traces don't overlay meaningfully)

[Per-object Signature Card] ──feeds──> [Per-object YAML config values]
[Per-object YAML config] ──parameterizes──> [Universal compliant_insert algorithm]

[Termination Criterion (data-derived)] ──requires──> [≥5 episodes/object across success+failure mix]
                                       ──requires──> [Cross-episode overlay (to compare candidates)]
                                       ──enables──> [Per-object termination block in YAML]

[Failure-mode Library View] ──requires──> [Failed episodes tagged with user_notes]
                            ──requires──> [Cross-episode signature comparison]
                            ──enables──> [Failure-recovery rules in YAML (retry vs abort)]

[Manual-guidance Fallback Mode] ──requires──> [Episode lifecycle abstracted from termination logic]
                                ──enables──> [Adding new parts without pre-existing config]

[Statistical Classifier Carve-out] ──requires──> [Per-object signature card stats]
                                   ──conflicts──> [Insufficient episode count] (≥30 episodes recommended for tree)

[Bounded Retract+Retry] ──requires──> [Stagnation detection (dz/dt < ε)]
                        ──requires──> [Distinct retract pose (= safe height above hover)]
                        ──conflicts──> [Hard wrench-saturation abort] (must distinguish retry-able from abort-able)
```

### Dependency Notes

- **Signature extraction depends on enriched logging.** Drop pose, drop commanded Fz, or drop event markers and the signature card becomes unreliable. *This is the strongest dependency: telemetry schema must be correct before any analysis is built, because re-collecting demos is the expensive operation.*
- **Cross-episode overlay depends on time-alignment, not just trace plotting.** Naive overlay on absolute time produces unreadable plots. The first-contact-event alignment is what makes overlay useful — and finding "first contact" requires either event_marker or auto-detection of |F| > threshold.
- **Per-object YAML schema depends on termination-criterion answer.** The YAML can't be designed until you know whether it needs a single criterion field or a combinator. Recommendation: design the schema with combinator support (cheap), use single-rule configs initially.
- **Bounded retract+retry conflicts with hard wrench-saturation abort** in subtle ways. If retract-criterion fires at the same instant as wrench saturation, which wins? Recommendation: wrench saturation always aborts (safety > recovery); retract-and-retry only fires if force is *bounded* but progress is *stalled*.
- **Statistical classifier conflicts with low episode count.** A decision tree on 20 episodes overfits trivially. The carve-out is honest: try it, expect rules to win, but document the comparison.
- **Manual-guidance fallback depends on the lifecycle being abstracted from the termination logic.** If termination logic is hardcoded in the wrapper, fallback mode either has to re-implement the wrapper or hack a flag. Cleaner: wrapper takes a termination predicate as parameter; fallback passes `lambda: False` (operator ends via SIGTERM).

---

## MVP Definition

### Launch With (v1) — PoC for FMB1 (4 parts, 20-episode dataset)

Minimum viable to validate the concept (one universal algorithm + per-object configs derived from data).

- [ ] **Episode wrapper with PRE→HOVER→ZERO→ACTIVE→DONE/ABORT lifecycle** — without this, no safe collection
- [ ] **Safe-height-then-home exit** — without this, every successful demo risks crash on exit
- [ ] **F/T zero with residual-bias verification** — without this, force-mode integrates wrong baseline
- [ ] **SIGTERM/SIGUSR1/SIGUSR2/abort signal interface** — operator interaction baseline
- [ ] **Force ceiling (≤5 N default) + wrench-saturation abort (~25 N)** — safety baseline
- [ ] **Enriched CSV schema** (phase, event_marker, full pose+target, per-axis errors, wrench, gripper width, commanded Fz)
- [ ] **Sidecar JSON metadata** with end-of-episode user_notes prompt
- [ ] **Static HTML dashboard** with single-episode F-vs-t, T-vs-t, Z-vs-t, F-vs-Z phase plot, 3D trajectory, metadata panel
- [ ] **Cross-episode overlay view** (filter by object + outcome, time-align on first contact)
- [ ] **Per-object signature card** (median Fz at success, peak |T|, lateral travel, descent duration)
- [ ] **YAML config schema** with `defaults.yaml` + per-object overrides
- [ ] **Universal `compliant_insert.py`** parameterized by YAML
- [ ] **20-episode FMB1 dataset** (5 episodes × 4 objects, mix of clean + intentional misalignments)
- [ ] **Termination-criterion derivation deliverable** (decision documented with evidence)
- [ ] **Bounded 1–2 retract+retry with stagnation detection**
- [ ] **Generalization validation** on one part from second assembly

### Add After Validation (v1.x) — Once core works on FMB1

- [ ] **Manual-guidance fallback mode** for unknown parts — adds calibration ramp once basic flow is solid
- [ ] **Statistical classifier carve-out** (decision tree on signature features) — only if rules can't separate failure modes
- [ ] **Failure-mode library view in dashboard** — once enough failed episodes exist to cluster
- [ ] **Live terminal display of |F|/|T| during ACTIVE** — operator UX polish
- [ ] **Default suggestions in user_notes prompt** — reduces friction once vocabulary stabilizes

### Future Consideration (v2+) — Beyond PoC milestone

- [ ] **Full coverage of second assembly** (currently scoped as one part for generalization validation)
- [ ] **Real-time signature classification** ("episode is failing" warning) — requires v1 classifier first
- [ ] **Rosbag parallel logging** — additive, low-risk, useful if PlotJuggler/Foxglove debugging becomes needed
- [ ] **Pause/resume mid-ACTIVE** — only if SIGTERM-and-restart proves too slow in practice
- [ ] **Cross-robot portability layer** — only when a second robot exists

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Episode lifecycle state machine | HIGH | LOW | P1 |
| Safe-height-then-home exit | HIGH (safety) | LOW | P1 |
| F/T zero with bias verification | HIGH | LOW | P1 |
| Signal interface (SIGTERM/USR1/USR2/abort) | HIGH | LOW | P1 |
| Force ceiling + wrench saturation | HIGH (safety) | LOW | P1 |
| Enriched CSV schema | HIGH | LOW | P1 |
| Sidecar JSON metadata + user_notes prompt | HIGH | LOW | P1 |
| Single-episode dashboard plots (F/T/Z vs t) | HIGH | LOW | P1 |
| F-vs-Z phase plot | HIGH | LOW | P1 |
| Cross-episode overlay | HIGH | MEDIUM | P1 |
| Per-object signature card | HIGH | MEDIUM | P1 |
| YAML config + defaults inheritance | HIGH | LOW | P1 |
| Universal parameterized algorithm | HIGH | MEDIUM | P1 |
| Bounded retract+retry with stagnation detection | MEDIUM | MEDIUM | P1 |
| Generalization validation on second-assembly part | HIGH | MEDIUM | P1 |
| Manual-guidance fallback for unknown parts | MEDIUM | MEDIUM | P2 |
| Statistical classifier carve-out | MEDIUM (research) | MEDIUM | P2 |
| Failure-mode library view | MEDIUM | HIGH | P2 |
| Live terminal F/T display | LOW | LOW | P2 |
| 3D trajectory rendering | MEDIUM | MEDIUM | P2 |
| Event-marker vertical lines on plots | MEDIUM | LOW | P2 |
| Pause/resume during ACTIVE | LOW | MEDIUM | P3 |
| Rosbag parallel logging | LOW | LOW | P3 |
| Real-time signature classification | MEDIUM | HIGH | P3 |

**Priority key:**
- **P1**: Must have for FMB1 PoC milestone (Core Value)
- **P2**: Should have, add when v1 is validated
- **P3**: Nice to have, defer to follow-up milestones

---

## Comparison to Surveyed Systems

| Feature | NIST Benchmark (Falco/Marvel) | Suarez-Ruiz/Pham Fine-Assembly Framework | FANUC iRPickPRO | Variable Admittance + RL (MDPI 2025) | Our Approach |
|---------|------------------------------|------------------------------------------|-----------------|--------------------------------------|--------------|
| Force success threshold | 17 N at TCP | Variable per-task | Configurable per-macro | Learned | ≤5 N commanded, derived per-object from demos |
| Termination criterion | Position (peg fully inserted) | Hybrid force+position state machine | Position + force timeout | RL-learned implicit | **Project deliverable** — derive from data |
| Per-object parameterization | Manual config per peg geom | Per-task framework instantiation | Teach-pendant macros + lookup table | RL params per task | YAML inheritance (`defaults.yaml` + overrides) |
| Demo collection | Single-shot benchmarking, no demos | Demo-driven Gaussian Mixture Regression for admittance | Teach pendant programming (no demos) | RL rollouts (autonomous, no human demos) | Kinesthetic demos with operator narration |
| Failure recovery | Retry-on-fail, no smart recovery | State-machine fallback to search phase | Search macros (spiral, helical, etc.) | RL-explored implicit | Bounded 1–2 retract+retry with stagnation detection |
| Telemetry schema | Standardized in benchmark protocol | Per-experiment custom | Robot-internal logs (proprietary) | Reward + state vectors | CSV + sidecar JSON, dashboard-friendly |
| Operator role during run | Setup only, no narration | Demonstrator (kinesthetic) | Teach-then-replay | None (autonomous) | Demonstrator + annotator (event markers + notes) |
| Vision in loop | No (force-only benchmark) | Optional in framework | Optional (iRVision integration) | No (proprioceptive only) | **No** (gripper occludes AprilTag) |

---

## Sources

### NIST benchmarks and assembly protocols
- [Comparative Peg-in-Hole Testing of a Force-Based Manipulation Controlled Robotic Hand (NIST)](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=922206)
- [Peg-in-Hole Data | NIST](https://www.nist.gov/el/intelligent-systems-division-73500/peg-hole-data)
- [Benchmarking Protocols for Evaluating Small Parts Robotic Assembly Systems (Falco/Marvel et al., PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7537423/)
- [Comparative Peg-in-Hole Testing PMC mirror](https://pmc.ncbi.nlm.nih.gov/articles/PMC11008496/)

### Schimmels accommodation/admittance theory (passive compliance basis)
- [FASN Research page — Schimmels (Marquette)](https://www.eng.mu.edu/schimmelsj/research.html)
- [Schimmels & Peshkin — Force-Assembly with Friction (1994)](https://peshkin.mech.northwestern.edu/publications/1994_Schimmels_ForceAssembly.pdf)
- [Schimmels — Admittance matrix design for force-guided assembly (1992)](http://peshkin.mech.northwestern.edu/publications/1992_Schimmels_AdmittanceMatrixDesign.pdf)
- [The Eigenscrew Decomposition of Spatial Stiffness Matrices](https://www.semanticscholar.org/paper/The-eigenscrew-decomposition-of-spatial-stiffness-Huang-Schimmels/821173676efbcb933a4844809c7cc58cf747718b)

### Suarez-Ruiz / Pham fine-assembly framework
- [A framework for fine robotic assembly (Suarez-Ruiz & Pham, ICRA 2016)](https://ntu.edu.sg/) (paper widely cited; primary access via institutional libraries)
- [Can robots assemble an IKEA chair? (Suarez-Ruiz, Zhou, Pham, Science Robotics 2018)](https://www.science.org/doi/10.1126/scirobotics.aat6385)

### Compliant insertion / hybrid control / phase identification
- [Feature-Based Compliance Control for Peg-in-Hole (arXiv 2103.16003)](https://arxiv.org/pdf/2103.16003)
- [Position Identification in Force-Guided Robotic Peg-in-Hole Assembly (ScienceDirect 2014)](https://www.sciencedirect.com/science/article/pii/S2212827114011342/pdf)
- [Contact Pose Identification for Peg-in-Hole Assembly under Uncertainties (arXiv 2101.12467)](https://arxiv.org/pdf/2101.12467)
- [Robust Peg-in-Hole Assembly under Uncertainties via Compliant and Interactive Contact-Rich Manipulation (arXiv 2506.22766)](https://arxiv.org/html/2506.22766)
- [Research on hybrid force/position control method for robot peg-in-hole assembly (Sage 2025)](https://journals.sagepub.com/doi/10.1177/16878132241304254)
- [Research on Robotic Peg-in-Hole Assembly Method Based on Variable Admittance (MDPI 2025)](https://www.mdpi.com/2076-3417/15/4/2143)
- [Active compliance control of robot peg-in-hole assembly based on combined reinforcement learning (Springer)](https://link.springer.com/article/10.1007/s10489-023-05156-5)

### Demonstration / teaching / annotation
- [Teach Industrial Robots Peg-Hole-Insertion by Human Demonstration (Chen 2016 AIM)](https://wjchen84.github.io/publications/C2016_AIM.pdf)
- [DexForce: Extracting Force-informed Actions from Kinesthetic Demonstrations (arXiv 2501.10356)](https://arxiv.org/html/2501.10356v1)
- [FILIC: Dual-Loop Force-Guided Imitation Learning (arXiv 2509.17053)](https://arxiv.org/html/2509.17053)
- [ROSAnnotator: Web Application for ROSBag Data Analysis (arXiv 2501.07051)](https://arxiv.org/html/2501.07051v1)
- [polymathrobotics/event_recording — automatic event-driven rosbag recording](https://github.com/polymathrobotics/event_recording)
- [Versatile Demonstration Interface (arXiv 2410.19141)](https://arxiv.org/html/2410.19141)

### Commercial reference
- [FANUC iRPickTool/iRPickPRO Productivity Option (Motion Controls Robotics)](https://motioncontrolsrobotics.com/downloads/techdocs/iRPickTool_iRPickPRO.pdf)

---

*Feature research for: force-compliant peg-in-hole assembly with kinesthetic-demo data collection*
*Researched: 2026-05-01*
