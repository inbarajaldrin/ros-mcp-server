# Project Research Summary

**Project:** Compliant Insertion Studio (force-compliant peg-in-hole on UR5e + Robotiq 2F-85)
**Domain:** Contact-rich manipulation with kinesthetic-demo data collection and offline analytical (rule-derived) policy synthesis
**Researched:** 2026-05-01
**Confidence:** MEDIUM-HIGH (stack and architecture validated against live system; features/pitfalls drawn from converging literature + UR-specific forum reports)

## Executive Summary

This is a **single-operator, low-volume, transparency-first** subsystem inside an existing ROS2 MCP server. The expert pattern for force-compliant peg-in-hole at this scale (NIST/Falco benchmarking, Suarez-Ruiz/Pham fine-assembly framework, Schimmels accommodation theory, recent UR5e low-force evaluation work) is: parametric universal algorithm + per-object configs derived from a small set of kinesthetic demonstrations + offline rule extraction from a transparent dashboard. RL/transformers are wrong-tool for ~20 episodes; rosbag/Foxglove are wrong-tool for a single-operator filesystem-as-bus workflow. The recommended stack is everything-already-installed plus two CDN scripts (Plotly.js 3.5.1 + PapaParse 5.5.3); no Flask, no React, no sklearn, no PyTorch.

The single most important architectural insight is the **build-order interlock**: the schema must be locked first so the wrapper can write it, then the wrapper must collect real demos before the dashboard is built (building a dashboard against synthetic data produces a dashboard that visualizes a fiction), then the algorithm/termination criteria are *derived from the data the dashboard surfaces*, then integration into `translate_object.py:1085` is the very last step. Three independent research threads (architecture build-order DAG, features dependency graph, pitfalls phase mapping) converge on this ordering. A second cross-cutting insight, also independently triangulated, is that **termination criterion is a deliverable, not an input** — Z-reached / force-absorbed / motion-stopped each have documented failure modes (chamfer-rest false-positive, snap-fit peak-then-drop inversion, multi-peg uneven seating producing only torque), so the schema must support a combinator of ≥2 signals and the *answer* of which combinator/thresholds is chosen per-object emerges from the demo data.

The dominant risks are F/T-zero contamination (operator hand on part during ZERO bakes operator load into "gravity"; documented `set_payload`+`zero_ftsensor` coupling bug in UR 5.4.x; sensor temperature drift over 20–30 min of session warm-up) and demo-selection bias (operator silently curates "good" demos so the derived parameters generalize only to the easy half of reality). Mitigations are protocol-level (pre-zero "hands off" gate, ≥2 s "STEP BACK" hands-off window before SIGTERM, save-aborts-don't-delete, mandatory failure quotas, ≥5 consecutive autonomous successes per object as the validation gate) and config-level (single-source-of-truth `set_payload` at bringup only, `gain_scaling=0.5` / `damping=0.7` defaults from MDPI 2026 low-force eval, `ur_robot_driver` upgrade 2.12→2.13 for the F/T frame bugfix before collection starts).

## Key Findings

### Recommended Stack

Everything except two CDN scripts is already on disk (verified live via `ros2 interface show`, `ros2 topic hz`, `apt list --installed`). The `/force_torque_sensor_broadcaster/wrench` topic publishes at **500 Hz** (not 125 Hz) — important for CSV-writer sizing; recommended approach is read at 500 Hz, log at 100 Hz. Recommend an `apt upgrade` to `ur_robot_driver` 2.13.0 (April 2026) for an F/T frame bugfix *before* the 20-episode collection starts, since re-collecting demos is the expensive operation.

**Core technologies:**
- ROS2 Humble + `rclpy` 3.3.15 — already installed; matches operator's UR5e setup
- `ur_robot_driver` 2.12.0 → **upgrade to 2.13.0** — F/T frame bugfix; do this before data collection
- `ur_msgs/srv/SetForceMode` — verified live; the existing `compliant_insert.py` placeholder already calls it correctly
- Python 3.10 + NumPy 2.2.6 + PyYAML 6.0.2 + scipy — episode wrapper, per-object YAML, quaternion math (note: `pyproject.toml` pins `numpy<2` but apt has 2.2.6 — reconcile before milestone closes)
- Plotly.js 3.5.1 (full bundle, CDN) — `scattergl` (WebGL, not SVG) for time series; required to scale past ~50 episodes
- PapaParse 5.5.3 (CDN) — RFC-4180-correct in-browser CSV parsing

**Explicitly NOT used:** Flask/FastAPI (no backend — `file://` HTML), React/Vue/Svelte (one page, no routing), PyTorch/TensorFlow/sklearn (20 episodes; hand-rolled NumPy if a classifier is needed at all), rosbag2 (opaque to a static-HTML dashboard; CSV+JSON sidecar wins at this scale), `dashboard_client/recover` (pendant is in Local mode; service calls fail), direct URScript injection (project constraint).

Full detail: `.planning/research/STACK.md`.

### Expected Features

The 24 features in FEATURES.md split cleanly into three tiers, all backed by NIST benchmarking, Schimmels admittance theory, Suarez-Ruiz/Pham fine-assembly framework, Chen 2016 AIM, and DexForce / FILIC for kinesthetic-demo conventions.

**Must have (table stakes):**
- Explicit `PRE → HOVER → ZERO → ACTIVE → DONE/ABORT` lifecycle with phase tag in CSV (no phase tag = traces can't be sliced for analysis later)
- F/T zero with measured residual bias verification + post-zero drift check (1 s residual sample, not just instantaneous)
- Safe-height-then-home exit on every termination path (direct `move_home` plans through inserted base — observed)
- Idempotent SIGTERM cleanup (always switch back to position controller + stop force mode)
- Force ceiling ≤ 5 N commanded + wrench-saturation abort ~25 N (operator hand near robot)
- Enriched CSV schema: phase, event_marker, full pose+target, per-axis errors, wrench, gripper width, commanded Fz
- Sidecar JSON metadata + free-text user_notes prompted at episode end (forced, not skippable — operator memory decays in minutes)
- Single-episode dashboard: F/T/Z vs t with synced cursors, F-vs-Z phase plot, metadata panel
- YAML config schema with `defaults.yaml` + per-object override (deep-merge)
- Universal `compliant_insert.py` parameterized by YAML
- Per-object termination criterion expressed as a *combinator* of named primitives, not a single rule

**Should have (differentiators):**
- Cross-episode overlay view with **time-alignment on first-contact event** (not absolute t=0 — naive overlay is unreadable)
- Per-object signature card auto-computing the values that will be typed into the YAML (median Fz at success, |Tx|/|Ty| peak distributions, lateral travel during ACTIVE, descent duration)
- Manual-guidance fallback for unknown parts (full 6-DOF compliance, no autonomous termination, operator-terminated only — the calibration ramp)
- Bounded retract+retry (max 1–2) gated by stagnation detection (dz/dt < ε for > T seconds while |Fz| < target)
- 3D trajectory rendering with target marker (catches lateral-drift failures invisible in 1D Z-vs-t)

**Defer (v2+):**
- Statistical classifier carve-out (only if hand-rules can't separate failure modes; held-out-object validation gate, ≥15pp lift required)
- Failure-mode library view (clusters failed episodes — needs ≥5 failures collected first)
- Live terminal F/T display, pause/resume mid-ACTIVE, rosbag parallel logging, real-time signature classification, cross-robot portability
- Full coverage of second assembly (PoC scope is one part for generalization validation only)

**Anti-features (deliberately excluded):** Vision-in-the-insert-loop (gripper occludes AprilTag during ACTIVE), online RL (20 rollouts insufficient), multi-strategy retry chains (parameter explosion outpaces episode count), VLM mid-insert orchestration (LLM latency incompatible with control loop), dashboard styling polish, server-backed log store.

Full detail: `.planning/research/FEATURES.md`.

### Architecture Approach

Greenfield subsystem inside a brownfield repo. Five layers, mostly already-existing:

**Major components:**
1. **`translate_object.py:run_perform_insert_real`** (existing) — single-line edit at `:1085` to retarget script_path; arg passthrough already covers everything except `--user-notes` (collect-mode only)
2. **`primitives/compliant_insert_episode.py`** (NEW dispatcher, ≤150 LOC, no ROS deps) — resolves YAML, picks collect vs runtime mode, surfaces unknown-object fallback, spawns/imports wrapper
3. **`primitives/compliant_insert.py`** (existing scaffold ~250 LOC, evolves) — owns lifecycle, ROS subscriptions, force-mode RPCs, CSV/meta writing, signal handling, safe-height exit
4. **`primitives/insert_configs/defaults.yaml` + `<object>.yaml`** (NEW, committed) — deep-merge override pattern; loaded by lazy filename lookup, never glob-all (a YAML parse error in `inverted_u_yellow.yaml` must not break a `u_brown` insert)
5. **`tools/analyze_inserts.html`** (NEW, single-page, no build step) — Plotly.js + PapaParse from CDN, drag-drop `logs/` folder, auto-pair CSV ↔ meta JSON by filename stem

Telemetry bus is the filesystem (`logs/insert_*.csv` + `*.meta.json`, gitignored). Configs are the source of truth (committed, small, diff-readable). No DB, no ROS bag, no message broker, no runtime classifier — family selection is offline-only ("which family is this part?" is decided once by a human staring at the dashboard, committed as a YAML file). Subprocess boundary between `translate_object.py` and dispatcher follows existing `__RESULT_JSON__` sentinel convention; in-process between dispatcher and wrapper is recommended (simpler, faster) with subprocess as a fallback if rclpy state leaks during testing.

Full detail: `.planning/research/ARCHITECTURE.md`.

### Critical Pitfalls

The 20 pitfalls in PITFALLS.md cluster into two anti-patterns flagged loudest by the research, plus a handful of UR-specific gotchas:

1. **Operator hand on part during ZERO contaminates parameters for the entire session.** `zero_ftsensor` has no notion of "gravity vs contact" — it just subtracts the current measurement. If the operator is still touching the part when ZERO fires, that load is baked into the "gravity bias," and post-zero ACTIVE drifts in the wrong direction. Avoid: pre-zero "hands off" SIGUSR gate, "RELEASE PART NOW" prompt with 2 s wait, post-zero drift sample (1 s) logged separately to meta JSON, abort-and-re-zero if drift > 1 N.
2. **`set_payload` mid-session with `zero_ftsensor` is a documented UR 5.4.x bug** (forum-confirmed) — RTDE force data goes to zero or stale baseline. Avoid: single source of truth — `set_payload` is called *once* at robot bringup with gripper mass+CoG, **never** mid-episode. Validate every `zero_ftsensor` by reading wrench for 0.5 s and comparing to previous post-zero reading; warn if differ > 1 N with no pose change.
3. **Termination on Z-reached alone false-positives at chamfer touch** (peg tip rests on chamfer, not seated). For multi-peg parts this is worse: one peg seated + one peg unseated produces a *torque*, not increased Fz; torque-blind termination misses it entirely. Avoid: never use Z-reached alone; schema requires `termination: { primary, secondary, must_agree: bool }` with ≥2 signals; multi-peg parts add a torque-band requirement (`|Tx| < tx_tol AND |Ty| < ty_tol` for ≥ 0.3 s).
4. **`gain_scaling > 1` causes oscillation against hard surfaces; default URScript `damping_factor=0.025` drifts forever.** Default to `gain_scaling=0.5` / `damping_factor=0.7` in `defaults.yaml` (MDPI 2026 low-force eval). Per-object override only with a comment justifying why. Never expose as per-episode tuning — it's a system parameter, tweaking it episode-to-episode contaminates the dataset.
5. **Demo selection bias** — operators silently re-record "off-feeling" episodes and only save the clean ones. Resulting dataset is the easy half of reality. Avoid: save aborted episodes with `aborted_by_operator` outcome (do NOT delete on abort); mandate quotas (of 5 demos per object, ≥1 `intentional_misalignment` and ≥1 `failure_mode_demo`); dashboard shows success/failure ratio and flags 100%-success as "selection bias likely."
6. **F/T sensor temperature drift** over the first 20–30 min of session — 1–3 N baseline shift. Mitigation: 10-minute warm-up SOP (idle, gripper closed, in HOVER pose) before the first recorded episode; per-episode `post_zero_bias` logged; dashboard grays out cross-session overlays where bias differs > 0.5 N.
7. **Validation gate strengthened: ≥5 consecutive autonomous successes per object**, not single-success. Per-object validation requires no operator intervention; validation failures are NOT silently retried (logged to `logs/validation_*`). Dashboard shows per-object success rate over the most recent 10 attempts so regressions are visible immediately.

Plus: controller-switch races (poll `list_controllers` for actual transition, never trust the service Trigger response alone), trajectory queue residuals after switch (send a current-pose hold command before switching to flush queue, then verify TCP velocity < 5 mm/s before entering ACTIVE), Plotly.js memory bloat at ~50 episodes (use `scattergl` not `scatter` from day 1, decimate to 20 Hz for overview), snap-fit signature inversion (force peaks then drops at seat — declare `signature_type` per object), AprilTag re-emergence mid-ACTIVE (vision lockout flag set during ACTIVE; reject pose updates older than 200 ms).

Full detail: `.planning/research/PITFALLS.md`.

## Implications for Roadmap

The build-order DAG below is **non-negotiable** — it emerged from three independent research threads (architecture build dependencies, feature dependency graph, pitfall phase-mapping) all converging. Re-collecting demos is the expensive operation; therefore the schema must be right *before* collection, and the dashboard must be built *against real data* (not synthetic) so the parameter-derivation phase has the right shapes to work with.

### Phase 1: Schema + Wrapper Foundation (lock the data contract first)

**Rationale:** Every downstream consumer (dashboard, signature card, termination derivation, per-object YAMLs) depends on the CSV+meta schema. Get the schema wrong, and the 20-episode collection produces logs that have to be re-collected. Three researchers independently flagged this as the gating dependency. Wrapper extension is paired with schema lock because the wrapper is the only writer.
**Delivers:** `primitives/compliant_insert.py` extended to full lifecycle (PRE → HOVER → ZERO → ACTIVE → DONE/ABORT → safe-height-then-home); enriched CSV schema (phase, event_marker, full pose+target, per-axis errors, wrench, gripper width, commanded Fz); sidecar `.meta.json` with `pre_zero_wrench` / `post_zero_wrench` / `post_zero_drift_1s`; idempotent SIGTERM cleanup; signal interface (SIGUSR1 marker, SIGUSR2 re-zero, SIGTERM=success, custom abort); pre-zero "RELEASE PART NOW" gate + ≥2 s "STEP BACK" hands-off window before SIGTERM; controller-switch verification (poll `list_controllers` + post-switch velocity check); `primitives/shared/insert_config.py` loader with deep-merge; `defaults.yaml` with `gain_scaling=0.5` / `damping=0.7`; **pre-collection setup tasks:** `apt upgrade` to `ur_robot_driver` 2.13.0, document warm-up SOP (10 min arm idle in HOVER), establish single-source-of-truth `set_payload` at bringup only, reconcile `pyproject.toml` numpy pin.
**Addresses (FEATURES.md):** All P1 lifecycle, telemetry, and config-schema features.
**Avoids (PITFALLS.md):** #1 hand-on-zero, #2 temperature drift (warm-up SOP), #3 controller-switch race, #5 gain instability, #6 damping defaults, #16 set_payload bug, #17 trajectory queue residual, #18 re-zero discontinuity, #15 protective-stop handling.

### Phase 2: 20-Episode FMB1 Data Collection (don't build the dashboard against synthetic data)

**Rationale:** Architecture and pitfalls research both flag "build dashboard against synthetic data" as a top trap. The shapes of real F/T traces (especially whether `inverted_u_yellow` is monotonic or snap-fit, whether `line_green` shows multi-point contact dynamics) are unknown until collection happens, and a dashboard built against fictional traces visualizes a fiction.
**Delivers:** 20 episodes (5 × 4 FMB1 objects) including ≥1 `intentional_misalignment` and ≥1 `failure_mode_demo` per object; aborted episodes preserved (not deleted), tagged with reason; operator user_notes captured at every episode end; `post_zero_bias` and uptime logged per episode; collection done in `--collect` mode using the wrapper's `MANUAL_GUIDED` fallback config (no autonomous termination — operator-SIGTERM only).
**Addresses (FEATURES.md):** P1 data-collection features; calibration ramp.
**Avoids (PITFALLS.md):** #8 operator-force contamination (hands-off window enforced by Phase 1 wrapper), #9 demo selection bias (mandatory quotas + abort preservation).

### Phase 3: Static HTML Dashboard (built against the real shapes from Phase 2)

**Rationale:** Now there's real data to develop against. Dashboard is the parameter-derivation surface — it's where the operator stares at F-vs-Z phase plots and types values into YAMLs. Built too early it visualizes fiction; built too late it blocks Phase 4.
**Delivers:** `tools/analyze_inserts.html` — single-page, Plotly.js + PapaParse from CDN, no build step; single-episode view (F vs t / T vs t / Z vs t synced cursors, F-vs-Z phase plot, 3D trajectory, event-marker vertical lines, metadata panel); cross-episode overlay with **time-alignment on first-contact event** (not absolute t=0); per-object signature card auto-computing median Fz, peak |T|, lateral travel, descent duration **using only the hands-off window samples** (not full episode); auto-detection of `signature_type` (monotonic vs snap-fit) from F-vs-Z loop shape; segmented stats partitioned on re-zero events; built with `scattergl` + decimation from day 1, validated with synthetic 100-episode load test.
**Addresses (FEATURES.md):** All P1 dashboard features.
**Avoids (PITFALLS.md):** #14 dashboard memory bloat, #18 re-zero discontinuity, #8 operator-force contamination (window-restricted stats), #2 temperature drift (gray out cross-session overlays).

### Phase 4: Algorithm Derivation — Per-Object YAMLs + Termination Criterion (the deliverable)

**Rationale:** Termination criterion is a project deliverable, not an input — three sources independently confirmed this. The dashboard's signature cards feed the YAML values; the F-vs-Z phase plots and signature_type detection feed the termination combinator choice. This phase is *where the research happens* — the operator stares at overlays, picks rules, validates them.
**Delivers:** `primitives/insert_configs/u_brown.yaml`, `u_orange.yaml`, `line_green.yaml`, `inverted_u_yellow.yaml` (defaults + per-object overrides only); per-object termination block with combinator (`{ primary, secondary, must_agree: true }` or `{ combine: weighted }`); per-object `signature_type`, `peg_count`, `force_mode_type` explicit; documented "tuning a new part in 30 min" SOP that fixes the parameter-tuning order (gain/damping → selection_vector from observed compliant axes → commanded_Fz from hands-off window median → speed_limit last); decision documented with evidence (which combinator/thresholds and why).
**Addresses (FEATURES.md):** P1 parametric algorithm + termination criterion deliverable.
**Avoids (PITFALLS.md):** #4 selection_vector confusion, #7 Z-reached false-positive, #10 parameter coupling, #12 snap-fit inversion, #13 multi-peg uneven seating.

### Phase 5: Dispatcher + Integration (the very last step)

**Rationale:** The integration edit at `translate_object.py:1085` is the *last* step. Until the dispatcher works standalone (operator can run `compliant_insert_episode.py --object-name u_brown` and get a successful insert), do not redirect `translate_object.py` to it — otherwise a broken dispatcher silently breaks the whole `--insert` flow including ablations.
**Delivers:** `primitives/compliant_insert_episode.py` dispatcher (lazy YAML lookup by exact filename, never glob-all; unknown-object fallback prints `__RESULT_JSON__` with explicit `hint` field naming the exact `--collect` command to run; in-process call to wrapper); `translate_object.py:1085` script_path edit (one line); `.gitignore` entries for `logs/insert_*.csv` + `logs/insert_*.meta.json`; vision-lockout flag set during ACTIVE phase (MCP server context); end-to-end ablation validation (`recording_dryrun_real_u_brown.yaml` completes insert leg autonomously).
**Addresses (FEATURES.md):** P1 integration features.
**Avoids (PITFALLS.md):** #19 AprilTag re-emergence override; broken-dispatcher-breaks-ablations failure mode.

### Phase 6: Generalization Validation (≥5 consecutive autonomous successes, not single-success)

**Rationale:** Validation gate is strengthened — ≥5 consecutive autonomous successes per object, not single-success. "It worked once" is a documented top failure mode (Pitfall #20). This phase also validates the "tuning a new part in 30 min" SOP from Phase 4 against a part from a *second* assembly using config-only changes — that's the generalization claim.
**Delivers:** Per-object autonomous success rate (≥5 consecutive, no operator intervention) for each FMB1 part, logged to `logs/validation_*`; one part from a second assembly tuned via the documented 30-min SOP, no algorithm changes; per-object success-rate dashboard view; validation failures preserved (not silently retried); milestone exit criteria documented with evidence.
**Addresses (FEATURES.md):** Generalization validation, validated requirements promotion.
**Avoids (PITFALLS.md):** #20 single-success deployment, #11 classifier overfit (heuristics-first discipline maintained).

### Phase Ordering Rationale

- **Why Phase 1 before everything:** Schema is the data contract. Wrong schema = re-collect demos = expensive. Wrapper is the only schema writer, so it must consume parametric inputs from day one (do not hard-code values "to make parametric later" — that produces two parallel implementations).
- **Why Phase 2 before Phase 3:** Building a dashboard against synthetic data is a documented trap (architecture research) — it produces a dashboard that visualizes a fiction. Real F/T shapes are needed.
- **Why Phase 3 before Phase 4:** The per-object YAML values come from staring at dashboard signature cards. Without the dashboard, parameter derivation is a Jupyter scratchpad with no visual cross-episode overlay.
- **Why Phase 4 before Phase 5:** YAMLs must exist (and be validated standalone) before the dispatcher routes to them. Otherwise a missing YAML breaks the integrated flow at the worst time.
- **Why Phase 5 before Phase 6:** Generalization validation runs through the integrated `translate_object.py --insert` path (matches the deployed surface the LLM agent calls). Validating only the standalone wrapper would miss integration regressions.
- **Phases 1 and 2 can interleave heavily** in practice: each demo session reveals wrapper bugs (missed event_marker rows, bad CSV flush on SIGTERM) and fixing them is part of Phase 1, not a separate "polish" pass. The phase boundary is conceptual; execution is iterative.

### Research Flags

Phases likely needing deeper research during planning:

- **Phase 4 (Algorithm Derivation):** This phase *is* research — the termination criterion answer is unknowable until Phase 2's data exists. Plan for `/gsd-research-phase` before writing each per-object YAML. Specifically: snap-fit detection logic for `inverted_u_yellow`, multi-peg torque thresholds if any FMB1 part turns out to be multi-peg, and the heuristic-vs-classifier comparison gate (held-out-object validation, ≥15pp lift required).
- **Phase 6 (Generalization Validation):** The "30-min SOP" claim is a research deliverable. Plan a `/gsd-research-phase` to design the held-out-part validation protocol (what counts as "no algorithm change" — is config-only with new termination thresholds permissible?).

Phases with standard patterns (skip research-phase):

- **Phase 1 (Schema + Wrapper):** Fully specified by STACK + ARCHITECTURE + PITFALLS research; pre-collection setup tasks enumerated. Standard ROS2 + rclpy patterns.
- **Phase 2 (Data Collection):** Process work, not research. Enforce protocol from PITFALLS.
- **Phase 3 (Dashboard):** Plotly.js + PapaParse + `scattergl` patterns are well-documented (verified in STACK.md). Decimation and lazy-load patterns specified in PITFALLS #14.
- **Phase 5 (Dispatcher + Integration):** Mechanical integration; one-line edit at `translate_object.py:1085`, fallback message format specified in ARCHITECTURE.md.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Live-verified on operator's system (`ros2 interface show`, `ros2 topic hz`, `apt list --installed`); CDN versions checked against official releases pages; `ur_robot_driver` 2.13.0 release date confirmed. |
| Features | MEDIUM-HIGH | Table stakes drawn from converging literature (NIST/Falco, Schimmels, Suarez-Ruiz/Pham, Chen 2016 AIM, DexForce, FILIC); some operator-UX features (pause/resume, default suggestions in user_notes) flagged LOW confidence inline. |
| Architecture | HIGH | Component boundaries / file layout / build order validated against existing repo conventions and the working `compliant_insert.py` placeholder. The runtime classifier question (MEDIUM) is intentionally deferred per PROJECT.md. |
| Pitfalls | MEDIUM-HIGH | UR/Robotiq specifics HIGH (official docs + forum reports including the documented 5.4.x `set_payload`/`zero_ftsensor` bug); small-data classifier guidance MEDIUM; some operator-process pitfalls drawn from LfD survey literature MEDIUM. |

**Overall confidence:** MEDIUM-HIGH. The stack and architecture are solid (live-verified). The features and pitfalls are well-supported but include some inferences from literature that need empirical confirmation against the actual FMB1 parts. The biggest unknown — termination criterion — is *expected* to be unknown; that's why it's a deliverable.

### Gaps to Address

- **`pyproject.toml` numpy pin vs runtime numpy 2.2.6 inconsistency:** Either lift the pin (recommended for 2026) or downgrade. Reconcile in Phase 1; the wrapper as scoped doesn't use numpy-2-only API, so it's not a blocker but should be fixed before milestone closes.
- **In-process vs subprocess between dispatcher and wrapper:** Both viable; recommendation is in-process. Phase 5 should empirically pick whichever debugs more cleanly. If wrapper crashes leak rclpy state into the dispatcher, fall back to subprocess.
- **`--insertion-type {prismatic,legacy}` flag survival:** Architecture research suggests deprecating once the new path is validated (one milestone), since the dispatcher subsumes the family/parameter selection. Defer the deprecation decision to Phase 5 or Phase 6.
- **Whether any FMB1 part is actually snap-fit or multi-peg:** Unknown until Phase 2 data exists. Phase 1 schema must support both (`signature_type`, `peg_count` fields in YAML); Phase 3 dashboard must auto-detect; Phase 4 derives thresholds. Don't pre-specify.
- **Hands-off window duration (≥2 s recommended):** Empirical — operator may need adjustment after first 5 demos. Make it a config knob, not a hard-coded constant.
- **Statistical classifier vs hand-rules outcome:** Explicitly deferred per PROJECT.md to a post-data-collection phase. Affects only the analysis tooling, not the runtime architecture.

## Sources

### Primary (HIGH confidence)
- Live system verification on operator's UR5e (`ros2 topic hz`, `ros2 interface show`, `apt list --installed`), 2026-05-01
- [Universal_Robots_ROS2_Driver — humble branch](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/tree/humble) — driver/controller/SetForceMode (verified against installed 2.12.0)
- [UR ROS2 Driver releases](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/releases) — 2.13.0 release dates and notes
- [ur_controllers documentation](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_controllers/doc/index.html) — `gain_scaling`, `damping_factor`, force-mode parameters and stability warnings
- [URScript Dynamic Force Control](https://www.universal-robots.com/articles/ur/programming/urscript-dynamic-force-control/) — `force_mode` type 1/2/3 semantics
- [UR e-Series UR5e User Manual](https://s3-eu-west-1.amazonaws.com/ur-support-site/40974/UR5e_User_Manual_en_US.pdf) — protective stop behavior and recovery procedure
- [UR Support PDF: Understanding Protective Stops](https://s3-eu-west-1.amazonaws.com/ur-support-site/76519/Understanding%20Protective%20Stops.pdf) — protective stop fundamentals + 5 s cooldown
- [Robotiq FT-300 Sensor Manual](https://assets.robotiq.com/website-assets/support_documents/document/FT_Sensor_Instruction_Manual_PDF_20181218.pdf) — calibration drift, temperature, install-induced bias
- [Robotiq Knowledge: FT-300S operating and calibrating](https://blog.robotiq.com/knowledge/operation-and-calibration-of-the-ft-300s-5-1736280819067)
- [MDPI Sensors 2026: Experimental Evaluation of UR5e Collaborative Robot Force Control in Low-Force Applications](https://www.mdpi.com/1424-8220/26/5/1709) — empirical gain/damping recommendations
- [Plotly.js releases](https://github.com/plotly/plotly.js/releases) — v3.5.1 (May 2026) latest
- [Plotly.js dist README](https://github.com/plotly/plotly.js/blob/master/dist/README.md) — bundle sizes / CDN URLs
- [PapaParse 5.5.3 docs](https://www.papaparse.com/) — current version, FileReader integration
- [ROS2 Control: Controller Manager (Humble)](https://control.ros.org/humble/doc/ros2_control/controller_manager/doc/userdoc.html) — switch_controller semantics
- [ros2_controllers force_torque_sensor_broadcaster](https://control.ros.org/humble/doc/ros2_controllers/force_torque_sensor_broadcaster/doc/userdoc.html) — wrench publishing pattern
- [Plotly Community: Browser memory consumption advice](https://community.plotly.com/t/browser-memory-consumption-advice/83632)
- [Plotly.js issue #553: Performance with 180k+ datapoints](https://github.com/plotly/plotly.js/issues/553)
- [Snap-Fits Assembly Insertion Force Simulation Study (Springer)](https://link.springer.com/chapter/10.1007/978-981-15-9505-9_87) — snap-fit force signature
- [MDPI Sensors: Peg-in-Hole Two-phase F/T Sensor for Dual-arm](https://www.mdpi.com/1424-8220/17/9/2004) — multi-peg analysis, F/T threshold tuning fragility
- [Springer 2025: Advances in Robotic Peg-in-Hole Assembly Comprehensive Review](https://link.springer.com/article/10.1186/s10033-025-01349-w) — failure modes, jamming, wedging fundamentals
- NIST benchmarks: [Comparative Peg-in-Hole Testing](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=922206), [Benchmarking Protocols for Small Parts Robotic Assembly (Falco/Marvel et al.)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7537423/)
- Schimmels accommodation theory: [Force-Assembly with Friction (1994)](https://peshkin.mech.northwestern.edu/publications/1994_Schimmels_ForceAssembly.pdf), [Admittance matrix design (1992)](http://peshkin.mech.northwestern.edu/publications/1992_Schimmels_AdmittanceMatrixDesign.pdf)
- [MDPI Robotics: Learning from Demonstrations Survey](https://www.mdpi.com/2218-6581/11/6/126) — operator bias / demo selection bias as known top-three failure mode
- PROJECT.md — authoritative for milestone scope, constraints, design decisions

### Secondary (MEDIUM confidence)
- [UR Forum: 5.4.x set_payload and zero_ftsensor](https://forum.universal-robots.com/t/5-4-x-rtde-set-payload-and-zero-ftsensor/5146) — payload-zero coupling bug
- [UR Forum: zero_ftsensor() precision](https://forum.universal-robots.com/t/precision-of-zero-ftsensor/42537) — single-sample limitation
- [UR Forum: Force mode to Protective Stop](https://forum.universal-robots.com/t/force-mode-to-protective-stop/15664) — common protective stop triggers
- [UR Forum: Dashboard restart without teach pendant](https://forum.universal-robots.com/t/dashboard-restart-program-after-safety-mode-violation-without-teach-pendant/2038) — Local mode recovery limitations
- [MoveIt2 issue #450: ros2_control/Servo race](https://github.com/moveit/moveit2/issues/450) — controller-switch race patterns
- [Suarez-Ruiz/Pham — A framework for fine robotic assembly (ICRA 2016 / Science Robotics 2018)](https://www.science.org/doi/10.1126/scirobotics.aat6385)
- [Chen 2016 AIM: Teach Industrial Robots Peg-Hole-Insertion by Human Demonstration](https://wjchen84.github.io/publications/C2016_AIM.pdf)
- [DexForce: Extracting Force-informed Actions from Kinesthetic Demonstrations (arXiv 2501.10356)](https://arxiv.org/html/2501.10356v1)
- [FILIC: Dual-Loop Force-Guided Imitation Learning (arXiv 2509.17053)](https://arxiv.org/html/2509.17053)
- [Feature-Based Compliance Control for Peg-in-Hole (arXiv 2103.16003)](https://arxiv.org/pdf/2103.16003)
- [Variable Admittance + RL for Peg-in-Hole (MDPI 2025)](https://www.mdpi.com/2076-3417/15/4/2143)
- [MDPI Robotics: Practical Roadmap to LfD for Robotic Manipulators](https://www.mdpi.com/2218-6581/13/7/100) — small-dataset pitfalls; case for heuristics
- [Milvus: Handling overfitting in small datasets](https://milvus.io/ai-quick-reference/how-do-you-handle-overfitting-in-small-datasets) — small-N overfit risks

### Tertiary (LOW confidence — flagged inline in source files)
- Pause/resume mid-ACTIVE feature value (FEATURES.md) — may be unnecessary
- Rosbag parallel logging exclusion (FEATURES.md) — weak anti-feature; could legitimately be added as parallel logger
- Operator-UX features (default suggestions, terminal F/T display) — drawn from general LfD UX heuristics, not project-specific validation

---
*Research completed: 2026-05-01*
*Ready for roadmap: yes*
