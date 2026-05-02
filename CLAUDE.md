<!-- GSD:project-start source:PROJECT.md -->
## Project

**Compliant Insertion Studio**

A data-collection wrapper, analyzer dashboard, and **parametric peg-in-hole policy** for force-compliant assembly inserts on a UR5e + Robotiq 2F-85, replacing the current broken `prismatic_peg_insertion.py` real-mode insert path. Operator runs guided demonstrations; the system records F/T + pose telemetry per episode; analysis surfaces per-object parameters (axis-wise compliance, force levels, termination criteria, retry behavior) for a single universal insert algorithm parameterized differently per part. Proof-of-concept target: FMB1 assembly (u_brown, u_orange, line_green, inverted_u_yellow); design must generalize to a second assembly without rework.

**Core Value:** **Replace the failing `--insert` real-mode path with a force-compliant insert primitive that works reliably across every FMB1 part and is a one-config-file extension to any new part.** If everything else slips, the FMB1 inserts must complete autonomously end-to-end.

### Constraints

- **Tech stack**: ROS2 Humble, Python 3.10, `rclpy`, `ur_robot_driver`, Robotiq 2F-85 driver. Force mode via `ur_msgs/srv/SetForceMode` only — no direct URScript injection.
- **Hardware**: One physical UR5e + Robotiq 2F-85 + workspace cameras. Single-instance, no parallel data collection.
- **Operator time**: ~5 demos per object × 4 objects = ~30–60 min collection sessions. Design must accommodate iterative collection across multiple sessions.
- **Pendant mode**: Local mode preferred (operator retains manual control). Dashboard service calls (`--recover`, etc.) cannot be automated.
- **Compliance**: Force-mode commanded wrench must stay gentle (≤ 5 N) by default — gear / part / fixture damage limits.
- **Safety**: Operator's hand near robot during demos. Must always be able to interrupt cleanly (SIGTERM cleanup must reliably switch back to position controller and stop force mode).
- **Existing API**: `translate_object.py --insert --mode real` flow must be preserved as the integration point. The new system replaces what runs *inside* that call, not the call signature.
- **Data location**: All logs to `logs/insert_*.csv` + `.meta.json` in repo root. Dashboard reads from there. Not committed (binary-ish, large) — `.gitignore` entry needed.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

## Stack Philosophy
- **No backend.** Dashboard is a `file://` HTML opened in a browser. No Flask, no FastAPI, no Node.
- **No framework.** No React/Vue/Svelte for the dashboard, no episode-lifecycle library for the recorder. Bash launcher → Python script → static HTML.
- **No learning frameworks.** No PyTorch, TensorFlow, JAX, or scikit-learn. Statistical classification (if it's needed at all after the data is in) lives in ~50 lines of NumPy.
- **No new ROS2 packages beyond what's installed.** The driver, controllers, and msgs already on the system cover 100% of the force-mode + telemetry surface.
## Recommended Stack
### Core Technologies
| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| ROS2 Humble | LTS | Middleware, controller manager, driver substrate | Already installed; matches operator's UR5e setup. No rationale to migrate to Jazzy/Rolling for this milestone. |
| Python | 3.10.12 | Episode wrapper, CSV writer, signal handlers, force-mode RPC client | Default for ROS2 Humble (rclpy is Python 3.10). Already used throughout `primitives/`. |
| `rclpy` | 3.3.15 (installed) | ROS2 client lib for the episode wrapper node | Standard. Existing primitives all use it. No alternative. |
| `ur_robot_driver` | 2.12.0 (installed) → 2.13.0 available | Driver, controller bringup, force_mode_controller | 2.13.0 (released April 2026 for Humble) ships F/T frame bugfixes — **recommend upgrade before data collection** (see Pitfall: stale F/T frame). |
| `ur_msgs` | 2.3.0 (installed) → 2.4.0 available | `SetForceMode.srv` definition | Verified live: `ros2 interface show ur_msgs/srv/SetForceMode` matches the API the existing `compliant_insert.py` already calls. No upgrade required for Humble. |
| `geometry_msgs` | (ROS2 stock) | `WrenchStamped`, `PoseStamped` for telemetry subscriptions | Already a transitive dep. |
| `std_srvs` | (ROS2 stock) | `Trigger` for `/zero_ftsensor` and `/stop_force_mode` | Already a transitive dep. |
| `controller_manager_msgs` | 2.48.0 (installed) | `SwitchController.srv` for entering/leaving force_mode_controller | Verified live; existing `compliant_insert.py` already uses this. |
| `PyYAML` | 6.0.2 (installed) | Per-object YAML configs (`primitives/insert_configs/<object>.yaml`) | Standard, ships with most distros, zero-dep. Project already uses YAML for ablation configs. |
| `numpy` | 2.2.6 (installed) | Per-axis error math in episode wrapper, offline feature extraction | **Note**: `pyproject.toml` pins `numpy<2`, but the runtime has 2.2.6 — likely overridden by ROS apt packages. Episode wrapper code should stay on numpy-1-compatible API surface (avoid `np.bool8`, `np.float_`, etc.) to keep the pyproject pin honest if that environment is reconstructed. |
| Plotly.js | **3.5.1** (CDN) | All dashboard plots: F vs t, T vs t, Z vs t, F-vs-Z phase, 3D trajectory | Single-file, zero-build, full scientific chart vocabulary including 3D trajectory + multi-axis sync. The exact tool the project description names. |
| PapaParse | **5.5.3** (CDN) | In-browser CSV parsing of `logs/insert_*.csv` | RFC-4180 correct, streaming, zero-dep, FileReader-friendly. ~45 KB minified. The standard for browser CSV in 2026. |
### Supporting Libraries
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `transforms3d` or `scipy.spatial.transform.Rotation` | scipy 1.11+ (installed) | Quaternion → euler for `dyaw/dpitch/droll` per-sample error in CSV writer | Use `scipy.spatial.transform.Rotation` (scipy already in `pyproject.toml`); avoid adding `transforms3d` as a new dep. |
| `tf2_ros` | (ROS2 stock) | Transform `task_frame` pose into `base_link` for `SetForceMode` request, transform wrench frames if needed | Only if non-`base_link` task frames are used. Default to `base_link` task frame and skip TF entirely. |
| `pandas` | 2.2.3 (installed) | **Optional**: offline cross-episode aggregation in a Jupyter scratchpad if the dashboard view turns out insufficient | Do **not** ship as a runtime dep of the analyzer. Browser-side aggregation in JS is sufficient for 20 episodes × ~30 s × 500 Hz = ~300k rows total. Pandas stays as an "operator's notebook" tool. |
### Development Tools
| Tool | Purpose | Notes |
|------|---------|-------|
| `bash` launcher script | One-command "start an episode" entry point that wraps `ros2 run primitives compliant_insert_episode.py --object <name>` and prepares the log dir | Keeps operator's loop simple. ~30 lines. Living next to the primitive. |
| `ros2 service call` (CLI) | Manual zeroing / force-mode stop during debugging without restarting the episode wrapper | Not a dependency — just a workflow. |
| `rqt_plot` or `plotjuggler` | Live diagnostic during demo collection (operator can sanity-check F/T zeroed before pushing) | Both already typically installed with ROS2 Humble. **Do not embed plotjuggler into the workflow** — it is a sidecar diagnostic, not the dashboard. |
| Browser (Chromium/Firefox) | Render `tools/analyze_inserts.html` from `file://` | Modern browsers can read `file://`-relative CSVs via `<input type="file">` + FileReader, or via `fetch()` with `--allow-file-access-from-files` Chromium flag. Recommend the FileReader path (no flag, drag-and-drop). |
## Installation
# Recommended: bump UR driver to 2.13.0 for the F/T frame bugfix
# Python deps already in pyproject.toml — nothing to add.
# (PyYAML, numpy, scipy, pandas all present.)
# Dashboard CDN (no install — just <script src=...> in tools/analyze_inserts.html):
#   https://cdn.plot.ly/plotly-cartesian-3.5.1.min.js   (~1.4 MB min, ~463 KB gz)
#   https://cdn.plot.ly/plotly-gl3d-3.5.1.min.js        (only if 3D trajectory needed; 1.6 MB min)
#                                                        OR use full plotly-3.5.1.min.js (4.9 MB)
#   https://cdn.jsdelivr.net/npm/papaparse@5.5.3/papaparse.min.js
## Alternatives Considered
| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| **CSV + JSON sidecar** for telemetry | `rosbag2` (MCAP storage) | Use rosbag2 if you ever need to replay topics through the live ROS graph (e.g., for an offline planner test) or if topic count grows past ~10. For a 5-topic, post-hoc-analyzed kinesthetic demo, CSV + meta JSON is faster to write a dashboard against, easier to diff, easier for the operator to open in any tool. **Stick with CSV.** |
| **Static HTML dashboard** | Foxglove Studio | Foxglove if the operator needs live streaming during the demo (synced cursors across F/T + camera + URDF). Foxglove is overkill for offline per-episode review and locks the analyzer into a desktop app instead of a `file://` HTML the project explicitly asks for. |
| **Plotly.js** | Chart.js, ECharts, D3, uPlot | Chart.js can't do 3D trajectories. ECharts is heavier and SVG/Canvas-mixed. D3 is a primitive — you'd write Plotly yourself. uPlot is faster on huge series but lacks 3D. The PROJECT.md picks Plotly.js explicitly; corroborated by ecosystem analysis. |
| **PapaParse** | Native `String.prototype.split(',')` | Native split is fine for ~hundreds of rows but breaks on quoted fields (likely in `user_notes`) and can't stream. ~45 KB for correctness is the right trade. |
| **Rule-based heuristics** in NumPy | scikit-learn (KMeans/DecisionTreeClassifier) | Only if hand-derived rules over interpretable features (median Fz at success, |Tx|/|Ty| peak distribution, lateral travel during ACTIVE) **fail** to separate success/failure families across the 20-episode dataset. Even then, scikit-learn is a heavy import for one classifier — implement KMeans by hand in ~30 lines of NumPy first. The PROJECT.md flags this as an explicit research carve-out for **after** data is collected. **Do not add scikit-learn to dependencies in this milestone.** |
| **Bash launcher → Python script → HTML** (script-driven) | Episode-lifecycle framework (e.g., `behaviortree.cpp`, `py_trees`, `smach`) | A behavior tree makes sense when you have ≥4 nested decision branches in a state machine. The PROJECT.md lifecycle is `PRE → HOVER → ZERO → ACTIVE → DONE/ABORT` — five linear states with two terminal branches. Implement as a `Phase = Enum(...)` and `match self.phase:` blocks. Adding `py_trees` is more code, more concepts, more rebuilds, and zero capability gain. **Stay script-driven.** |
| **Single `compliant_insert.py` file with all phases inline** | Class hierarchy with `Phase` strategy objects | The current `~250 LOC` placeholder is a single-file script. The PROJECT.md asks for ~5 phases each ~30–80 LOC plus a CSV writer and signal handlers — totalling ~500 LOC. Single file is fine; a `phases/` directory of strategy objects is over-engineering for a five-state linear FSM. |
## What NOT to Use
| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **React / Vue / Svelte / Solid** for the dashboard | The dashboard is one page, one user, no routing, no server. Bringing a framework adds a build step (Vite/webpack), a `node_modules/`, and a dev server, and forces you to either ship a `dist/` or run the build before each demo. Plotly + a `<div id="...">` does this in one HTML file. | Plain HTML + vanilla JS + Plotly.js + PapaParse, all from CDN. |
| **PyTorch / TensorFlow / JAX** | Zero learning happens at runtime. Even offline, 20 episodes is far below the data threshold where these frameworks earn their import-time cost. | NumPy for feature extraction; if a classifier is needed, hand-rolled KMeans or a 5-line decision tree. |
| **scikit-learn** | Same as above. The "is it KMeans or hand-rolled rules?" decision is deferred to post-collection per PROJECT.md, and even if KMeans wins, k=2 or k=3 over 6-feature episode vectors is ~30 LOC of NumPy. Adding sklearn pulls in scipy.sparse, joblib, threadpoolctl, etc. | NumPy. |
| **rosbag2 for primary telemetry** | The operator's analysis loop is "drop a CSV in `logs/`, refresh the browser." rosbag2 inserts a deserialization step (and a Python ROS context, or a third-party reader) between the data and the eyeballs. Also, rosbag2's MCAP files don't diff cleanly in git, can't be opened in Excel, and require a tool to inspect. | Plain CSV (one row per F/T sample) + sidecar JSON (per-episode metadata). |
| **Flask / FastAPI / aiohttp** for the dashboard backend | There is no backend. The browser reads CSV files from disk via `<input type="file">` or local fetch. | Static HTML loaded via `file://`. |
| **Direct URScript injection** for force mode | `urscript_interface` exists in the driver, but the project's constraint section explicitly says "Force mode via `ur_msgs/srv/SetForceMode` only — no direct URScript injection." Mixing modalities also splits the controller-manager state model. | `force_mode_controller`'s `~/start_force_mode` and `~/stop_force_mode` services. |
| **`dashboard_client/recover` for fault recovery** | Constraint: pendant is in Local mode. Dashboard service calls fail. | Operator manually clears protective stops on the pendant. Wrapper code must not assume `recover` works. |
| **Ad-hoc per-object Python scripts** (one file per part) | Project decision: "Parametric universal algorithm + per-object YAML, not per-object scripts." Already an enforced design. | Single `compliant_insert.py` + `primitives/insert_configs/<object>.yaml`. |
| **`KEEP_ALL` history QoS** on the `/wrench` subscription | The broadcaster publishes at **500 Hz** (verified live on operator's robot — see Pitfall). Keeping all messages buffers unbounded if the wrapper falls behind. | Match the broadcaster's QoS: `RELIABLE` + `KEEP_LAST(1)` + `VOLATILE`. The CSV writer reads the latest sample at its own (e.g., 100 Hz) tick, not every wrench message. |
| **`numpy 2.x` API features in episode-wrapper code** if `pyproject.toml` numpy<2 pin matters | The repo pins `numpy<2` in `pyproject.toml` but the apt-installed ROS Python env has 2.2.6. Code that runs in both contexts will silently break under numpy 1.x if it uses 2.x-only API (`np.unique(..., equal_nan=...)`, e.g.). | Either lift the `numpy<2` pin in `pyproject.toml` (recommended — numpy 2 is the system default by 2026), or stick to the numpy-1-compatible subset in episode-wrapper code. |
## Stack Patterns by Variant
- Subsample at the wrapper, not at the broadcaster.
- The 500 Hz wrench rate is fine to read but doesn't need to be logged at full rate. Log at 100 Hz (every 5th message) to keep CSVs ~30 MB per episode instead of ~150 MB. Operator can later upsample if needed by re-running with logging at 500 Hz.
- Move per-episode logs to a per-operator subdir: `logs/<operator>/insert_<object>_<ts>.csv`.
- No infrastructure change required — dashboard's auto-discovery scans recursively.
- Drop the gl3d bundle and switch to `plotly-cartesian-3.5.1.min.js` (1.4 MB → 463 KB gzipped). Six 2D plots cover everything else.
- Add a 5-line `python3 -m http.server 8000 --directory <repo>` wrapper to `tools/serve_dashboard.sh`. Still no backend, just a static file server. Browser can then `fetch('/logs/')` and parse a directory listing.
- Do **not** add Flask. The bar for a dynamic backend in this project is "you need to call a Python function from JS," and you don't.
## Version Compatibility
| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `ur_robot_driver` 2.12.0 | `ur_msgs` 2.3.0 | Verified live on system. Force mode works. |
| `ur_robot_driver` 2.13.0 | `ur_msgs` 2.4.0 | **Recommend upgrade** for F/T frame bugfix in 2.13.0. ABI for `SetForceMode.srv` is unchanged between 2.3.0 and 2.4.0. |
| Plotly.js 3.5.1 | Modern browsers (Chromium 90+, Firefox 88+) | v3 dropped IE/legacy support. Operator runs Linux with current Firefox/Chromium — no concern. |
| PapaParse 5.5.3 | Modern browsers + FileReader | No version-pinning concerns. |
| Python 3.10 | rclpy 3.3.x (Humble) | Default for Humble. Do not upgrade Python 3.11+ — Humble's binary `rclpy` won't load. |
| `numpy 2.2.6` (installed) | `pyproject.toml` says `numpy<2` | **Inconsistency.** Either lift the pin (recommended for 2026) or downgrade. The episode wrapper as scoped doesn't use numpy-2-only API, so it's not a blocker, but the pin should be reconciled before the milestone closes. |
## Verified Live on the Operator's System
- `ros2 interface show ur_msgs/srv/SetForceMode` — request fields match what `compliant_insert.py` calls; defaults (`type=2 NO_TRANSFORM`, `damping_factor=0.025`, `gain_scaling=0.5`) are the same the existing placeholder uses.
- `ros2 topic info /force_torque_sensor_broadcaster/wrench` — type `geometry_msgs/msg/WrenchStamped`, QoS `RELIABLE / KEEP_LAST(1) / VOLATILE`.
- `ros2 topic hz /force_torque_sensor_broadcaster/wrench` — **~500 Hz** sustained (not 125 Hz — important for CSV-writer sizing).
- `python3 -c "import numpy, pandas, yaml; print(...)"` — numpy 2.2.6, pandas 2.2.3, PyYAML 6.0.2 all present.
- `apt list --installed | grep ros-humble` — driver 2.12.0, ur_msgs 2.3.0, controller_manager 2.48.0 confirmed.
## Sources
- [Universal_Robots_ROS2_Driver — humble branch](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/tree/humble) — current Humble release verified, 2.13.0 (April 2026) — **HIGH** (official repo, version verified against locally installed 2.12.0).
- [UR ROS2 Driver releases page](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/releases) — release dates and notes for 2.12.0 → 2.13.0 — **HIGH**.
- [ur_controllers documentation](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_controllers/doc/index.html) — `SetForceMode` interface, services exposed by `force_mode_controller` — **HIGH** (official UR docs, cross-checked against `ros2 interface show`).
- [ros2_controllers force_torque_sensor_broadcaster](https://control.ros.org/humble/doc/ros2_controllers/force_torque_sensor_broadcaster/doc/userdoc.html) — wrench publishing pattern, `~/zero_ftsensor` service location — **HIGH**.
- [Plotly.js releases](https://github.com/plotly/plotly.js/releases) — v3.5.1 (May 2026) latest — **HIGH** (official releases page).
- [Plotly.js dist README](https://github.com/plotly/plotly.js/blob/master/dist/README.md) — bundle sizes, CDN URL pattern — **HIGH**.
- [PapaParse 5.5.3 docs](https://www.papaparse.com/) — current version, FileReader integration — **HIGH**.
- [rosbag2 + MCAP guidance](https://mcap.dev/guides/python/ros2) — used for the "alternatives considered" comparison; not adopted — **MEDIUM** (official MCAP docs, but our use case doesn't apply).
- Live system verification (`ros2 topic hz`, `ros2 interface show`, `apt list --installed`) on operator's UR5e setup, `2026-05-01` — **HIGH** (empirical, on the actual hardware).
- PROJECT.md `/home/aaugus11/Documents/ros-mcp-server/.planning/PROJECT.md` — constraints, scope, design decisions — **HIGH** (authoritative for this milestone).
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Key Rules — every agent must follow
- **Research before code**: spend 5–15 min on `WebSearch` + `WebFetch` before any non-trivial deliverable. Find existing tools, papers, vendor docs, and reference implementations. Don't reinvent.
- **Clone references locally**: useful repos go to `_references/repos/` via `git clone --depth 1`; useful articles get saved as markdown to `_references/articles/`. Both directories are gitignored.
- **Honesty over confidence**: if you don't know something, say so. Do not write SOPs or specs based on shallow grounding. Pause and research.
- **Per-piece copy/modify/write-fresh decision**: when borrowing from a reference, decide explicitly per piece — not whole-repo. Credit sources in code comments.
- **Inline-default working pattern**: do work in the main conversation so the operator can see and intervene. Subagents only when work is genuinely independent + parallel + would pollute main context (the 4-researcher init was the right use case; routine code/planning is not).
- **Phase boundaries are guidance, not gates**: if requirements from later phases are coupled to current work, finish them together and mark them complete now. Update REQUIREMENTS.md traceability to record where each requirement actually completed.
- **All project deliverables under `compliant_insertion_studio/`**: the entire subsystem is a self-contained folder droppable into other robotics projects. New code defaults to that folder unless modifying an existing host-repo primitive.
- **F/T calibration is three layers**: foundational payload calibration (per-mount, one-time, sets `set_target_payload`), session-level smoke test (per-session, confirms sensor health), per-pose `zero_ftsensor` (immediately before force mode). Per-pose alone does not substitute for foundational.
- **Pendant in Local mode**: do not write code requiring Remote mode or `dashboard_client/recover`. Recovery from protective stops is manual.
- **Force-mode wrench ≤ 5 N default**: higher only with explicit per-task override and operator awareness.
- **SIGTERM cleanup must be idempotent and reliable**: wrapper must reach safe-state DONE exit even if force mode is already stopped or controller switch fails partway.
- **Hands-off window during F/T zero**: operator confirmation gate before zero, +1 s post-zero drift check, no operator load during baseline windows.
- **Safe height before move_home when holding a part**: direct `move_home` plans straight-line trajectories that ignore inserted bases.
- **Don't commit unless explicitly approved**, no `Co-Authored-By` lines (operator global rule).
- **`_references/` and `compliant_insertion_studio/logs/` are gitignored**: never commit reference repos or telemetry.
- **Ask the operator before**: adding a new top-level dependency, writing > 200 LOC without checkpoint, performing any robot motion, modifying primitives outside `compliant_insertion_studio/`, departing from a documented decision in PROJECT/REQUIREMENTS/ROADMAP.
## Anti-patterns — explicit "don't"
- Writing 100-line SOPs based on extrapolation rather than documented procedures
- Confusing similar-sounding concepts (e.g., "calibration" vs "bias offset" — they are different things on a UR5e)
- Citing thresholds that you guessed at as if they came from research
- Shipping a deliverable without distinguishing "verified empirically" from "research-backed" from "extrapolated"
- Spawning a subagent for routine work
- Splitting coupled work across phase boundaries to honor the convention rather than because the work is actually separable
- Putting project deliverables outside `compliant_insertion_studio/`
- Treating `zero_ftsensor` as a substitute for correct payload identification
- Code that assumes Remote pendant mode or dashboard recovery automation
## Decision matrix — copy / modify / write-fresh
| Decision | When to use |
|---|---|
| **Copy (lift file, attribute source)** | Code is in our language + framework + license is compatible + fits our architecture as-is |
| **Modify after copying** | Mostly fits, needs only surface tweaks (paths, message types, function names) |
| **Write fresh from algorithm/pattern** | Reference is in different language/framework/era, but the algorithm or pattern is sound — translate the *idea*, not the lines |
| **Skip** | Reference is well-known but doesn't fit our stack/scope (e.g., a node requiring an accelerometer we don't have) |
## When to use which calibration layer
| Layer | Frequency | What it does | Trigger |
|---|---|---|---|
| **Foundational** payload calibration | Per gripper mount (one-time) | Recovers mass + CoG + bias via Kubus 2007 LSQ; outputs `set_target_payload(mass, cog)` for bringup launch | New gripper / new jig / sensor remount / orientation-dependent bias observed |
| **Session** F/T smoke test | Per session | Zero + 5 s hold + bias verification in known neutral pose; pass/fail per axis | Start of session / after protective stop / after physical bump / when force-mode misbehaves |
| **Per-pose** `zero_ftsensor` | Immediately before each force-mode entry | Single-pose bias subtraction | Inside the wrapper's PRE phase, after smoke passes |
## Detailed rationale (humans only — auto-summarizer drops most of what follows)
### Research before code — procedure
- Search 2–4 query variations (vendor docs, GitHub, papers, community forums)
- Identify candidate repos / articles
- Clone repos into `_references/repos/`, save articles to `_references/articles/`
- Read enough to make an honest copy/modify/write-fresh decision per piece
- Only then write code
### Phase boundaries — finish coupled work together (rationale)
- Project-defined deviation from conventional GSD
- If requirements from later phases are tightly coupled to current work, finish them together and mark them complete now, regardless of which phase they nominally belong to
- Update REQUIREMENTS.md traceability when this happens to record where each requirement actually completed
- ROADMAP.md phase-completion still requires *all* of that phase's owned requirements to be done — but those done early count
- Do **not** intentionally pull future work forward to "save time" — only collapse when the coupling is real and finishing-now is materially cheaper than finishing-later
- This avoids the conventional-GSD waste of revisiting touched files in a later phase to make small additions that could have been done in one pass
### Folder structure rationale
- All Compliant Insertion Studio code lives under `compliant_insertion_studio/` (see REQUIREMENTS.md → "Project Layout" for full tree)
- The entire subsystem is droppable into other robotics projects with a one-line edit at the host's `translate_object` equivalent
- Implication: when adding new code for this project, default to placing it under `compliant_insertion_studio/`. Only put code in `primitives/` (or elsewhere) if you're modifying *existing* primitives or fixing a bug in the host repo unrelated to this project's deliverables
### Calibration hierarchy detail
- **Foundational** (per-mount): payload mass + CoG via `set_target_payload()` after running calibration script (or URCap Measure wizard). If this is wrong, nothing downstream is right — orientation-dependent bias persists no matter how often you `zero_ftsensor`
- **Session-level**: F/T smoke test confirms the sensor's bias is close to zero in a known pose. If this fails, foundational is suspect
- **Per-pose**: `zero_ftsensor` before force mode subtracts residual bias. Fast, cheap, but does NOT substitute for foundational
### Inline default rationale
- The GSD `workflow.research / plan_check / verifier / code_review / pattern_mapper` toggles are all `false` in `.planning/config.json` to enforce inline-default
- If you find yourself wanting to spawn an agent for routine work, ask first
- The 4-researcher init at project start was right because the 4 surveys were genuinely independent (each researcher knew nothing about the others), parallel (5 min instead of 20), and produced artifacts (the .md files) rather than intermediate chatter that would have polluted the main context
### Safety conventions detail
- Operator's pendant preference is Local mode (manual control retained)
- Don't call `dashboard_client/recover`, `dashboard_client/play`, or other Remote-mode services
- Recovery from protective stops is manual on the pendant
- After protective stop, operator clears it; only then can your code resume
- Force-mode wrench limits exist because gear/part/fixture damage is the binding constraint, not benchmark performance
- SIGTERM cleanup is operator trust. If the wrapper's cleanup is unreliable, operators will hesitate to use SIGTERM and instead try to interrupt other ways (which may leave the robot in force mode)
### Commit discipline detail
- Pre-existing-codebase fixes (bug fixes outside the project's scope) get their own commit, separate from project deliverables
- Project commits should reference requirement IDs they complete (e.g., `feat(WRAP-01..05): episode wrapper PRE/HOVER/ZERO phases`)
- The `_references/` folder is gitignored — never committed
- Telemetry logs are gitignored — never committed
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
