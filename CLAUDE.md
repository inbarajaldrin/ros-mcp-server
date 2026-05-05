<!-- GSD:project-start source:PROJECT.md -->
## Project

**Compliant Insertion Studio**

A data-collection wrapper, analyzer dashboard, and **parametric peg-in-hole policy** for force-compliant assembly inserts on a UR5e + OnRobot RG2, replacing the current broken `prismatic_peg_insertion.py` real-mode insert path. Operator runs guided demonstrations; the system records F/T + pose telemetry per episode; analysis surfaces per-object parameters (axis-wise compliance, force levels, termination criteria, retry behavior) for a single universal insert algorithm parameterized differently per part. Proof-of-concept target: FMB1 assembly (u_brown, u_orange, line_green, inverted_u_yellow); design must generalize to a second assembly without rework.

**Core Value:** **Replace the failing `--insert` real-mode path with a force-compliant insert primitive that works reliably across every FMB1 part and is a one-config-file extension to any new part.** If everything else slips, the FMB1 inserts must complete autonomously end-to-end.

### Constraints

- **Tech stack**: ROS2 Humble, Python 3.10, `rclpy`, `ur_robot_driver`, OnRobot RG2 driver. Force mode via `ur_msgs/srv/SetForceMode` only — no direct URScript injection.
- **Hardware**: One physical UR5e + OnRobot RG2 + workspace cameras. Single-instance, no parallel data collection.
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
- **WRAP-VERIFY end-to-end PASSED on u_brown** (2026-05-03): FSM walks PRE → HOVER → ZERO → ACTIVE → DONE; 15,177 telemetry samples per run; CSV+meta JSON written to `compliant_insertion_studio/logs/`.
- **F/T payload calibration recovered** (2026-05-03): mass=2.1109 kg, CoG=[-0.0032, +0.0031, -0.0318]m. Pasted into bringup as `set_target_payload(2.1109, [-0.0032, 0.0031, -0.0318])`.
- **Reusable orchestrator script validated**: `compliant_insertion_studio/scripts/run_assembly_step.py` works for u_brown grasp_id=1. Other 3 FMB1 grasp_ids need 5-min verification before Phase 3 collection.
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
- **Two execution tracks — away-from-robot and at-robot**: every requirement is tagged either `[N]` (no-robot — can be done from anywhere with the codebase) or `[R]` (robot-required — needs the physical UR5e + bringup). When the operator is away from the robot, work the `[N]` track. When the operator is at the robot, work the `[R]` track and any `[N]` items needed to unblock it. **`.planning/TRACKS.md` is the live list of what's ready in each track right now.** Update it whenever a task transitions states (ready → in-progress → done, or new requirements get tagged).
- **Bring-up runbook**: `compliant_insertion_studio/docs/SETUP.md` is the full cold-start guide (repos, processes, per-task commands, troubleshooting). Read it before doing anything at-robot.
- **Phase 3 entry point**: `python3 -m compliant_insertion_studio.scripts.run_assembly_step --object-name X --base-name base1 --grasp-id N` runs the full canonical pick→rotate→place→regrasp→rotate→insert sequence. One CLI per assembly step. Use `--already-held --current-object-orientation QX QY QZ QW` to skip pick/regrasp.
- **Held-object pose chains, not reads**: when the gripper holds a part, NEVER read its pose from `/objects_poses_real` (camera occluded by gripper). Chain `current_object_orientation` from the previous primitive's `__RESULT_JSON__` output.
- **Strip ANSI before parsing ROS2 CLI output**: `ros2 control list_controllers` (and likely others) emit `\x1b[…m` color escapes even when stdout is a pipe. Use `re.sub(r'\x1b\[[0-9;]*m', '', line)` before tokenizing. Silently breaks naïve parsers.
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
- Killing the gripper bridge with `pkill -f gripper_control` (won't match — actual cmdline is `python3 /opt/ros/humble/bin/ros2 run …`). Use `pkill -f "socat.*ttyUR"` or `kill -9 <PID>` after `ps aux | grep gripper_control`.
- Restarting socat-using processes without 5+ second wait between kill and respawn (PTY/termios race produces `(22, 'Invalid argument')` on next pyserial open).
- Treating `translate_object --insert` as the insert path. The new `compliant_insert` wrapper is the replacement (FSM: PRE→HOVER→ZERO→ACTIVE→DONE). The legacy CLI doesn't split HOVER from ACTIVE.
- Launching primitives subprocesses by script path (`python3 primitives/move_to_safe_height.py`). Use module mode (`python3 -m primitives.move_to_safe_height`) — the script-path form fails with ModuleNotFoundError because primitives import siblings, and the failure is often swallowed inside cleanup paths.
- Treating wrench data as `base_link` frame. The CSV `wrench_frame_id` column says `tool0_controller`. Direction-aware features (r_cop = ‖(-Ty/Fz, Tx/Fz)‖, F_lat in operator's intuitive frame) MUST be computed in tool frame; raw `Tx, Ty` magnitudes look small (~0.05 Nm) until normalized by Fz.
- Using counter-residual direction for force corrections during wedge-breaking. When the peg is wedged at one corner, the wrist sensor reads the OPPOSITE direction (the part is pressed into the rim edge on the other side). Counter-residual = AWAY from target. Use CAD-derived TOWARD-target direction (`target_xy − tcp_xy`) instead.
- Pushing harder downward / cardinal force pokes to break peg-on-rim wedges. Empirically (4 iterations × 12 corrections = ABORT each time on 2026-05-04) it deepens the wedge. The right action is retract 0.5–1.5 mm + drop Fz to -2 to -4 N + 1.5–2.0 s spiral search at lower `gain_scaling` (0.4–0.7) and `damping_factor` (0.15–0.30). Sources: Chhatpar 2001, FANUC, Robotiq.
- Detecting "stuck" from instantaneous v_z. Force-mode oscillation makes v_z dip momentarily even mid-wedge. Use net z-descent over a 2 s window with Fz smoothed over 0.5 s.
- **Self-matching `pgrep -f` in `until` loops.** A bash one-liner like `until ! pgrep -f "loop_iterate.*u_orange" >/dev/null; do sleep 2; done` (run via Bash-tool `eval`) **never exits** because the spawning bash itself contains the literal pattern in its cmdline — pgrep matches its own host shell, returns 0 forever. Failure mode observed 2026-05-04: 7 monitors leaked across one session, all with status "running" but pegged on `pgrep`. Fixes: (a) match the python invocation specifically with `pgrep -f "python3.*loop_iterate"`, (b) wait on a known PID with `while kill -0 <PID> 2>/dev/null; do sleep 2; done`, or (c) use the Monitor tool which streams events out-of-band. Don't use `pgrep -f` to wait on processes spawned via Bash tool.
## Decision matrix — copy / modify / write-fresh
| Decision | When to use |
|---|---|
| **Copy (lift file, attribute source)** | Code is in our language + framework + license is compatible + fits our architecture as-is |
| **Modify after copying** | Mostly fits, needs only surface tweaks (paths, message types, function names) |
| **Write fresh from algorithm/pattern** | Reference is in different language/framework/era, but the algorithm or pattern is sound — translate the *idea*, not the lines |
| **Skip** | Reference is well-known but doesn't fit our stack/scope (e.g., a node requiring an accelerometer we don't have) |
## OnRobot RG2 firmware quirks (verified 2026-05-03)
- **No precise positioning mode**: only modes 1 (grip), 8 (stop), 16 (grip_w_offset). Both 1 and 16 are GRIP commands — close past target by 1-5mm depending on direction. Mode 16 (default in our bridge) overshoots ±3-5mm; mode 1 ±2mm but targets RAW width (caller must add ~9.2mm fingertip offset). Width-based grasp checks must tolerate ≥5mm error (already widened in `move_to_grasp.py:888`).
- **Safety circuit latch**: bits 3, 5 of status reg 268 (`safety_circuit_1/2`). Per OnRobot docs: "can only be reset by power cycling." Software path: Modbus write `unit=63 addr=0 value=2` triggers Compute Box power-cycle (~10s); requires pendant STOP+PLAY after to re-attach URCap. NOT in local `onrobot.py`; documented in upstream `Osaka-University-Harada-Laboratory/onrobot.restartPowerCycle()`.
- **Width topics differ by 9.2mm**: `/gripper_width` is RAW mechanism, `/gripper_width_offset` is jaw-tip-to-jaw-tip gap (raw − 2 × 4.6mm fingertip). For grasping, use `/gripper_width_offset`.

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

### Phase 5 iterative-loop workflow (live as of 2026-05-04)

Three-script chain for tuning the insert primitive on one object:

```
loop_iterate.py  →  iterate_insert.py  →  compliant_insert.py (wrapper)
   harness            one-attempt              FSM: PRE→HOVER→ZERO→ACTIVE→DONE/ABORT
   N consecutive       delegates setup to
   successes           run_assembly_step --setup-only
                       then launches wrapper
```

Canonical command (one object, ≥5 consecutive successes):

```
python3 -m compliant_insertion_studio.scripts.loop_iterate \
  --object-name u_orange --base-name base1 --grasp-id 1
```

Tighter target / specific first held quat:

```
python3 -m compliant_insertion_studio.scripts.loop_iterate \
  --object-name u_orange --base-name base1 --grasp-id 1 \
  --target-success-count 3 \
  --first-held-quat 0.0062 -0.6494 0.7604 0.0055
```

`iterate_insert.py` (one attempt) flow:
1. Capture base1 world pose from `/objects_poses_real` BEFORE grasp (camera unobstructed by gripper)
2. Run `run_assembly_step.py --setup-only` → executes pick → rotate → place → regrasp → rotate, prints `PARSED HELD_QUAT [...]` on stdout
3. Launch the wrapper with `--base-world-pose X Y Z QX QY QZ QW` (CAD-derived predicted target) and `--use-default-base-position` and the held quat
4. Wrapper FSM runs to DONE (predicate met) or ABORT (timeout / max-corrections / safety)

`loop_iterate.py` chains across attempts: parses `PARSED HELD_QUAT` from attempt 1's stdout and passes it to attempts 2+ as `--already-held --held-quat <captured>`. Attempt 1 does the full canonical sequence (pick → rotate → place → regrasp → rotate). Attempts 2..N do **rotate-only** via `run_assembly_step --already-held --setup-only` (jumps straight to step 12 = rotate_object). **The gripper stays closed across every iteration** — no release, no regrasp. But **rotate_object IS called every attempt**, re-snapping EE orientation to canonical face-down.

Why the every-attempt rotate is critical: the wrapper's cleanup retracts holding whatever EE orientation the prior insert left, which is typically tilted by a few degrees because peg/slot tolerance lets the held part rotate slightly during force-mode wedging. Without re-rotating, the next attempt starts tilted (we measured a 4.5° baked-in tilt on 2026-05-04 iter 7 because we wrongly skipped rotate) and burns 20+ corrections fighting the angle instead of just inserting. Calling `rotate_object` is cheap (~3 s, one trajectory) and idempotent (snaps to nearest fold-equivalent of canonical).

Why every release+regrasp cycle is bad: (a) wastes ~30 s, (b) the place-down disturbs the part on the table, (c) `run_assembly_step` opens the gripper as its first action when NOT in `--already-held` mode, dropping any held part wherever the robot currently is. We hit this on 2026-05-04 (iter 5 + iter 6 setups failed at `move_to_grasp` because the part had drifted). Keeping the part held end-to-end + re-rotating each attempt is the canonical pattern.

### Wrapper key APIs

- `cad_lookup.predict_tcp_at_seat(base, object, grasp_id, base_world_xyz, base_world_quat_xyzw, flange_offset_m=0.2286)` — full chain `T_world_base ∘ T_base_object_seat ∘ T_object_grasp_point ∘ T_grasp_point_tcp`. Reads `~/Documents/aruco-grasp-annotator/data/fmb_assembly1.json` + `grasp_points/<obj>_grasp_points.json`.
- `--config <yaml>` — defaults to `compliant_insertion_studio/configs/defaults.yaml` (universal, shape-agnostic). Per-shape YAMLs are deleted; the termination predicate is derived from CAD chain, not shape geometry.
- `--use-default-base-position` — required when `--base-position` is not passed; computes hover/insert targets from CAD chain.

### Cancel-safety chain (REQUIRES idempotent process-group cleanup)

```
Operator Ctrl+C
  → loop_iterate SIGTERM handler
    → os.killpg(iterate_insert pgid, SIGTERM)
      → iterate_insert SIGTERM handler
        → os.killpg(wrapper pgid, SIGTERM)
          → wrapper run_done():
              stop_force_mode → switch to scaled_joint_trajectory_controller
              → _await_controller_active (gates the next step)
              → python3 -m primitives.move_to_safe_height
              → exit
```

Each layer wraps the child in `os.setsid` (`preexec_fn=os.setsid` in `subprocess.Popen`) so SIGTERM hits the whole pgroup. The wrapper's `_await_controller_active` is what makes the safe-height subprocess actually run — without it, the script-mode invocation can fail silently (ModuleNotFoundError swallowed) leaving force_mode_controller active.

### Engagement gate must allow z-drop dominance (2026-05-04 PM)

The CAD-derived predicted xy can be 15–20 mm off the actual seat (observed on iter v70: peg seated at xy=(0.0301,−0.3598), CAD-predicted (0.0257,−0.3560), error ~17 mm even after spiral recentering). A strict 6 mm `engagement_dist_thresh_m` then wrongly rejects a peg that's clearly inside the slot. Add a z-drop dominance shortcut in ENTRY_SETTLE: if `surface_z − tcp_z ≥ engaged_z_drop_dominant_m` (default 20 mm), accept engagement regardless of CAD-xy distance. Operator-demo full descent is 25–30 mm, so 20 mm is a safe "definitely inside" threshold. The dist gate still catches metastable low-force states where peg sits on top of rim (z_drop ≪ 20 mm).

### Contact xy ≠ Seat xy (2026-05-04 evening)

Critical late-session finding (likely root cause of an entire 12-hour failed iteration session): the **first-contact xy** (where peg first hits Fz>5N) is NOT the **seat xy** (where peg ends up after descending through chamfer). For u_orange today: contact at (+0.0308, -0.3554), seat at (+0.0341, -0.3635). 5.7mm offset in X, 8mm in Y. The first contact lands on the **rim**, the seat is the **slot center**. **`hole_xy_prior` should be the SEAT xy**, not the contact xy. Using contact xy as the spiral target points the search at the rim, where the peg can't drop in.

For each operator demo, extract BOTH values; use SEAT xy as the prior:
- contact_xy = first ACTIVE row with `fz > 5N`
- seat_xy = LAST ACTIVE row (or row where descent rate drops to 0 with z near predicted seat z)

### State-independent global seat detector (2026-05-04 evening)

The FSM's local `_motion_stopped_first_t` resets on every state transition (ENTRY_SETTLE↔FIND_HOLE↔INSERT). With our gate flapping under marginal conditions, this means a SEATED peg can sit motionless for 60+ seconds without ever accumulating the 0.75s sustain needed for the local seat predicate to fire. **Add a state-independent seat detector that runs every tick regardless of FSM state** (in `update()` before state dispatch): if `surface_z - tcp_z >= 20mm AND |dz/dt| < 0.5mm/s AND tilt < 5°` for 1.0s sustained, declare exit_done. Verified offline on FAIL_INSERT CSV: would have correctly fired at t=74.75s when peg had been seated for 60s.

### Cross-run signal analysis findings (2026-05-04 evening)

Comparing 6 successful vs 122 failed u_orange attempts at each timepoint post-contact:
- **t=1.0s: success has F_lat=3.5N + xy_excursion=0.9mm; failure has F_lat=0.6N + xy=0.3mm.**
- **t=1.5s: success F_lat=4.7N, xy=2.1mm; failure 0.8N, 0.4mm.**
- t=2.0s: success z_drop=2.2mm; failure 0.16mm.
- Successes have ~5N sustained F_lat for 1-1.5s causing 2mm xy excursion, then peg drops at t=2s.
- Failures don't accumulate this lateral motion → never engage chamfer.

Implication: an `INITIAL_PRESS` phase commanding 5N sustained for 1.5s in a fixed direction is the missing mechanism. The spiral PD only generates 0.5-2N sustained because PD is reactive — error stays small. Operator's hand applies sustained directed push that algorithm doesn't replicate. Implemented in FSM as `find_hole_press_*` knobs.

### Anti-pattern: tuning parameters without first analyzing data

12-hour session in 2026-05-04 wasted hours on parameter iteration (16+ versions v82-v97) before doing the analytical work that should have come first. **Rule for next sessions: write a CSV cross-run analyzer FIRST. Iterate parameters only after the data has revealed the actual failure mechanism.** Monitor commands are for state-transition events, not for diagnosing why peg isn't moving.

### Mode A vs Mode B (Phase 5)

- **Mode A** = pure compliance with universal termination predicate:
  `motion_stopped AND tcp_z_reached_predicted (CAD-derived) AND descended_post_contact ≥ 25 mm`, sustained 1 s.
- **Mode B** = active correction triggered by stuck-detection:
  `net z-descent over 2 s window < 0.5 mm/s AND smoothed Fz > 6 N`, sustained 2 s. State machine: NORMAL → CORRECTING → COOLDOWN → NORMAL. Up to `max_corrections=12` per episode, then ABORT.

### Mode B action-type research (2026-05-04, /tmp/gpt-task-result-123444.txt)

We tested 4 iterations of "push harder" Mode B (downward Fz=-9N + lateral force pokes 6N + counter-torque 0.3Nm). All 4 ABORT at 12 corrections. The peg is geometrically wedged on the rim; pushing deeper deepens the wedge. **Action type matters more than action magnitude.**

Right action for peg-on-rim wedge:
1. Retract 0.5–1.5 mm to unload static friction
2. Drop Fz to -2 to -4 N during search (NOT -9 N)
3. Run continuous spiral search 1.5–2.0 s, radius 0.25 → 1.0 mm
4. Lower force-mode gain during search: `gain_scaling=0.4–0.7` (vs nominal 1.0), `damping_factor=0.15–0.30` (vs nominal 0.7)
5. Restore nominal Fz/gains after correction

Sources GPT cited: Chhatpar & Branicky (2001) — spiral pitch = clearance; FANUC force-control manual — switch to search when error > chamfer + clearance/2; Robotiq spiral-search practice; Tang et al. (2016) on three-point contact.

### Wrench-feature hierarchy (verified empirically on iter-4 dataset)

- **Wrench frame is `tool0_controller`**, NOT `base_link`. Telemetry CSV column `wrench_frame_id` confirms. Direction-aware features (r_cop, F_lat in operator's intuitive frame) MUST be computed in tool frame, then optionally transformed via TCP quaternion if needed for base-frame analysis.
- **r_cop = ‖(-Ty/Fz, Tx/Fz)‖** is the missing direction signal. On iter-4 ACTIVE phase: median r_cop = 5.5 mm, mean COP vector = (-1.0, -5.4) mm (consistent direction, not noise) — for a ~22 mm peg, that's rim-contact-scale lever arm. Raw `Tx, Ty` look "tiny" (~0.05 Nm) until you normalize by Fz.
- **Counter-residual is geometrically wrong for wedges**. When peg is wedged at (-X,-Y) corner, wrist sensor reads (+X,+Y) because the part is being pressed into the (+X,+Y) rim edge. Counter-residual direction = AWAY from target. **Use TOWARD-TARGET direction** (CAD-derived `target_xy − tcp_xy`) for force corrections.
- **Stuck-detection: net z-descent over 2 s window**, NOT instantaneous v_z. Smooth Fz over 0.5 s window before threshold-checking — force-mode oscillation makes instantaneous fz dip below threshold momentarily, breaking sustain timers.

### Subprocess invocation rule

Inside the wrapper / orchestrator scripts, ALWAYS launch primitives via module mode:

```
python3 -m primitives.move_to_safe_height ...    # CORRECT
python3 primitives/move_to_safe_height.py ...    # WRONG: ModuleNotFoundError
```

The script-path form fails because primitives import sibling modules from the `primitives` package. We hit this twice on 2026-05-04 — once swallowed silently inside the cleanup path, leaving force_mode_controller active after ABORT.
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
