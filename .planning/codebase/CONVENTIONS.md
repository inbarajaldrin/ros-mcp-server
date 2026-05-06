# Project Conventions — Compliant Insertion Studio

**Single source of truth for project working patterns.** Auto-summarized into `CLAUDE.md` by `gsd-tools generate-claude-md` (which keeps headings, bullets, and tables; drops prose). The "Key Rules" section is structured to survive that summarizer intact.

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
- **Diagnostic / verification scripts MUST tee output to a timestamped log file** under `compliant_insertion_studio/logs/diagnostics/<script>_<YYYYMMDD_HHMMSS>.log`. Operator + Claude both read structured logs efficiently; copy-paste from terminal scrollback is slow, error-prone, and discards data above the scrollback limit. Pattern: open a log file at start, dual-write each interesting line via a small `tee()` helper or `sys.stdout = TeeStream(stdout, logfile)`. Include the log path in the script's startup banner so operator knows where it lives.

## Hardware + workspace geometry — verified empirically 2026-05-03

Pin these so future work doesn't re-derive them:

- **Workspace orientation (operator-confirmed via verify_baselink_motion.py)**:
  - `+X` in base_link = **robot's RIGHT**   (`-X` = robot's LEFT)
  - `+Y` in base_link = **FORWARD** away from base   (`-Y` = back toward base)
  - `+Z` in base_link = **UP**   (`-Z` = down to floor)
  - The operator's rectangular workspace has the robot base mounted at the long-side center; the EE only sits inside the workspace when `shoulder_pan = +π/2`. (`HOME_JOINTS` in `primitives/shared/config.py` encodes this.)
- **`base` ↔ `base_link` differ by `R_z(180°)`** — X and Y signs flip between the two frames; Z is preserved. Verify with `ros2 run tf2_ros tf2_echo base base_link`.
- **Frame each ROS topic actually publishes in (live-verified)**:
  - `/tcp_pose_broadcaster/pose` → **`base`** (NOT `base_link`)
  - `/force_torque_sensor_broadcaster/wrench` → **`tool0_controller`** (NOT `base_link`; post-driver-PR-#1652)
  - Wrapper transforms wrench tool0_controller → base_link before logging; SCHEMA.md columns are in base_link.
- **`force_mode_controller` silently auto-transforms `task_frame` to `<tf_prefix>base`** — sending `task_frame.header.frame_id="base_link"` + raw wrench produces silent X/Y inversion (the wrench is interpreted in the rotated frame). Always send `task_frame.header.frame_id="base"` + apply explicit base_link → base sign flip on the wrench in code. Pattern in `compliant_insertion_studio/wrapper/compliant_insert.py::_start_force_mode()`.
- **`HOME_JOINTS = [+90°, -90°, +90°, -90°, -90°, 0°]`** in `primitives/shared/config.py` — joint-space "tidy home" matching the workspace + the F/T calibration starting orientation. Use `move_home.py --joint-space` (or for diagnostics, `move_joints.py send --positions ...`) when you need predictable joint config (e.g. before F/T calibration). Cartesian `HOME_POSE` is unchanged for general primitives; the IK seed there happens to land at `shoulder_pan ≈ +79°` which is OK for most workflows but not for calibration.
- **F/T payload is set per-mount and applied per-session by `launch_robot.sh`**: foundational calibration result lives at `compliant_insertion_studio/configs/ft_calibration_<gripper_id>_<date>.yaml`. **As of 2026-05-06, `launch_robot.sh real` automatically calls `/io_and_status_controller/set_payload`** with values from the most-recent calibration YAML (sources `result.mass_kg` + `result.cog_xyz_m`). The earlier convention "paste set_target_payload into bringup launch" was never actually implemented — that line was missing from every launch file from project start, force_mode ran with 0 kg gravity comp, ~21N of unmodeled load, ~8mm TCP drift per APPROACH. Pendant payload (set via the teach pendant) does NOT propagate to ROS-side force_mode. Re-run calibration only when the gripper, jig, or sensor mount changes; drop the new YAML in `configs/` and `launch_robot.sh` picks it up next bringup. Current calibration: 2.1109 kg, CoG ≈ [-0.003, +0.003, -0.032] m.

## Operator workflow — Local mode + bringup quirks

- **`launch_robot.sh real|fake [--rviz] [--ip <IP>]`** is the single entry point for bringup (works for both real and fake hardware). RViz with the dual-RobotModel config (UR5e + RG2) comes up via `--rviz`. Activates `scaled_joint_trajectory_controller` automatically.
- **`close_robot.sh [-v]`** is the single shutdown entry point. Three-phase shutdown (RViz SIGTERM → ros2 launch SIGINT → UR driver children SIGTERM-grace-SIGKILL) matching the proven pattern from `_real_mode_stash`. Filters cursor IDE processes from pgrep matching.
- **Local-mode pendant blocks every action command** — dashboard `play`, `stop`, `pause`, `power_on`, `power_off`, `brake_release`, `unlock_protective_stop`, `restart_safety`, `shutdown`, `load_program` all reject with "Command is not allowed due to safety reasons". Only read-only queries (`status`, `mode`, `safety`, `running`, ...) work in Local mode. `utils/ursim_cli.py` surfaces a clear hint when one of these gets rejected.
- **Bringup restart leaves a stale URCap link** — when `close_robot.sh` runs and a new bringup is launched, the External Control URCap node on the pendant loses its connection to the new `ros2_control_node`. The pendant still SHOWS Play as active, `program_running` reports true, but ros2_control trajectories are silently rejected with `"Velocity or acceleration limits exceeded. Enable robot in URcap to fix this."` — even for tiny moves. **Fix: operator presses STOP then PLAY on the pendant** (re-establishes the External Control link). Cannot be done from code in Local mode.
- **Protective stop recovery is manual on the pendant** — `dashboard_client/unlock_protective_stop` is also Local-mode-blocked. Operator clears it via the touchscreen prompt.
- **Pendant program is `external_control.urp`** — keep this loaded. After bringup restart + STOP+PLAY, it's the program that hosts ros2_control's External Control connection.

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

---

## Detailed rationale (humans only — auto-summarizer drops most of what follows)

### Research before code — procedure

- Search 2–4 query variations (vendor docs, GitHub, papers, community forums)
- Identify candidate repos / articles
- Clone repos into `_references/repos/`, save articles to `_references/articles/`
- Read enough to make an honest copy/modify/write-fresh decision per piece
- Only then write code

This applies to every substantive deliverable, not just the "interesting" ones. F/T calibration was the trigger that made the operator surface this principle explicitly; the rule is general.

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

---

*Defined: 2026-05-01 during project initialization, after multiple operator clarifications.*
*Update mechanism: edit this file → run `gsd-tools generate-claude-md` → commit both this file and CLAUDE.md.*
