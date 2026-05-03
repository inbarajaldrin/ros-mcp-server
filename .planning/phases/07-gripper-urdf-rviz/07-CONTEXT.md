# Phase 7: Gripper URDF + RViz Visualization - Context

**Gathered:** 2026-05-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Render the OnRobot RG2 gripper geometry in RViz attached to the UR5e `tool0` link, sourced from the existing `RG2.usd` asset in `~/Documents/isaac-sim-mcp/exts/ur5e-dt/`. Phase 7 is **visualization-only** — no Python primitive code changes, no FK migration, no replacement of `GRIPPER_CENTER_TOOL_OFFSET`. The custom DH-based offset stays in service across all 51 call sites; URDF integration is purely a visual layer for RViz preview during calibration / wrapper / dashboard work.

</domain>

<decisions>
## Implementation Decisions

### URDF Source

- **D-1:** Generate the RG2 URDF using the **Isaac Sim USD→URDF exporter** driven on `~/Documents/isaac-sim-mcp/exts/ur5e-dt/assets/gripper/RG2.usd`. The operator already has Isaac Sim + isaac-sim-mcp installed; the project skill `isaac-sim-extension-dev` covers the workflow.
- **D-2:** **Reverse-engineer the existing `isaac-sim-mcp` quickstart "attach gripper" flow** in the `ur5e-dt` extension to extract the canonical `tool0` → `rg2_base_link` transform. Do not guess or hand-measure the offset — the canonical value already lives in the attach logic the operator uses to mount the RG2 onto the UR5e in Isaac Sim. This is the single source of truth for the TF.
- **D-3:** First plan task is **RESEARCH-01** — read the relevant `ur5e-dt` extension code (especially the quickstart attach script), extract the transform, document it in `_references/articles/rg2_attach_transform.md`, and confirm the USD→URDF exporter output passes a sanity check (no broken mesh paths, single base link, fingers as fixed children of the base for v1).

### URDF Integration Mechanism

- **D-4:** **Standalone gripper URDF + static TF.** Do NOT modify the UR description xacro tree. Generated RG2 URDF lives under `compliant_insertion_studio/urdf/rg2.urdf` (or `.xacro`). A `static_transform_publisher` (or equivalent) publishes `tool0` → `rg2_base_link` so RViz's TF chain composes the gripper with the UR robot model.
- **D-5:** **Two-RobotModel-display setup in RViz.** UR description loads as the primary `robot_description`. RG2 loads via a second `robot_state_publisher` (or `joint_state_publisher` for the fixed RG2) and a separate RobotModel display in RViz pointed at the gripper URDF topic. Both render in the same RViz window because they share the TF tree.
- **D-6:** **ROADMAP Phase 7 success criterion #3 (`ros2_control / MoveIt collision-check integration so the gripper is part of the planning scene`) is DROPPED.** The standalone-URDF approach explicitly does not put the gripper into ros2_control's planning scene. Collision checking continues to use the custom Python preflights (`primitives/rotate_object.py`, gripper-tip preflight in calibration). Future phase may revisit if collision pain emerges. **Planner must update ROADMAP.md to reflect this** when writing PLAN.md.

### Gripper Actuation Model

- **D-7:** **Fixed-joint everywhere.** All RG2 joints in the URDF are `type="fixed"`. The gripper renders as a static shape with fingers at a fixed default position. No `joint_state_publisher` wiring for finger motion, no subscription to OnRobot driver gripper-width topic, no mimic joints. v1 ships fast; finger animation is deferred (see Deferred Ideas).

### Offset Replacement Strategy

- **D-8:** **`GRIPPER_CENTER_TOOL_OFFSET` is NOT touched in this phase.** All 51 call sites stay byte-identical. `primitives/shared/config.py:8` retains `GRIPPER_CENTER_TOOL_OFFSET = np.array([0.0, 0.0, 0.2286])`. Phase 7 produces a parallel visualization layer; the FK/offset migration is explicitly out of scope and deferred to a future phase.

### Validation Gate

- **D-9:** **Visual + TF cross-check.**
  - (a) Gripper renders correctly at `tool0` in fake-hardware RViz across at least 5 robot poses (home + 4 obliques pulled from `compliant_insertion_studio/configs/calibration_poses.yaml`).
  - (b) Sweep test: command continuous joint motion through fake-hardware bringup; gripper visibly follows EE smoothly with no detached / lagging frames.
  - (c) The static TF `tool0` → `rg2_base_link` matches the transform extracted from the isaac-sim-mcp quickstart attach logic within **1 mm translation / 1° rotation**. This is the regression gate — if the offset drifts from Isaac Sim, the static-TF setup is wrong.

### Working Pattern (project conventions, applied to Phase 7)

- **D-10:** First plan task must be research per CONVENTIONS §1 — `RESEARCH-01: USD→URDF tooling + RG2 attach reverse-engineering`. No URDF / launch / RViz config writing happens before that task completes and saves findings to `_references/articles/`.
- **D-11:** All Phase 7 deliverables under `compliant_insertion_studio/` per CONVENTIONS §"All project deliverables under `compliant_insertion_studio/`". URDF in `compliant_insertion_studio/urdf/`, RViz config in `compliant_insertion_studio/rviz/`, launch additions in `compliant_insertion_studio/launch/`, docs in `compliant_insertion_studio/docs/`.
- **D-12:** Phase 7 is entirely `[N]` no-robot. RViz fake-hardware bringup is `[N]`; no real-robot motion in this phase.

### Claude's Discretion

- File names within `compliant_insertion_studio/urdf/` (e.g., `rg2.urdf` vs `rg2.urdf.xacro` vs `rg2_macro.xacro`) — pick whatever the Isaac Sim exporter naturally produces, then adapt.
- Whether to publish the static TF via a `static_transform_publisher` ROS2 node or via a small Python ROS2 node that reads the value from a YAML config (planner picks based on what's simpler given the bringup launch structure).
- Whether to extend the existing fake-hardware bringup launch or create a new `compliant_insertion_studio/launch/ur5e_with_rg2.launch.py` (planner picks; the latter is more drop-in friendly per the "self-contained subsystem" rule).
- Whether to verify the offset cross-check with a manual `ros2 run tf2_ros tf2_echo` script vs a small Python pytest (either is fine; the goal is the 1 mm/1° gate).

### Folded Todos

None — no pending todos in `.planning/todos/`.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents (researcher, planner, executor) MUST read these before writing anything.**

### Isaac Sim source-of-truth (highest priority — D-2 anchor)

- `~/Documents/isaac-sim-mcp/exts/ur5e-dt/` — entire extension. Look for the quickstart "attach gripper" / scene-bringup flow. The transform between UR5e EE and RG2 already lives in here; it is the canonical value that Phase 7 must match (D-9c regression gate).
- `~/Documents/isaac-sim-mcp/exts/ur5e-dt/assets/gripper/RG2.usd` — source USD asset for the exporter (3.1 MB).
- `~/Documents/isaac-sim-mcp/exts/ur5e-dt/assets/robot/ur5e.usd` — for cross-reference only; we don't re-export UR5e.
- Project skill `isaac-sim-extension-dev` (in available-skills list) — covers Isaac Sim extension workflow including USD → URDF tooling.

### Project conventions and contract

- `.planning/codebase/CONVENTIONS.md` — Key Rules (Research before code, All deliverables under `compliant_insertion_studio/`, Inline-default, Two-track `[N]/[R]/[N→R]`).
- `.planning/PROJECT.md` — Core Value: "Replace the failing `--insert` real-mode path." Phase 7 must not regress the existing custom DH-based offset path.
- `.planning/ROADMAP.md` Phase 7 section (lines 120–137) — Goal, Depends on, anticipated requirements, success criteria. **Note:** success criterion #3 (`ros2_control / MoveIt collision-check`) is DROPPED by D-6; planner must update ROADMAP when writing PLAN.md.
- `.planning/phases/07-gripper-urdf-rviz/.continue-here.md` — session pause notes, blocking anti-patterns acknowledged at the start of this discussion.

### Existing project code (read to know what NOT to break)

- `primitives/shared/config.py:8` — `GRIPPER_CENTER_TOOL_OFFSET = np.array([0.0, 0.0, 0.2286])` — stays untouched per D-8.
- `primitives/move_to_safe_height.py:23,252,253`; `primitives/rotate_object.py:45,301,675,1086,1117`; `primitives/move_home.py:12,80`; `queries/verify_clearance.py:44`; `queries/get_current_gripper_center_pose.py:24,39` — current 51 call sites of `GRIPPER_CENTER_TOOL_OFFSET`. None are modified by Phase 7.
- `~/ros2_ws/src/Universal_Robots_ROS2_Description/urdf/ur_macro.xacro:454-467` — defines `tool0` link (the static-TF parent for the RG2). Read-only; Phase 7 does not modify UR description.

### USD → URDF tooling (research scope for RESEARCH-01)

- *To be filled by RESEARCH-01* — Isaac Sim's URDF exporter API path (likely under `omni.importer.urdf` or the newer `isaacsim.asset.exporter.urdf` namespace; depends on installed Isaac Sim version). Researcher saves discovered API path to `_references/articles/usd_to_urdf_isaac_sim.md`.

### Calibration poses (validation set for D-9a/b)

- `compliant_insertion_studio/configs/calibration_poses.yaml` — pull home + 4 oblique poses from here for the 5-pose visual validation set.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `~/Documents/isaac-sim-mcp/exts/ur5e-dt/` quickstart attach flow — already encodes the canonical `tool0` → `rg2_base_link` transform (operator-confirmed source of truth). Reverse-engineer rather than re-derive.
- `RG2.usd` asset — source of truth for gripper geometry; do not re-mesh from CAD.
- `~/ros2_ws/src/Universal_Robots_ROS2_Description/urdf/ur_macro.xacro` — defines `tool0` (ROS-Industrial canonical EE frame) — no changes needed.
- Project skill `isaac-sim-extension-dev` — provides Isaac Sim extension dev workflow including USD→URDF.
- Project skill `stl-mesh-analyzer` — could verify mesh dimensions / hole alignment if exporter output is suspect.

### Established Patterns

- All project deliverables go under `compliant_insertion_studio/` (CONVENTIONS, applied to D-11).
- `[N]/[R]/[N→R]` two-track split — Phase 7 is fully `[N]` (D-12).
- Research-before-code: 5–15 min web/docs/repo search + clone references to `_references/repos/` BEFORE writing code (CONVENTIONS §1, applied to D-10).

### Integration Points

- **Fake-hardware bringup** — RViz preview uses fake-hardware UR bringup (existing PIDs 668318/668344/668452 from the prior session, can be killed and restarted). Phase 7 launch must extend or compose with this; do NOT introduce real-hardware dependency.
- **TF tree** — Static TF `tool0` → `rg2_base_link` is the bridge between UR robot_description and RG2 standalone URDF. Both render in same RViz because they share `tool0` as a TF chain anchor.
- **No primitive code touched.** D-8 keeps `primitives/`, `queries/`, `_real_mode_stash/` untouched. The static-TF / RG2 URDF setup is parallel infrastructure.

</code_context>

<specifics>
## Specific Ideas

- **Operator-cited reference flow:** isaac-sim-mcp's "quickstart button that attaches the gripper to the ur5e" is the authoritative source for the tool0→RG2 transform. Phase 7 plan must explicitly reverse-engineer this rather than measure-from-CAD or guess from the USD `xformOp:translate` values.
- **Gripper must visually follow the EE.** The TF static publish must be loaded together with bringup so the gripper render moves as the robot moves. Sanity-check explicitly via the D-9b sweep test.
- **No animation in v1.** The user's stated complaint about the existing setup is "annoying to maintain", not "I can't see the gripper open/close." Visualizing fingers as a static blob meets the bar; finger animation is a v2 nice-to-have.
- **No code refactor.** Operator was explicit: "we don't have to change the code's offset logic at all. this is purely a visualization build we will extend its capabilities later." This locks D-8.

</specifics>

<deferred>
## Deferred Ideas

These came up during discussion but belong in a future phase. Don't lose them.

- **Migrate `GRIPPER_CENTER_TOOL_OFFSET` to URDF FK lookup.** All 51 call sites would switch to a TF lookup or auto-derived constant. Out of scope for Phase 7 per D-8. Suggested trigger: a future phase where a non-RG2 gripper is being introduced, or where the constant goes stale relative to a refined URDF.
- **Mimic prismatic finger pair fed by `/gripper_width` topic.** Would let RViz show real-time grip width during demos. Trigger: if Phase 4 dashboard or Phase 3 demo replay surfaces "I want to see grip state animate." Currently deferred per D-7.
- **`ros2_control` / MoveIt planning-scene integration of RG2 collision geometry.** Originally Phase 7 success criterion #3, dropped per D-6. Trigger: if downstream phases (especially Phase 3 data collection or Phase 5 algorithm validation) hit collision pain that custom Python preflights can't catch.
- **Xacro-into-UR-tree integration approach.** v2 alternative if the standalone-URDF + static-TF approach proves brittle (e.g., RViz robot_description topic conflicts, joint_state_publisher disagreements). Currently deferred.
- **RViz screenshot regression baseline.** Saving 5-pose RViz screenshots to `compliant_insertion_studio/docs/phase7_visual_validation/` was offered as part of validation option 3 but not selected. Could be added later if visual regressions become a recurring issue.

### Reviewed Todos (not folded)

None — no pending todos to review.

</deferred>

---

*Phase: 07-gripper-urdf-rviz*
*Context gathered: 2026-05-02*
