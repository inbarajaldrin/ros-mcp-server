# Phase 7: Gripper URDF + RViz Visualization - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-02
**Phase:** 07-gripper-urdf-rviz
**Areas discussed:** URDF source path, URDF integration mechanism, Gripper actuation model, Offset replacement strategy, Validation criteria

---

## Blocking Anti-Pattern Acknowledgment (pre-discussion gate)

Per `.continue-here.md`, two blocking anti-patterns were acknowledged before any gray-area work:

1. **Writing SOPs/specs from extrapolation** — Prevention: Phase 7 plan's first task must be `RESEARCH-01` (5–15 min web/docs/repo search + clone refs). No URDF / launch / RViz config writing before that completes.
2. **Running robot moves without RViz preview** — Prevention: Phase 7 is fully `[N]` no-robot. Acceptance gate D-9b explicitly requires a fake-hardware sweep test (gripper follows EE visibly).

---

## Area Selection

| Option | Description | Selected |
|--------|-------------|----------|
| URDF source path | How do we get a usable URDF + how does it integrate with UR5e? | ✓ |
| Gripper actuation model | Does the URDF show fingers moving? | ✓ |
| Offset replacement strategy | How do we kill GRIPPER_CENTER_TOOL_OFFSET in 51 sites? | ✓ |
| Validation criteria | What proves Phase 7 succeeded? | ✓ |

**Notes:** All four areas selected. No areas deferred.

---

## URDF Source Path

### Question 1: How should we produce the OnRobot RG2 URDF?

| Option | Description | Selected |
|--------|-------------|----------|
| Find community RG2 URDF first (Recommended) | Search ROS-Industrial / GitHub / OnRobot samples; copy + adapt if found, fall back to USD→URDF otherwise. | |
| USD→URDF via Isaac Sim exporter | Drive Isaac Sim's URDF exporter on RG2.usd. One source of truth (USD); exporter output may need cleanup. | ✓ |
| Hand-write URDF from extracted meshes | Dump RG2 meshes via usdcat/usdview, hand-write minimal URDF. Most labor, easiest to get inertials wrong. | |
| You decide based on research findings | Lock first task as RESEARCH-01, defer the choice. | |

**User's choice:** USD→URDF via Isaac Sim exporter

**Notes:** Operator follow-up message: "use the skill inside the isaacsim mcp folder it already has a lot of scripts that can be used or reverse engineered." → Project skill `isaac-sim-extension-dev` and the `~/Documents/isaac-sim-mcp/exts/ur5e-dt/` extension are first-class references.

### Question 2: Where does the generated RG2 URDF live, and how does it attach to UR5e?

| Option | Description | Selected |
|--------|-------------|----------|
| Xacro macro under compliant_insertion_studio/urdf/ (Recommended) | Wrap RG2 in xacro, include into UR xacro tree at tool0. ros2_control sees gripper natively. | |
| Static merged URDF, regenerated | Python regen script emits one merged URDF; bringup loads it directly. | |
| Standalone gripper URDF + static TF | RG2 stays its own URDF; static TF tool0→rg2_base_link bridges it. RViz shows it; ros2_control / MoveIt planning scene don't see it. | ✓ |

**User's choice:** Standalone gripper URDF + static TF

**Notes:** Operator follow-up message: "but i hope the gripper moves with the ur5e ee so it should be fixed to the tf link properly you can refer isacsim to lead the robot scene using quickstart button that attaches the gripper to the ur5e which can be used to match the tf of the gripper connection." → The static TF must be sourced from the existing isaac-sim-mcp ur5e-dt quickstart attach flow (D-2). This canonicalizes the offset rather than guessing.

### Question 3: ROADMAP success criterion #3 (ros2_control collision-check) tension

| Option | Description | Selected |
|--------|-------------|----------|
| Drop the collision-check criterion (Recommended for v1) | RViz visual is the goal; collision-check stays in custom Python preflights. ROADMAP updates. | ✓ |
| Keep it as a stretch | Standalone v1; migrate to xacro-into-UR-tree as a follow-up phase if pain emerges. | |
| Switch to xacro-into-UR-tree after all | Reconsider integration mechanism; revise D-2. | |

**User's choice:** Drop the collision-check criterion

**Notes:** ROADMAP.md Phase 7 success criterion #3 must be updated by the planner when writing PLAN.md.

---

## Gripper Actuation Model

### Question: Should the URDF show the RG2 fingers moving in RViz?

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed everywhere — visual blob (Recommended) | All RG2 joints fixed-type. Static shape, no joint_state wiring. | ✓ |
| Mimic prismatic finger pair, /gripper_width-fed | Subscribe to OnRobot driver's gripper width topic; RViz shows real-time grip width. | |
| Fixed v1, mimic v2 if telemetry needs it | Defer mimic until Phase 3/4 surfaces a need. | |

**User's choice:** Fixed everywhere — visual blob

**Notes:** Mimic-pair animation deferred to a future phase. Trigger condition logged in CONTEXT.md "Deferred Ideas".

---

## Offset Replacement Strategy

### Question: How do we replace GRIPPER_CENTER_TOOL_OFFSET in its 51 call sites?

| Option | Description | Selected |
|--------|-------------|----------|
| Auto-derive constant from URDF FK at import (Recommended) | Keep constant name; compute value from URDF at import time. Callers untouched. | |
| Big-bang refactor to TF lookup helper | All 51 sites swap to live TF lookup. Pure but high blast radius. | |
| New gripper_center TF frame, gradual migration | Publish static TF; migrate callers when touched for other reasons. | |
| **(Free-text response — see notes)** | | ✓ |

**User's choice:** "we ddont do that. we dont have to change the codes offset logic at all. this is purely a visualization build we wille xtends its capabilities later"

**Notes:** Operator explicitly removed offset replacement from Phase 7 scope. All 51 call sites stay byte-identical. The custom DH-based `GRIPPER_CENTER_TOOL_OFFSET = [0, 0, 0.2286]` continues to be the source of truth for primitive code. URDF integration is parallel visualization infrastructure. Migration to URDF FK is now a Deferred Idea, triggered by future need (e.g., new gripper introduced). This significantly reduces Phase 7 scope and risk.

---

## Validation Criteria

### Question: Given Phase 7 is visualization-only, what's the validation gate?

| Option | Description | Selected |
|--------|-------------|----------|
| Visual + TF cross-check (Recommended) | 5-pose RViz preview + sweep test + TF cross-check vs isaac-sim-mcp attach offset within 1 mm/1°. | ✓ |
| Visual only | Operator eyeballs RViz; no automated regression. | |
| Visual + TF cross-check + screenshot artifact | Adds RViz screenshots saved to compliant_insertion_studio/docs/phase7_visual_validation/. | |

**User's choice:** Visual + TF cross-check

**Notes:** TF cross-check anchors the static-TF offset to the canonical isaac-sim-mcp value (D-9c). Screenshot artifact is logged as a Deferred Idea.

---

## Closing

### Question: Anything else to surface before writing CONTEXT.md?

| Option | Description | Selected |
|--------|-------------|----------|
| I'm ready for context | Write CONTEXT.md with the decisions captured above. | ✓ |
| Explore more gray areas | Surface additional gray areas (bringup launch arg, RViz config location, static TF lifecycle, etc.). | |
| Revisit one of the four answered areas | Reopen any of the four. | |

**User's choice:** I'm ready for context

---

## Claude's Discretion

- Specific file names within `compliant_insertion_studio/urdf/` (e.g., `rg2.urdf` vs `rg2.urdf.xacro`).
- Choice of `static_transform_publisher` vs custom Python TF node.
- Whether to extend existing fake-hardware bringup launch or add a new `compliant_insertion_studio/launch/ur5e_with_rg2.launch.py`.
- Cross-check implementation: manual `ros2 run tf2_ros tf2_echo` vs Python pytest.

## Deferred Ideas (also in CONTEXT.md `<deferred>`)

- Migrate `GRIPPER_CENTER_TOOL_OFFSET` to URDF FK lookup (51 call-site refactor).
- Mimic prismatic finger pair fed by `/gripper_width` topic.
- ros2_control / MoveIt planning-scene integration of RG2 collision geometry (originally success criterion #3).
- Xacro-into-UR-tree integration mechanism (v2 alternative).
- RViz screenshot regression baseline in `compliant_insertion_studio/docs/phase7_visual_validation/`.
