# Phase 7 — Gripper URDF + RViz Visualization

> **Status:** v1 shipped, visualization-only. RG2 renders attached at UR5e tool0
> in fake-hardware RViz. The custom DH-based `GRIPPER_CENTER_TOOL_OFFSET` in
> `primitives/shared/config.py` is **untouched** — Phase 7 is parallel
> infrastructure, not a refactor (per `.planning/phases/07-gripper-urdf-rviz/07-CONTEXT.md` D-8).

## What this gives you

A second `robot_state_publisher` + a `static_transform_publisher` that together
make the OnRobot RG2 gripper render in RViz attached to the UR5e's `tool0`
frame. The gripper visibly follows the EE during fake-hardware joint motion
(verified across 4 poses; see `phase7_visual_validation/`).

## How to launch

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch /home/aaugus11/Documents/ros-mcp-server/compliant_insertion_studio/launch/ur5e_with_rg2.launch.py
```

Defaults: fake-hardware UR bringup + RG2 URDF + dual-RobotModel RViz config.

| Argument   | Default | Notes |
|------------|---------|-------|
| `rviz`     | `true`  | Set `false` to skip launching RViz. |
| `rg2_only` | `false` | Set `true` to skip the UR bringup and only load the RG2 URDF + a `world→tool0` static TF for fast RG2-only iteration. |
| `tf_xyz`   | `"0 0 0"` | Static TF translation `tool0 → rg2_base_link` (meters). |
| `tf_rpy`   | `"0 0 3.141592653589793"` | Static TF rotation `tool0 → rg2_base_link` (radians, xyz Euler). The default of `(0, 0, π)` came from a live read of the isaac-sim-mcp ur5e-dt quickstart attach scene — see "How the canonical TF was derived" below. |

To override the static TF without editing the launch file:

```bash
ros2 launch /home/aaugus11/.../ur5e_with_rg2.launch.py \
    tf_rpy:="0 0 3.14159265" tf_xyz:="0 0 0.005"
```

This is the right hook for tuning if a different gripper or a different mount adapter changes the offset.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│ ros2 launch ur5e_with_rg2.launch.py                                        │
└────────────────────────────────────────────────────────────────────────────┘
        │
        ├── ur_robot_driver/launch/ur_control.launch.py
        │       (use_fake_hardware:=true, launch_rviz:=false)
        │       │
        │       ├── ros2_control_node + UR controllers
        │       └── robot_state_publisher  →  /robot_description (UR URDF)
        │              └─ TF: world → base_link → … → wrist_3_link → tool0
        │
        ├── robot_state_publisher (rg2 namespace)
        │       └─ /rg2/robot_description  (RG2 URDF, single rg2_base_link)
        │
        ├── static_transform_publisher
        │       └─ TF: tool0  →  rg2_base_link   (xyz=0 0 0,  rpy=0 0 π)
        │
        └── rviz2
                └─ Two RobotModel displays:
                     • UR5e   on  /robot_description
                     • RG2    on  /rg2/robot_description
                   Both render in the same view because they share the TF
                   tree at tool0.
```

The two `robot_description` topics are NOT merged into one URDF. They're two
independent URDFs, bridged by the static TF. This is the standalone-URDF +
static-TF approach picked in CONTEXT.md D-4 (vs xacro-merged-into-UR-tree,
which was rejected because it requires writing into the UR description package).

**Tradeoff:** ros2_control / MoveIt do not see the gripper geometry as part of
the planning scene. Collision checking remains in the project's existing
custom Python preflights (e.g., `primitives/rotate_object.py`'s gripper-tip
check). That tradeoff was the explicit drop of ROADMAP §Phase 7 success
criterion #3 (see CONTEXT.md D-6).

## How the URDF was generated

`compliant_insertion_studio/scripts/extract_rg2_urdf.py` — a `pxr`-only script
that reads the OnRobot RG2 USD asset
(`~/Documents/isaac-sim-mcp/exts/ur5e-dt/assets/gripper/RG2.usd`) and writes:

- `compliant_insertion_studio/urdf/rg2/rg2.urdf` — single `rg2_base_link`,
  no joints, 7 visual + 7 collision blocks.
- `compliant_insertion_studio/urdf/rg2/meshes/*.obj` — extracted mesh
  geometry, one OBJ per RG2 sub-link.

This sidesteps Isaac Sim's `isaacsim.asset.exporter.urdf` / `nvidia.srl.from_usd.to_urdf.UsdToUrdf`,
which fails on the RG2 with `"Unable to convert this USD to URDF because it has
kinematic loops"` — the OnRobot RG2's 4-bar parallel-finger linkage forms loops
that the SRL exporter can't reduce to a tree. Phase 7 is fixed-joint-everywhere
anyway (CONTEXT.md D-7), so the loop-joint information is irrelevant: we just
glue all the meshes onto a single link.

To regenerate the URDF from the USD:

```bash
python3 /home/aaugus11/Documents/ros-mcp-server/compliant_insertion_studio/scripts/extract_rg2_urdf.py
```

There's also `compliant_insertion_studio/scripts/export_rg2_urdf_standalone.py`
(the Isaac Sim driver that *would* run the SRL exporter if the kinematic-loop
limitation were lifted). Kept as a reference for future grippers without loops.

## How the canonical TF was derived

The static-TF rotation was extracted from a **live read** of the `isaac-sim-mcp`
ur5e-dt extension's quickstart attach scene, after `play()` had run a few
physics steps so the fixed-joint constraint settled.

Procedure (preserved in `_references/articles/rg2_attach_transform.md`):

1. Launch the ur5e-dt extension:
   ```bash
   nohup ~/env_isaaclab/bin/isaacsim --ext-folder ~/Documents/isaac-sim-mcp/exts --enable ur5e-dt > /tmp/isaacsim.log 2>&1 &
   ```
   (The skill's `isaacsim_launch.sh launch ur5e-dt` has a false-positive in
   `is_running()` when invoked from inside a bash `eval`, so we bypass it.)
2. Run quickstart via the extension's MCP socket (port 8766):
   ```python
   send_cmd({"type":"quick_start", "params":{}})
   ```
3. Read the world transforms of `/World/UR5e/wrist_3_link` and
   `/World/RG2_Gripper` (the asset root — see "Pitfall" below for why **not**
   the leaf `/World/RG2_Gripper/onrobot_rg2_base_link`):
   ```python
   xc = UsdGeom.XformCache()
   xc.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/RG2_Gripper"))
   ```
4. Compose:
   ```
   R(rg2_base_link in tool0)  =  R(wrist_3_link in world).inv() * R(rg2_in_world)
                              ≈  rpy_xyz (0, 0, π)
   ```
   (UR5e's URDF chain `wrist_3_link → flange → tool0` evaluates to identity
   rotation, so this works out.)

### Pitfall — anchoring on the wrong frame gives a 90° error

The first iteration anchored the live read on
`/World/RG2_Gripper/onrobot_rg2_base_link` (the leaf prim). That prim has its
own xformOp:orient of `(-0.7071, 0, 0, 0.7071)` — a `-90° about Z` rotation
relative to its parent `/World/RG2_Gripper`. **Our URDF already encodes that
`-90° about Z` as the per-mesh `<visual><origin rpy="0 0 -1.5708">` blocks for
the base mesh and left-side links** (and `+1.5708` for right-side links).

If the live-read and the URDF both encode the leaf's own rotation, you double-count
it and get a static TF that's 90° too rotated (which manifests as the gripper
visibly facing the wrong direction in RViz).

The fix: anchor the live read on the **asset root** `/World/RG2_Gripper`, whose
frame == our URDF's `rg2_base_link` frame. After that correction the canonical
TF cleaned up to the expected `rpy=(0, 0, π)`.

Verification gate (CONTEXT.md D-9c): the static TF must match the live Isaac
Sim attach reading within `1 mm / 1°`. Live reading after correction:
`rpy=(1e-5°, -4e-4°, 179.13°)`. The 0.87° from a clean 180° is within
physics-settle noise (one quickstart run produced 7 mrad of float drift before
the joint constraint reached steady state).

## Visual validation set

`compliant_insertion_studio/docs/phase7_visual_validation/`:

| File | Pose | Purpose |
|------|------|---------|
| `01_initial_pose.png` | UR5e default (joints all zero) | Initial confirmation, **WRONG TF (rpy=0)** — gripper rotation off |
| `03_pose_facedown.png` | Face-down, joints `(0, -π/2, π/2, -π/2, -π/2, 0)` | Initial verification, **WRONG TF (rpy=π/2)** — gripper still rotated 90° wrong (the fix iteration) |
| `07_corrected_180z.png` | Same face-down pose | **CORRECT TF (rpy=π)** — gripper hangs naturally below wrist |
| `08_corrected_pose_forward.png` | Forward-pointing | Sweep test pose 1 — gripper follows |
| `09_corrected_pose_side.png` | Side-pointing | Sweep test pose 2 — gripper follows |
| `10_corrected_wrist3_rotated.png` | Face-down, wrist_3 rotated +90° | Confirms wrist_3 rotation propagates into the gripper as expected |

## Adapting to a different gripper

The architecture is gripper-agnostic. To swap the OnRobot RG2 for another gripper:

1. Generate a URDF for the new gripper. Easiest path: `extract_rg2_urdf.py`
   adapted to the new USD (or use Isaac Sim's URDF exporter if the new gripper
   has no kinematic loops, or lift a community URDF if one exists).
2. Drop the new URDF + meshes under
   `compliant_insertion_studio/urdf/<gripper_name>/`.
3. Edit `compliant_insertion_studio/launch/ur5e_with_rg2.launch.py`:
   change `RG2_URDF_PATH` and (probably) the default `tf_rpy` / `tf_xyz`.
4. Determine the `tool0 → <gripper>_base_link` TF using the live-Isaac-Sim
   procedure above (or measure off the gripper's mounting drawing if the
   manufacturer publishes one).
5. Spin up + visually verify in RViz.

No Python primitives change. The `GRIPPER_CENTER_TOOL_OFFSET` in
`primitives/shared/config.py` stays the source of truth for force-mode /
trajectory math regardless of which gripper is on the flange.

## What this does NOT cover (deferred — see CONTEXT.md `<deferred>`)

- Migrate `GRIPPER_CENTER_TOOL_OFFSET` to URDF FK lookup (51-call-site refactor).
- Mimic prismatic finger pair fed by `/gripper_width` topic (animated grip width in RViz).
- ros2_control / MoveIt planning-scene integration of RG2 collision geometry
  (the explicit reason ROADMAP success criterion #3 was dropped).
- Xacro-into-UR-tree integration (would unlock the above; v2 alternative).

## Files

```
compliant_insertion_studio/
├── launch/ur5e_with_rg2.launch.py            ← compose UR + RG2 + RViz
├── rviz/ur5e_with_rg2.rviz                   ← dual-RobotModel RViz config
├── urdf/rg2/
│   ├── rg2.urdf                              ← single-link URDF (4.7 KB)
│   └── meshes/*.obj                          ← 7 visual + 7 collision OBJs
├── scripts/
│   ├── extract_rg2_urdf.py                   ← USD→URDF (the actual builder)
│   └── export_rg2_urdf_standalone.py         ← Isaac Sim SRL exporter driver
│                                                (kept as reference; fails on
│                                                RG2 due to kinematic loops)
└── docs/
    ├── gripper_urdf.md                       ← this file
    └── phase7_visual_validation/*.png        ← screenshot evidence

_references/articles/rg2_attach_transform.md  ← RESEARCH-01 deep notes
.planning/phases/07-gripper-urdf-rviz/07-CONTEXT.md  ← phase decisions
```
