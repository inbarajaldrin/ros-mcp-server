# Seams for sim2real: planner / drive-mate / execute-mate / vision (decided 2026-10-19)

Decided with the operator in a grill session after reading the sim side (prompt2assembly: `lab_tools.execute_mate`,
`mates.py`, the delivered `ros_cell.py` / `planner.py`), the installed NVIDIA insertion stack
(`~/Documents/isaac_manipulator`: `isaac_ros_manipulation_ur_dnn_policy`, the gear-assembly orchestrator; the
Isaac Lab `deploy/gear_assembly` env) and Fabrica (IsaacGymEnvs lineage) on dual-a4500. Shapes are fixed here;
every number (sigmas, clearances, force values) comes from the per-part experiments, never from this document.

## The facts the seams rest on

- In prompt2assembly **`execute_mate` authors no motion.** The planner drives the whole job; `execute_mate`
  certifies the mate against its declared acceptance (order satisfied, predecessors unmoved, relative pose inside
  the accepted seated set modulo the symmetry the task allows — FMB1: half-widths 1 mm, class `insertion`,
  fastening `free`, judged at rest) and journals it, refusing by name. The mate's closing trajectory is a fact
  of the design in the assembly graph.
- The delivered twin planner talks to a small ROS surface: `/{token}/joint_states`, `/objects/poses`
  (TFMessage), `FollowJointTrajectory`, a `Grip` action, the `ExecuteMate` action, plus its own IK, grasp
  choice and a position-controlled seat correction (`_correct_seat`, 0.55 mm) that only works in the twin.
- NVIDIA's deployed RL insertion (UR10e, Isaac ROS) is a ROS `Insert` action: goal = target pose from a
  separate perception pipeline; observation encoder (joint_pos, joint_vel, goal pose; stale joint states
  rejected > 10 ms) → rsl_rl policy → action decoder (delta joint positions × 0.025) → impedance controller
  (switched in by the orchestrator) → status checker (TF distance < 13 mm, 60 s). Trained with a ±5 mm
  constant-per-episode noise on the target pose; no force/torque in the observation; termination on tilt > 7°.
  Fabrica (Factory lineage) is the same shape with the reversed disassembly path as an observation.
- Our inserts command UR `force_mode_controller` through `SetForceMode` (wrench + selection vector) a few
  times a second; there is no Cartesian impedance servo on the real cell.

## The layers and what crosses each seam

| layer | owns | interface | must stay out |
| --- | --- | --- | --- |
| **vision** | base and part poses, symmetry-aware; the one held-part read after a grasp; the certification measurement | `VisionPose{T (base frame), sigma[x,y,z,yaw], t, stationary_since, source{view_z, detection_set, n}, symmetry}`; `poses()`, `held(part)` (part in the gripper frame, once per grasp), `certify(mate)` (relative pose vs declared seat, inside the accepted set?); `/objects/poses` | planning decisions, motion, inferring insertion success from an image |
| **planner** (authored; ours = the replay/loop/disassembly as the reference planner) | picks, openings, reorientations, intermediate placements, layouts, order; calls drive-mate then execute_mate | the ROS surface above + `drive_mate` | force control, camera reads, contact recovery |
| **drive-mate** (the insertion driver, classical FSM now, RL later) | bringing the held part into the accepted seated set under uncertainty; stage classification from the trace; bounded recoveries | ROS action `Insert`-style: goal `{mate_id, target: VisionPose, held: VisionPose}`; result `{seated, stage_reached, slide, trace_path, measured_correction}`; reads the graph itself (closing path, acceptance, symmetry) and derives stages/clearances from CAD | part names, per-part constants, learned seat offsets, camera reads |
| **execute_mate** | certification + journal (unchanged platform verb) | `ExecuteMate` action, fed by `vision.certify` | motion |
| **supervisor** (outside the three) | force caps, stage classifier over the trace, retries, light gate, recording, stop rules | — | — |

The actuator under drive-mate is its own seam (`CompliantMotion`): a stiffness + pose-delta target. Backends:
UR force mode (today, behind a shim for the FSM's wrench + selection) and a Cartesian admittance servo (to build
for the RL driver: ≥100 Hz, F/T in the loop, same global force caps; validated with nothing gripped first).

A rule for FMB2/FMB3: nothing in drive-mate, vision or the supervisor may name a part. Mate ids come from the
graph, stages from the seated meshes, symmetry from the symmetry data. Per-part knowledge is allowed only in the
planner layer and in the experiments that populate sigmas.

## Milestones (revised)

| | gate |
| --- | --- |
| M0 loop with today's fixes | 3 clean plain-layout cycles, then the staging loop |
| M1 measure | ease metrics and CAD stages per part per cycle (`insert_ease.py`, `insert_stages.py`, `INSERT_GEOMETRY.md`) |
| M2 base free + measured errors | 3 clean cycles with the base moved ~3 cm, then turned ~5°; held-yaw/xy read in use; sigmas from data |
| M3a drive-mate action + lab surface | the classical FSM behind `/ur5e/drive_mate`; `Grip` adapter over the RG2 bridge; `ExecuteMate` server certifying from vision; `/objects/poses` alias; the replay uses them |
| M4 twin planner transfer | the delivered FMB1 planner runs on the real cell with one substitution (`_correct_seat` → drive-mate) |
| M3b RL driver | admittance servo validated; Isaac ROS encoder→policy→decoder behind the same action; judged by the same ease metrics |

Ownership: everything above is built here. pose6d (the localizer distillation) ports in behind the VisionPose
contract when it is ready, the same way the twin planner ports in behind the ROS surface.
