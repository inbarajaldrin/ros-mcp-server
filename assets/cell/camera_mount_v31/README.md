# Camera mount v3.1 + D415, as carried by the RG2 (since 2026-11-10)

All in `rg2_base_link`, millimetres. Robot chain (prompt-to-assembly `robots/ur5e_rg2/composite.yaml`): tool0 -> quick_changer_link
rpy (0, 0, -pi/2); quick_changer_link -> rg2_base_link xyz (0, 0, 0.0181) rpy (0, -pi/2, 0).

| file | source |
|---|---|
| `V31_upper_saddle_camera_tower_assembly_coordinates.stl`, `V31_lower_saddle_captive_M4_assembly_coordinates.stl` | the operator's robo-chain job `.scratchpad/jobs/rg2-camera-mount/cad/` (Mac), copied unchanged |
| `D415_body_rg2_base_link.stl` | the same job's final collision BREPs `tools/evidence/v31/final_pairs/brep/D415_camera_solid_00{1..4}_fixed.brep`, tessellated (OCP 7.9.3, 0.2 mm), moved into rg2_base_link by fitting that job's saddle BREPs onto the two saddle STLs above (rms 0.10 mm, an axis permutation + translation); extent 25.9 x 99.0 x 26.1 mm, centre y 35 mm = the rear-screw datum |

Colour optical frame in rg2_base_link: xyz (65.82, 0, 88.91) mm, quat xyzw (0.612372, 0.612372, 0.353553, 0.353553) (TF-FOR-AGENT.md).
Used by `ablations/handeye_charuco/check_views.py` (clearance of everything the arm carries to the table and the arm links).
