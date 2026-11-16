# A pose from the vision layer (ros-mcp-server compliant_insertion_studio/docs/ARCHITECTURE_SEAMS.md, 2026-10-19):
# what was seen, how sure, how stale, from what. Numbers in it come from measurement, never from this file.
#
# header.frame_id  the frame the pose is in ("base" = UR base frame)
# header.stamp     when the camera saw it (frame ARRIVAL time on the cell PC: the RealSense stamps drift ~1 s)
std_msgs/Header header
string name                         # the part or base, as the assembly graph names it
geometry_msgs/Pose pose
float64[4] sigma                    # x, y, z (m), yaw (rad): one standard deviation, from repeatability data
builtin_interfaces/Time stationary_since   # the camera was still from this time on (poses from a moving camera are ~10 mm off)
string source                       # markers | depth_cad | markers+depth_cad | chained (from a previous read + kinematics)
float64 view_z                      # flange height of the view (m)
string detection_set                # which markers / which surfaces carried the estimate
int32 n                             # samples or points behind it
string symmetry                     # the symmetry group the pose is defined modulo (from the symmetry data)
