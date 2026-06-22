#!/usr/bin/env python3
"""
Verify Clearance - Checks if assembly objects have enough space for gripper to grab them.

Related files (clearance check pipeline):
  - queries/verify_clearance.py    (this file) — ROS query that checks poses and computes clearance
  - triggers/pre_assembly_check.py — handles failure by invoking human elicitation
  - elicitations/fix_scene.py      — Pydantic schemas for human interaction

For real-world assemblies, this script verifies:
1. All required objects for the assembly are present in the scene
2. Objects that are not yet assembled have sufficient clearance
3. No objects are too close to each other (gripper collision risk)

The algorithm:
1. Load assembly configuration by base name
2. Get current object poses from the scene
3. For each object in the assembly:
   a. Check if it's present in the scene
   b. Check if it's already assembled (using verify_assembly logic)
   c. If not assembled, compute distances to all neighboring objects
   d. Verify clearance is sufficient for gripper to operate
"""

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import time
import sys
import os


# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.data_path_finder import get_assembly_data_dir, get_symmetry_dir, find_assembly_json_by_base_name
from primitives.shared.config import GRIPPER_CENTER_TOOL_OFFSET

# Configuration (auto-discovered)
ASSEMBLY_DATA_DIR = str(get_assembly_data_dir())
SYMMETRY_DIR = str(get_symmetry_dir())
OBJECT_TOPIC_REAL = "/objects_poses_real"

# Gripper dimensions (UR10e RobotIQ Hand-E gripper specifications)
# These are approximate dimensions in meters
GRIPPER_WIDTH = 0.100      # ~10cm effective gripper width for clearance check
GRIPPER_DEPTH = 0.150      # ~15cm gripper depth (fingertip to palm)
GRIPPER_FINGER_LENGTH = 0.120  # ~12cm finger length
GRIPPER_LENGTH = 0.200     # ~20cm total length (palm to fingertip)

# Clearance margin for safety (meters)
CLEARANCE_MARGIN = 0.01    # 1cm safety margin

# --- grip_widths-backed clearance (replaces the legacy AABB + constant-GRIPPER_WIDTH check) -----------
# The real finger-mesh arc-sweep predicate (grip_widths) needs open3d + the bundle's mesh/FK assets and
# is hardcoded REPO=Path("."), so it cannot be imported into the ROS python env. We subprocess it via
# `uv run` (cwd = the bundle) through grip_widths_cli.py — the SAME seam control_gripper uses. The CLI's
# pre_grasp / pre_insert batch modes evaluate every candidate per part. See the design note:
# aruco-grasp-annotator/.local/plans/insert-feasibility-preflight-design-20260622.md.
import subprocess
import tempfile

_GRIP_WIDTHS_MARKER = "__GRIP_WIDTHS_JSON__"


def _run_bundle_cli(request, timeout=180.0):
    """Shell to grip_widths_cli.py in the aruco-tc bundle. Returns (result_dict, error)."""
    bundle = os.environ.get("ARUCO_TC_BUNDLE", os.path.expanduser("~/aruco-tc-bundle"))
    cli = os.path.join(bundle, ".local/scripts/grip_widths_cli.py")
    if not os.path.exists(cli):
        return None, f"grip_widths_cli.py not found at {cli} (set ARUCO_TC_BUNDLE to the bundle root)."
    tf = tempfile.NamedTemporaryFile(mode='w', delete=False, dir='/tmp', suffix='.json')
    try:
        json.dump(request, tf)
        tf.close()
        try:
            proc = subprocess.run(["uv", "run", "--no-project", cli, tf.name],
                                  cwd=bundle, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, "grip_widths_cli timeout"
        line = next((l for l in proc.stdout.splitlines() if l.startswith(_GRIP_WIDTHS_MARKER)), None)
        if line is None:
            return None, f"no result line (rc={proc.returncode}); stderr_tail={proc.stderr[-400:]!r}"
        return json.loads(line[len(_GRIP_WIDTHS_MARKER):].strip()), None
    finally:
        try:
            os.unlink(tf.name)
        except OSError:
            pass


def load_object_dimensions(object_name, assembly_config):
    """
    Load dimensions for an object from assembly config.

    Expected format in component:
    "dimensions": {
        "x": width_in_meters,
        "y": depth_in_meters,
        "z": height_in_meters
    }

    If not found, returns a default small object size.
    """
    for component in assembly_config.get('components', []):
        comp_name = component.get('name', '')
        if comp_name == object_name:
            dims = component.get('dimensions', {})
            if dims:
                return {
                    'x': dims.get('x', 0.05),
                    'y': dims.get('y', 0.05),
                    'z': dims.get('z', 0.05)
                }

    # Default small object dimensions if not found
    return {'x': 0.05, 'y': 0.05, 'z': 0.05}


class VerifyClearance(Node):
    def __init__(self, base_name=None, mode='real'):
        super().__init__('verify_clearance')

        self.base_name_input = base_name  # User-provided base name (for finding JSON file)
        self.base_name = None  # Actual base name from JSON (identified by type="board")
        self.assembly_json_file = None
        self.assembly_config = {}
        self.symmetry_dir = SYMMETRY_DIR
        self.mode = mode

        # Load assembly configuration
        if base_name is not None:
            self.assembly_config = self.load_assembly_config(base_name)
            # Auto-detect base from assembly config
            self.base_name = self.get_base_from_config()

        # Subscribers
        object_topic = OBJECT_TOPIC_REAL if mode == 'real' else "/objects_poses_sim"
        self.object_sub = self.create_subscription(TFMessage, object_topic, self.object_callback, 10)

        # Store current poses
        self.current_poses = {}
        self.pose_received = False

    def load_assembly_config(self, base_name=None):
        """Load assembly configuration from JSON file."""
        if base_name is not None:
            json_file = find_assembly_json_by_base_name(base_name, ASSEMBLY_DATA_DIR, self.get_logger())
            if json_file:
                self.assembly_json_file = json_file
            else:
                self.get_logger().error(f"Could not find assembly JSON for base '{base_name}'")
                return {}

        json_file = self.assembly_json_file
        if json_file is None:
            json_file = find_assembly_json_by_base_name("base", ASSEMBLY_DATA_DIR, self.get_logger())
            if json_file is None:
                self.get_logger().error("No assembly JSON file found")
                return {}

        try:
            with open(json_file, 'r') as f:
                config = json.load(f)
                return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.get_logger().error(f"Error loading assembly config from {json_file}: {e}")
            return {}

    def get_base_from_config(self):
        """
        Identify the base object from assembly config by looking for type="board".
        Returns the name of the base object.
        """
        for component in self.assembly_config.get('components', []):
            comp_type = component.get('type', '')
            if comp_type == 'board':
                base_name = component.get('name', '')
                self.get_logger().info(f"Identified base object from config: {base_name}")
                return base_name

        # Fallback: use the input base name if no board type found
        self.get_logger().warn(f"No component with type='board' found, using input name: {self.base_name_input}")
        return self.base_name_input

    def object_callback(self, msg):
        """Callback for object poses"""
        for transform in msg.transforms:
            frame_id = transform.child_frame_id
            self.current_poses[frame_id] = transform
        self.pose_received = True

    def transform_to_matrix(self, transform):
        """Convert ROS Transform to 4x4 transformation matrix"""
        t = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
        q = np.array([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w])
        r = R.from_quat(q)
        rotation_matrix = r.as_matrix()
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = t
        return T

    def get_object_position(self, object_name):
        """Get current position of object in world frame"""
        if object_name not in self.current_poses:
            return None

        transform = self.current_poses[object_name]
        return np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ])

    def get_object_orientation(self, object_name):
        """Get current orientation of object in world frame"""
        if object_name not in self.current_poses:
            return None

        transform = self.current_poses[object_name]
        q = np.array([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ])
        return R.from_quat(q).as_matrix()

    def get_assembly_components(self):
        """Get list of all components in the assembly"""
        return [comp.get('name', '') for comp in self.assembly_config.get('components', [])]

    def live_poses_dict(self):
        """Current scene poses as {name: {position{x,y,z}, quaternion{x,y,z,w}}} (metres, world) —
        the shape grip_widths_cli expects for the pre_grasp live scene."""
        out = {}
        for name, tr in self.current_poses.items():
            t = tr.transform.translation
            q = tr.transform.rotation
            out[name] = {"position": {"x": t.x, "y": t.y, "z": t.z},
                         "quaternion": {"x": q.x, "y": q.y, "z": q.z, "w": q.w}}
        return out

    def run_feasibility(self, do_pre_grasp, do_pre_insert, parts):
        """grip_widths-backed gates (replace the AABB clearance check). pre_grasp = live-pose pick
        access; pre_insert = GT-target insert feasibility (z_floor=part_bottom) + write the grasp-id
        manifest. Returns dict(report, manifest_path, infeasible_grasp, infeasible_insert, error)."""
        report, infeasible_grasp, infeasible_insert, manifest_path = {}, [], [], None
        if do_pre_grasp:
            res, err = _run_bundle_cli({"mode": "pre_grasp", "live_poses": self.live_poses_dict(),
                                        "base": self.base_name, "parts": parts})
            if err:
                return dict(report=report, manifest_path=None, infeasible_grasp=[],
                            infeasible_insert=[], error=f"pre_grasp: {err}")
            report["pre_grasp"] = res
            infeasible_grasp = [p["part"] for p in res["parts"] if p["verdict"] != "feasible"]
        if do_pre_insert:
            res, err = _run_bundle_cli({"mode": "pre_insert",
                                        "components": self.assembly_config.get('components', []),
                                        "base": self.base_name, "z_floor": "part_bottom", "parts": parts})
            if err:
                return dict(report=report, manifest_path=None, infeasible_grasp=infeasible_grasp,
                            infeasible_insert=[], error=f"pre_insert: {err}")
            report["pre_insert"] = res
            infeasible_insert = [p["part"] for p in res["parts"] if p["verdict"] != "feasible"]
            manifest_path = self._write_manifest(res)
        return dict(report=report, manifest_path=manifest_path, infeasible_grasp=infeasible_grasp,
                    infeasible_insert=infeasible_insert, error=None)

    def _write_manifest(self, pre_insert_res):
        """Write the up-front insert-feasibility manifest the assembly agent reasons over. Results-file
        shape ({"assembly_order": [ ... ]}), per part: verdict + the FEASIBLE candidate set + the DENIED
        set (with reasons). It does NOT assert a single 'best' pick — the gate's clearance filter has no
        notion of best; the AGENT analyses this CAD-derived info and CHOOSES (and on a gate/verify hit,
        restores + re-picks from the feasible set). This is what replaces Phase-1 empirical discovery."""
        base = os.getenv("MCP_CLIENT_OUTPUT_DIR", "").strip()
        if not base:
            self.get_logger().warn("MCP_CLIENT_OUTPUT_DIR unset — pre_insert manifest not written")
            return None
        order = {c["name"]: c.get("assembly_order", 0) for c in self.assembly_config.get('components', [])}
        assembly_id = self.assembly_config.get("assembly_id", self.base_name)

        def _fmt(c):
            # self-describing fields for the agent; only present keys (denied 'not top-down' carry no widths)
            out = {"candidate_id": c.get("candidate_id"), "axis": c.get("axis"),
                   "grip_width_mm": c.get("obj_w"), "clearance_mm": c.get("W_clear"),
                   "margin_mm": c.get("margin"), "reason": c.get("reason")}
            return {k: v for k, v in out.items() if v is not None}

        entries = []
        for p in pre_insert_res["parts"]:
            entries.append({"object_name": p["part"], "assembly_order": order.get(p["part"], 0),
                            "verdict": p["verdict"],
                            "clearing_candidates": [_fmt(c) for c in p["clearing_candidates"]],
                            "denied_candidates": [_fmt(c) for c in p.get("denied_candidates", [])]})
        entries.sort(key=lambda e: e["assembly_order"])
        manifest = {
            "assembly_id": assembly_id, "base_name": self.base_name,
            "generated_by": "verify_clearance --pre-insert (analytic; z_floor=part_bottom)",
            "note": ("clearing_candidates = grasps that clear neighbours at the seated pose (feasible). "
                     "No single 'best' is asserted — choose from these by reasoning over the geometry "
                     "(grip_width_mm, clearance_mm, margin_mm, axis). denied_candidates must NOT be used. "
                     "On a gate-deny or a placement/verify failure: restore the scene, pick a different "
                     "clearing candidate, and re-run (backtracking is expected)."),
            "assembly_order": entries}
        from pathlib import Path
        path = Path(base) / "logs" / f"InsertFeasibility_{assembly_id}_results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2))
        return str(path)

    def is_object_present(self, object_name):
        """Check if object is in the scene"""
        return object_name in self.current_poses

    def get_bounding_box_aabb(self, object_name, position, rotation=None):
        """
        Compute axis-aligned bounding box (AABB) for an object.

        Args:
            object_name: Name of the object
            position: Center position [x, y, z]
            rotation: Rotation matrix (3x3), if None assumes identity

        Returns:
            Dictionary with min and max corners of AABB
        """
        dims = load_object_dimensions(object_name, self.assembly_config)

        # Half dimensions
        half_x = dims['x'] / 2.0
        half_y = dims['y'] / 2.0
        half_z = dims['z'] / 2.0

        # Corner points in object local frame
        corners_local = np.array([
            [-half_x, -half_y, -half_z],
            [half_x, -half_y, -half_z],
            [-half_x, half_y, -half_z],
            [half_x, half_y, -half_z],
            [-half_x, -half_y, half_z],
            [half_x, -half_y, half_z],
            [-half_x, half_y, half_z],
            [half_x, half_y, half_z],
        ])

        # Transform to world frame if rotation provided
        if rotation is not None:
            corners_world = (rotation @ corners_local.T).T
        else:
            corners_world = corners_local

        # Translate by position
        corners_world = corners_world + position

        # Compute AABB
        aabb_min = np.min(corners_world, axis=0)
        aabb_max = np.max(corners_world, axis=0)

        return {
            'min': aabb_min,
            'max': aabb_max,
            'center': position,
            'dims': dims
        }

    def compute_min_distance_between_aabbs(self, aabb1, aabb2):
        """
        Compute minimum distance between two axis-aligned bounding boxes.

        Returns the minimum gap between the boxes. Negative value indicates overlap.
        """
        # Check distance along each axis
        distances = []

        for axis in range(3):
            min1, max1 = aabb1['min'][axis], aabb1['max'][axis]
            min2, max2 = aabb2['min'][axis], aabb2['max'][axis]

            # Distance along this axis
            if max1 < min2:
                distances.append(min2 - max1)  # aabb1 is before aabb2
            elif max2 < min1:
                distances.append(min1 - max2)  # aabb2 is before aabb1
            else:
                distances.append(0)  # Overlap along this axis

        # Minimum distance is the maximum gap (considering overlap)
        # If any axis has overlap (distance = 0), the minimum distance is 0
        min_dist = max(distances)
        return min_dist

    def check_clearance_for_object(self, object_name, exclude_objects=None):
        """
        Check if an object has sufficient clearance from others for gripper access.

        Args:
            object_name: Name of the object to check
            exclude_objects: List of object names to exclude from clearance check (e.g., assembled objects)

        Returns:
            Dictionary with clearance status and details
        """
        if exclude_objects is None:
            exclude_objects = []

        pos = self.get_object_position(object_name)
        rot = self.get_object_orientation(object_name)

        if pos is None:
            return {
                'status': 'not_present',
                'object': object_name,
                'message': f'Object {object_name} not found in scene'
            }

        aabb_target = self.get_bounding_box_aabb(object_name, pos, rot)

        # Check distances to all other objects
        min_clearance = float('inf')
        closest_object = None
        clearance_issues = []

        for other_name in self.current_poses.keys():
            # Skip self
            if other_name == object_name:
                continue

            # Skip excluded objects (e.g., assembled objects)
            should_skip = False
            for excluded in exclude_objects:
                if other_name == excluded:
                    should_skip = True
                    break
            if should_skip:
                continue

            # Get other object's AABB
            other_pos = self.get_object_position(other_name)
            other_rot = self.get_object_orientation(other_name)

            if other_pos is None:
                continue

            aabb_other = self.get_bounding_box_aabb(other_name, other_pos, other_rot)

            # Compute distance
            distance = self.compute_min_distance_between_aabbs(aabb_target, aabb_other)

            if distance < min_clearance:
                min_clearance = distance
                closest_object = other_name

            # Check against gripper width + margin
            required_clearance = GRIPPER_WIDTH / 2.0 + CLEARANCE_MARGIN
            if distance < required_clearance:
                clearance_issues.append({
                    'obstacle': other_name,
                    'distance': distance,
                    'required': required_clearance,
                    'deficit': required_clearance - distance
                })

        # Determine status
        if min_clearance == float('inf'):
            # No other objects in scene
            status = 'ok_isolated'
            message = 'Object is isolated in scene (sufficient clearance)'
        elif min_clearance >= GRIPPER_WIDTH / 2.0 + CLEARANCE_MARGIN:
            status = 'ok'
            message = f'Sufficient clearance (minimum gap: {min_clearance:.4f}m from {closest_object})'
        elif min_clearance >= 0:
            status = 'marginal'
            message = f'Marginal clearance (minimum gap: {min_clearance:.4f}m from {closest_object})'
        else:
            status = 'collision'
            message = f'Objects overlapping with {closest_object}'

        return {
            'status': status,
            'object': object_name,
            'message': message,
            'min_clearance': min_clearance if min_clearance != float('inf') else None,
            'closest_object': closest_object,
            'clearance_issues': clearance_issues,
            'gripper_required_width': GRIPPER_WIDTH / 2.0 + CLEARANCE_MARGIN
        }

    def verify_assembly_status(self, object_name, base_name):
        """
        Check if object is already assembled to base.
        Returns True if assembled, False if not.
        """
        # Get current poses
        obj_pos = self.get_object_position(object_name)
        base_pos = self.get_object_position(base_name)
        obj_rot = self.get_object_orientation(object_name)
        base_rot = self.get_object_orientation(base_name)

        if obj_pos is None or base_pos is None:
            return False

        # Get target position relative to base
        for component in self.assembly_config.get('components', []):
            comp_name = component.get('name', '')
            if comp_name == object_name:
                target_pos = component.get('position', {})
                target_pos = np.array([
                    target_pos.get('x', 0),
                    target_pos.get('y', 0),
                    target_pos.get('z', 0)
                ])
                break
        else:
            return False

        # Transform target position to world frame
        if base_rot is not None:
            target_world = base_pos + base_rot @ target_pos
        else:
            target_world = base_pos + target_pos

        # Check if current position is close to target
        position_error = np.linalg.norm(obj_pos - target_world)
        ASSEMBLY_TOLERANCE = 0.01  # 1cm tolerance

        return position_error < ASSEMBLY_TOLERANCE

    def run_verification(self):
        """Run the clearance verification and return results"""
        # Wait for pose data
        start_time = time.time()
        timeout = 5.0
        while not self.pose_received and (time.time() - start_time) < timeout:
            try:
                rclpy.spin_once(self, timeout_sec=0.1)
            except TypeError:
                # Fallback for older rclpy versions
                rclpy.spin_once(self)

        if not self.pose_received or not self.current_poses:
            return {
                'error': 'No pose data received',
                'base_input': self.base_name_input,
                'base_detected': self.base_name
            }

        # Verify base was properly detected
        if not self.base_name:
            return {
                'error': 'Could not identify base object from assembly configuration',
                'base_input': self.base_name_input,
                'assembly_file': self.assembly_json_file
            }

        results = {
            'base_input': self.base_name_input,
            'base_detected': self.base_name,
            'assembly_file': self.assembly_json_file,
            'objects_in_scene': list(self.current_poses.keys()),
            'components': [],
            'summary': {}
        }

        # Get all components in assembly
        components = self.get_assembly_components()

        if not components:
            results['error'] = f'No components found in assembly for base {self.base_name}'
            return results

        present_count = 0
        assembled_count = 0
        clearance_ok_count = 0
        missing_objects = []
        clearance_issues = []

        # First pass: identify missing, present, and assembled objects
        object_status = {}  # normalized_name -> 'missing' | 'base' | 'assembled' | 'unassembled'
        assembled_objects = []  # List of assembled object names

        for obj_name in components:
            normalized_name = obj_name

            # Check if object is present
            if not self.is_object_present(normalized_name):
                object_status[normalized_name] = 'missing'
                missing_objects.append(normalized_name)
            elif normalized_name == self.base_name:
                object_status[normalized_name] = 'base'
                present_count += 1
            else:
                # Check assembly status for non-base objects
                is_assembled = self.verify_assembly_status(normalized_name, self.base_name)
                if is_assembled:
                    object_status[normalized_name] = 'assembled'
                    assembled_objects.append(normalized_name)
                    assembled_count += 1
                    present_count += 1
                else:
                    object_status[normalized_name] = 'unassembled'
                    present_count += 1

        # Second pass: Check clearance and build results
        for obj_name in components:
            normalized_name = obj_name
            status = object_status[normalized_name]

            if status == 'missing':
                results['components'].append({
                    'name': normalized_name,
                    'status': 'missing',
                    'message': f'Object {normalized_name} not present in scene'
                })
            elif status == 'base':
                # Base is fixed - no clearance check needed for the base itself
                results['components'].append({
                    'name': normalized_name,
                    'status': 'base',
                    'message': f'Base object {normalized_name} is the reference frame (fixed position)'
                })
            elif status == 'assembled':
                results['components'].append({
                    'name': normalized_name,
                    'status': 'assembled',
                    'message': f'Object {normalized_name} is assembled to base (clearance check skipped)'
                })
            else:  # status == 'unassembled'
                # AABB clearance check RETIRED (2026-06-22): clearance is now the grip_widths-backed
                # --pre-grasp / --pre-insert gates (real finger-mesh, run in main via run_feasibility).
                # Here we only record presence/status + collect the unassembled set those gates act on.
                results['components'].append({
                    'name': normalized_name,
                    'status': 'unassembled'
                })

        # Summary. Clearance verdicts now come from the grip_widths gates (run_feasibility in main),
        # not from this presence pass — so we surface the unassembled set those gates operate on.
        unassembled_objects = [n for n, s in object_status.items() if s == 'unassembled']
        results['summary'] = {
            'total_components': len(components),
            'present_in_scene': present_count,
            'missing': len(missing_objects),
            'assembled': assembled_count,
            'unassembled': present_count - assembled_count,
            'missing_objects': missing_objects,
            'unassembled_objects': unassembled_objects
        }

        return results


def output_result(result, pretty=False):
    """Output JSON result"""
    if pretty:
        print(json.dumps(result, indent=2, default=str))
    else:
        print("__RESULT_JSON__")
        print(json.dumps(result, default=str))
        print("__END_RESULT_JSON__")


def main():
    parser = argparse.ArgumentParser(
        description='Verify clearance for assembly objects in real world'
    )
    parser.add_argument('--base-name', required=True, help='Base object name (e.g., base1)')
    parser.add_argument('--mode', choices=['sim', 'real'], default='real',
                        help='Simulation or real robot mode')
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='Timeout for waiting for poses (seconds)')
    parser.add_argument('--pretty', action='store_true',
                        help='Pretty print output for terminal readability')
    parser.add_argument('--pre-grasp', action='store_true',
                        help='Run ONLY the pre_grasp gate (live-pose pick access). Default: both gates.')
    parser.add_argument('--pre-insert', action='store_true',
                        help='Run ONLY the pre_insert gate (GT-target insert feasibility + manifest). '
                             'Default: both gates.')
    args = parser.parse_args()

    # Default: run both gates, in order pre_grasp -> pre_insert. A flag selects only that one.
    do_pre_grasp = args.pre_grasp or not (args.pre_grasp or args.pre_insert)
    do_pre_insert = args.pre_insert or not (args.pre_grasp or args.pre_insert)

    # Convert base_name from hyphenated to underscored for internal use
    base_name = args.base_name.replace('-', '_')

    rclpy.init()

    success = False
    error_msg = None
    simple_result = {}

    try:
        node = VerifyClearance(base_name=base_name, mode=args.mode)
        detailed_result = node.run_verification()

        # Check for errors in detailed result
        if 'error' in detailed_result:
            node.destroy_node()
            rclpy.shutdown()
            error_msg = detailed_result['error']
            simple_result = {
                'result': 'failure',
                'base_name': base_name,
                'ready_for_assembly': False,
                'error': error_msg
            }
        else:
            summary = detailed_result.get('summary', {})
            missing = summary.get('missing_objects', [])
            unassembled = summary.get('unassembled_objects', [])

            # grip_widths-backed gates on the present-but-unassembled parts (node still alive for poses).
            feas = node.run_feasibility(do_pre_grasp, do_pre_insert, unassembled)
            node.destroy_node()
            rclpy.shutdown()

            grasp_infeasible = feas['infeasible_grasp']
            insert_infeasible = feas['infeasible_insert']
            all_present = len(missing) == 0
            success = (feas['error'] is None and all_present
                       and not grasp_infeasible and not insert_infeasible)

            simple_result = {
                'result': 'success' if success else 'failure',
                'base_name': detailed_result.get('base_detected', base_name),
                'ready_for_assembly': success,
                'gates': {'pre_grasp': do_pre_grasp, 'pre_insert': do_pre_insert},
            }
            if feas['manifest_path']:
                simple_result['insert_manifest'] = feas['manifest_path']

            if not success:
                issues = []
                if feas['error']:
                    issues.append(feas['error'])
                if missing:
                    issues.append(f"{len(missing)} missing")
                    simple_result['missing_objects'] = missing
                if grasp_infeasible:
                    issues.append(f"{len(grasp_infeasible)} not pickable (pre_grasp)")
                    # route to the existing fix_scene elicitation (scene-setup is the human's to fix)
                    simple_result['objects_with_clearance_issues'] = grasp_infeasible
                if insert_infeasible:
                    # insert-infeasible = an assembly DESIGN/order problem (fixing the scene won't help);
                    # surface as failure but NOT via fix_scene. Proper handler is the step-5 wiring.
                    issues.append(f"{len(insert_infeasible)} not insertable (pre_insert)")
                    simple_result['objects_infeasible_insert'] = insert_infeasible
                simple_result['error'] = '; '.join(issues)

        output_result(simple_result, pretty=args.pretty)
        sys.exit(0 if success else 1)

    except Exception as e:
        error_msg = str(e)
        simple_result = {
            'result': 'failure',
            'base_name': base_name,
            'ready_for_assembly': False,
            'error': error_msg
        }
        rclpy.shutdown()
        output_result(simple_result, pretty=args.pretty)
        sys.exit(1)


if __name__ == '__main__':
    main()
