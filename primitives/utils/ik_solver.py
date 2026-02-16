import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R


# UR5e DH parameters
dh_params = [
    (0,  0.1625,  0,     np.pi/2),
    (0,  0,      -0.425,  0),
    (0,  0,      -0.3922, 0),
    (0,  0.1333,  0,     np.pi/2),
    (0,  0.0997,  0,    -np.pi/2),
    (0,  0.0996,  0,     0)
]


def rpy_to_matrix(rpy):
    return R.from_euler('xyz', rpy, degrees=True).as_matrix()

def dh_transform(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d],
        [0,   0,        0,       1]
    ])

def forward_kinematics(dh_params, joint_angles):
    T = np.eye(4)
    for i, (theta, d, a, alpha) in enumerate(dh_params):
        T_i = dh_transform(joint_angles[i] + theta, d, a, alpha)
        T = np.dot(T, T_i)
    return T

def ik_objective(q, target_pose):
    T_fk = forward_kinematics(dh_params, q)
    pos_error = np.linalg.norm(T_fk[:3, 3] - target_pose[:3, 3])
    rot_error = np.linalg.norm(T_fk[:3, :3] - target_pose[:3, :3])
    return 1.0 * pos_error + 0.1 * rot_error

def ik_objective_quaternion(q, target_pose):
    """
    Improved IK objective using quaternion-based orientation error.
    This avoids gimbal lock issues with Euler angles.
    """
    T_fk = forward_kinematics(dh_params, q)
    
    # Position error
    pos_error = np.linalg.norm(T_fk[:3, 3] - target_pose[:3, 3])
    
    # Orientation error using quaternions
    R_fk = R.from_matrix(T_fk[:3, :3])
    R_target = R.from_matrix(target_pose[:3, :3])
    R_error = R_target * R_fk.inv()
    angle_error = R_error.magnitude()  # This is the geodesic distance on SO(3)
    
    # Weight position error more heavily
    return 10.0 * pos_error + angle_error

def compute_ik(position, rpy, q_guess=None, max_tries=5, dx=0.001):
    from primitives.utils.unified_ik import IKSolverConfig, IKSolver

    if q_guess is None:
        q6 = -(np.mod(rpy[2] + 180, 360) - 180) # Adjust initial guess based on given yaw.
        q_guess = np.radians([85, -80, 90, -90, -90, q6])

    original_position = np.array(position)

    target_pose = np.eye(4)
    target_pose[:3, 3] = original_position
    target_pose[:3, :3] = rpy_to_matrix(rpy)

    # Use ik_objective (not quaternion) to preserve original behavior
    solver = IKSolver(IKSolverConfig(objective_fn=ik_objective))
    result = solver.solve(
        seeds=[q_guess],
        target_pose=target_pose,
        perturbations=max_tries,
        dx=dx,
    )

    if result is None:
        print(f"IK failed after {max_tries} attempts. Tried perturbing from {original_position}.")
    return result

def compute_ik_robust(position, rpy, max_tries=5, dx=0.001, multiple_seeds=True):
    """
    Enhanced IK solver using quaternion-based error metric.
    Better for non-standard orientations (not [0, 180, yaw]).

    Uses multiple seed evaluation with early termination for faster convergence.

    Args:
        position: [x, y, z] target position
        rpy: [roll, pitch, yaw] target orientation in degrees
        max_tries: Number of position perturbations per seed
        dx: Position perturbation step size
        multiple_seeds: If True, try multiple initial joint configurations

    Returns:
        Joint angles if successful, None otherwise
    """
    from primitives.utils.unified_ik import IKSolverConfig, IKSolver

    original_position = np.array(position)
    target_rot_matrix = rpy_to_matrix(rpy)

    # Create target pose
    target_pose = np.eye(4)
    target_pose[:3, 3] = original_position
    target_pose[:3, :3] = target_rot_matrix

    # Seed configurations to try
    if multiple_seeds:
        seed_configs = [
            np.radians([85, -80, 90, -90, -90, -(np.mod(rpy[2] + 180, 360) - 180)]),
            np.radians([90, -90, 90, -90, -90, rpy[2]]),
            np.radians([0, -90, 90, -90, -90, rpy[2]]),
            np.radians([180, -90, 90, -90, -90, rpy[2]]),
            np.radians([85, -100, 120, -110, -90, rpy[2]]),
            np.radians([85, -60, 60, -90, -90, rpy[2]]),
            np.radians([85, -80, 90, -90, 0, rpy[2]]),
            np.radians([85, -80, 90, -90, -180, rpy[2]]),
            np.radians([85, -70, 80, -100, -90, rpy[2]]),
            np.radians([85, -90, 100, -100, -90, rpy[2]]),
        ]
    else:
        q6 = -(np.mod(rpy[2] + 180, 360) - 180)
        seed_configs = [np.radians([85, -80, 90, -90, -90, q6])]

    print(f"Robust IK: Trying {len(seed_configs)} seed configurations with quaternion-based error...")

    solver = IKSolver(IKSolverConfig())
    result = solver.solve(
        seeds=seed_configs,
        target_pose=target_pose,
        perturbations=max_tries,
        dx=dx,
    )

    if result is not None:
        cost = ik_objective_quaternion(result, target_pose)
        print(f"Robust IK succeeded, cost={cost:.6f}")
        return result

    print(f"Robust IK failed after trying {len(seed_configs)} seed configurations")
    return None


def _min_jerk(tau):
    """Minimum-jerk position fraction: s(tau) for tau in [0, 1].
    Returns (position_frac, velocity_frac) where velocity_frac = ds/dtau.
    Zero velocity and acceleration at tau=0 and tau=1.
    """
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sd = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    return s, sd


def compute_cartesian_waypoints_ik(current_joints, target_z, num_waypoints=60,
                                   target_pos=None, target_orientation=None,
                                   active_joints=None):
    """Compute IK for Cartesian waypoints using Jacobian-based differential IK.

    Uses minimum-jerk position profile (smooth accel/decel at endpoints) with
    one damped pseudo-inverse solve per step.

    Args:
        current_joints: Current joint angles (6,).
        target_z: Target Z height in meters. Ignored if target_pos is provided.
        num_waypoints: Number of uniformly-timed waypoints. Positions follow
            minimum-jerk profile so they're denser at start/end (smooth accel).
        target_pos: Optional [x, y, z] target position. If provided, interpolates
            all three axes instead of Z-only.
        target_orientation: Optional 3x3 rotation matrix for the target orientation.
            If None, the current orientation is maintained.

    Returns:
        List of joint angle arrays, or None on failure.
    """
    from scipy.spatial.transform import Rotation as Rot

    DH_PARAMS = dh_params  # reuse module-level DH params

    def dh_matrix(theta, d, a, alpha):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st*ca, st*sa, a*ct], [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d], [0, 0, 0, 1]
        ])

    def fk(q):
        T = np.eye(4)
        for i, (t0, d, a, al) in enumerate(DH_PARAMS):
            T = T @ dh_matrix(q[i]+t0, d, a, al)
        return T

    def jacobian(q):
        """Compute 6x6 geometric Jacobian."""
        Ts = [np.eye(4)]
        for i, (t0, d, a, al) in enumerate(DH_PARAMS):
            Ts.append(Ts[-1] @ dh_matrix(q[i]+t0, d, a, al))
        p_ee = Ts[-1][:3, 3]
        J = np.zeros((6, 6))
        for i in range(6):
            z_i = Ts[i][:3, 2]
            p_i = Ts[i][:3, 3]
            J[:3, i] = np.cross(z_i, p_ee - p_i)
            J[3:, i] = z_i
        return J

    T_current = fk(current_joints)
    current_pos = T_current[:3, 3].copy()
    R_current = Rot.from_matrix(T_current[:3, :3])
    R_target = Rot.from_matrix(target_orientation) if target_orientation is not None else R_current

    # Determine target position
    if target_pos is not None:
        target_pos = np.asarray(target_pos, dtype=float)
    else:
        target_pos = np.array([current_pos[0], current_pos[1], target_z])

    d_total = target_pos - current_pos

    print(f"  FK current: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}]")
    print(f"  Target: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}], "
          f"{num_waypoints} waypoints (min-jerk profile)")

    # Compute waypoint positions using minimum-jerk profile
    waypoint_positions = []
    for i in range(1, num_waypoints + 1):
        tau = i / num_waypoints
        s, _ = _min_jerk(tau)
        waypoint_positions.append(current_pos + s * d_total)

    # Interpolate orientation via SLERP
    ori_key_times = [0.0, 1.0]
    ori_key_rots = Rot.from_matrix(np.stack([R_current.as_matrix(), R_target.as_matrix()]))
    from scipy.spatial.transform import Slerp
    slerp = Slerp(ori_key_times, ori_key_rots)

    # Default: activate all 6 joints to avoid singularities
    if active_joints is None:
        active_joints = [0, 1, 2, 3, 4, 5]
    damping = 1e-4

    q = current_joints.copy()
    waypoints = []
    max_pos_err = 0.0
    prev_pos = current_pos.copy()

    for i, wp_pos in enumerate(waypoint_positions):
        tau = (i + 1) / num_waypoints
        dp = wp_pos - prev_pos

        # Orientation step via rotation vector difference
        R_wp = slerp([tau])[0]
        T_check_pre = fk(q)
        R_cur = Rot.from_matrix(T_check_pre[:3, :3])
        ori_step = (R_wp * R_cur.inv()).as_rotvec()

        dx_step = np.zeros(6)
        dx_step[:3] = dp
        dx_step[3:] = ori_step

        J = jacobian(q)
        J_active = J[:, active_joints]
        JtJ = J_active.T @ J_active + damping * np.eye(len(active_joints))
        dq_active = np.linalg.solve(JtJ, J_active.T @ dx_step)

        for j, idx in enumerate(active_joints):
            q[idx] += dq_active[j]
        prev_pos = wp_pos.copy()

        # FK correction every step to minimize drift
        T_check = fk(q)
        pos_err = np.linalg.norm(T_check[:3, 3] - wp_pos)
        max_pos_err = max(max_pos_err, pos_err)

        if pos_err > 0.00025:  # correct if drift > 0.25mm
            dx_correction = np.zeros(6)
            dx_correction[:3] = wp_pos - T_check[:3, 3]
            ori_err_vec = (R_wp * Rot.from_matrix(T_check[:3, :3]).inv()).as_rotvec()
            dx_correction[3:] = ori_err_vec
            J2 = jacobian(q)
            J2_active = J2[:, active_joints]
            JtJ2 = J2_active.T @ J2_active + damping * np.eye(len(active_joints))
            dq_corr = np.linalg.solve(JtJ2, J2_active.T @ dx_correction)
            for j, idx in enumerate(active_joints):
                q[idx] += dq_corr[j]

        waypoints.append(q.copy())

    # Final verification
    T_final = fk(waypoints[-1])
    pos_err = np.linalg.norm(T_final[:3, 3] - target_pos) * 1000
    ori_err = np.degrees((R_target * Rot.from_matrix(T_final[:3, :3]).inv()).magnitude())
    print(f"  Final IK err: pos={pos_err:.2f}mm, ori={ori_err:.4f}° (max drift={max_pos_err*1000:.2f}mm)")

    if pos_err > 5.0:
        print(f"  WARNING: Final position error too high ({pos_err:.1f}mm)")
        return None

    return waypoints
