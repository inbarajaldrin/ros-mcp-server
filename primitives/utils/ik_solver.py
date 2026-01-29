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
