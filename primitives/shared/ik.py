"""
Centralized IK module for UR5e robot.

Combines low-level forward kinematics, IK objective functions, the
IKSolver class with early termination, and higher-level convenience
functions (compute_cartesian_waypoints_ik).
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

from primitives.shared.config import DH_PARAMS as dh_params


# ---------------------------------------------------------------------------
# Low-level kinematics
# ---------------------------------------------------------------------------


def rpy_to_matrix(rpy):
    return Rot.from_euler('xyz', rpy, degrees=True).as_matrix()


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


# ---------------------------------------------------------------------------
# IK objective functions
# ---------------------------------------------------------------------------

def ik_objective_quaternion(q, target_pose):
    """
    Improved IK objective using quaternion-based orientation error.
    This avoids gimbal lock issues with Euler angles.
    """
    T_fk = forward_kinematics(dh_params, q)

    # Position error
    pos_error = np.linalg.norm(T_fk[:3, 3] - target_pose[:3, 3])

    # Orientation error using quaternions
    R_fk = Rot.from_matrix(T_fk[:3, :3])
    R_target = Rot.from_matrix(target_pose[:3, :3])
    R_error = R_target * R_fk.inv()
    angle_error = R_error.magnitude()  # This is the geodesic distance on SO(3)

    # Weight position error more heavily
    return 10.0 * pos_error + angle_error


def _make_cached_quaternion_objective(target_pose: np.ndarray, dh: list = None) -> Callable:
    """Create a cached version of ik_objective_quaternion for a specific target pose.

    Pre-computes R_target once instead of calling R.from_matrix() on every
    objective evaluation (~1800 calls per minimize). Benchmarked at 1.43x
    faster per minimize call.
    """
    fk_dh = dh if dh is not None else dh_params
    target_pos = target_pose[:3, 3]
    R_target = Rot.from_matrix(target_pose[:3, :3])

    def objective(q, _target_pose):
        T_fk = forward_kinematics(fk_dh, q)
        pos_error = np.linalg.norm(T_fk[:3, 3] - target_pos)
        R_fk = Rot.from_matrix(T_fk[:3, :3])
        R_error = R_target * R_fk.inv()
        angle_error = R_error.magnitude()
        return 10.0 * pos_error + angle_error

    return objective


# ---------------------------------------------------------------------------
# IKSolver class with early termination
# ---------------------------------------------------------------------------

# L-BFGS-B options for high-precision IK (cost threshold is 0.0025).
# Tightened from previous relaxed tolerances to achieve ~0.25mm position accuracy.
# ftol=1e-7, gtol=1e-6 provides better convergence for tighter cost threshold.
_DEFAULT_SOLVER_OPTIONS = {'ftol': 1e-7, 'gtol': 1e-6}


@dataclass
class IKSolverConfig:
    """Configuration for IK solving."""
    cost_threshold: float = 0.0025      # Immediate return threshold (tight tolerance, ~0.25mm position error)
    acceptable_cost: float = 0.05       # Fallback acceptance threshold (~0.5mm position error)
    early_termination: bool = True      # Stop on good solution
    objective_fn: Optional[Callable] = None  # Custom objective; defaults to ik_objective_quaternion
    joint_bounds: Optional[list] = None      # Custom joint bounds; defaults to [(-pi, pi)] * 6
    solver_options: Optional[Dict] = None    # L-BFGS-B options; defaults to tightened tolerances
    dh_params: Optional[list] = None         # Custom DH params; defaults to standard dh_params


@dataclass
class IKResult:
    """Result from a single IK solve attempt."""
    joint_angles: np.ndarray
    cost: float
    has_collision: bool


class IKSolver:
    """
    IK solver with early termination and best-result tracking.

    Evaluates seed/perturbation combinations sequentially with early exit
    on the first good solution. Tracks the best result seen so far as a
    fallback if no solution meets the tight threshold.
    """

    def __init__(self, config: Optional[IKSolverConfig] = None):
        self.config = config or IKSolverConfig()
        self._best_result: Optional[IKResult] = None
        self._objective_fn = self.config.objective_fn or ik_objective_quaternion
        self._joint_bounds = self.config.joint_bounds or [(-np.pi, np.pi)] * 6
        self._solver_options = self.config.solver_options if self.config.solver_options is not None else _DEFAULT_SOLVER_OPTIONS
        self._dh_params = self.config.dh_params or dh_params
        self._use_cached_objective = self._objective_fn is ik_objective_quaternion

    def _solve_single(
        self,
        seed: np.ndarray,
        target_pose: np.ndarray,
        collision_checker: Optional[Callable] = None,
    ) -> Optional[IKResult]:
        """
        Solve IK for a single seed.

        Args:
            seed: Initial joint angle guess.
            target_pose: 4x4 homogeneous target pose.
            collision_checker: Optional callback(joint_angles) -> bool.
                Returns True if collision detected.

        Returns:
            IKResult if optimization succeeded, None otherwise.
        """
        # Use cached objective when possible (avoids re-creating R_target on every call)
        if self._use_cached_objective:
            obj_fn = _make_cached_quaternion_objective(target_pose, self._dh_params)
        else:
            obj_fn = self._objective_fn

        result = minimize(
            obj_fn, seed, args=(target_pose,),
            method='L-BFGS-B', bounds=self._joint_bounds,
            options=self._solver_options,
        )

        # Use result.fun directly instead of re-evaluating the objective
        cost = float(result.fun)

        if not result.success and cost >= self.config.acceptable_cost:
            return None

        has_collision = False
        if collision_checker is not None:
            has_collision = collision_checker(result.x)

        return IKResult(
            joint_angles=result.x.copy(),
            cost=cost,
            has_collision=has_collision,
        )

    def _update_best(self, ik_result: IKResult) -> bool:
        """
        Update best result. Returns True if this result should trigger
        early termination.
        """
        if ik_result is None or ik_result.has_collision:
            return False

        if ik_result.cost >= self.config.acceptable_cost:
            return False

        is_better = (
            self._best_result is None
            or ik_result.cost < self._best_result.cost
        )

        if is_better:
            self._best_result = ik_result

        # Check for early termination
        if self.config.early_termination and ik_result.cost < self.config.cost_threshold:
            return True

        return False

    def solve(
        self,
        seeds: List[np.ndarray],
        target_pose: np.ndarray,
        collision_checker: Optional[Callable] = None,
        perturbations: int = 1,
        dx: float = 0.001,
    ) -> Optional[np.ndarray]:
        """
        Solve IK across multiple seeds and perturbations with early exit.

        Args:
            seeds: List of initial joint angle guesses.
            target_pose: 4x4 homogeneous target pose.
            collision_checker: Optional callback(joint_angles) -> bool.
            perturbations: Number of x-axis perturbations per seed.
            dx: Position perturbation step size.

        Returns:
            Best joint angles if found, None otherwise.
        """
        self._best_result = None

        for seed in seeds:
            for p in range(perturbations):
                perturbed_pose = target_pose.copy()
                perturbed_pose[0, 3] += p * dx

                ik_result = self._solve_single(seed, perturbed_pose, collision_checker)
                if ik_result is not None:
                    terminated = self._update_best(ik_result)
                    if terminated:
                        return self._best_result.joint_angles

        if self._best_result is not None:
            return self._best_result.joint_angles
        return None

    def solve_collect(
        self,
        seeds: List[np.ndarray],
        target_pose: np.ndarray,
        collision_checker: Optional[Callable] = None,
        perturbations: int = 1,
        dx: float = 0.001,
        max_consecutive_collisions: int = 0,
        perturbation_start: int = 0,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Solve IK and collect ALL valid solutions (for rotate_object).

        Unlike solve(), this does NOT early-terminate and returns
        all collision-free solutions with cost < acceptable_cost.

        Args:
            seeds: List of initial joint angle guesses.
            target_pose: 4x4 homogeneous target pose.
            collision_checker: Optional callback(joint_angles) -> bool.
            perturbations: Number of x-axis perturbations per seed.
            dx: Position perturbation step size.
            max_consecutive_collisions: If > 0 and no solutions found yet,
                bail early after this many consecutive failed attempts
                (collision or non-convergence). 0 = no limit (default).
            perturbation_start: Start perturbation index (default 0).
                Use to skip already-probed perturbations.

        Returns:
            List of (joint_angles, cost) tuples for all valid solutions.
        """
        self._best_result = None
        results = []
        consecutive_failures = 0

        for seed in seeds:
            for p in range(perturbation_start, perturbations):
                perturbed_pose = target_pose.copy()
                perturbed_pose[0, 3] += p * dx

                ik_result = self._solve_single(seed, perturbed_pose, collision_checker)
                if ik_result is not None and not ik_result.has_collision and ik_result.cost < self.config.acceptable_cost:
                    results.append((ik_result.joint_angles, ik_result.cost))
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if (max_consecutive_collisions > 0
                            and consecutive_failures >= max_consecutive_collisions
                            and len(results) == 0):
                        return results

        return results


# ---------------------------------------------------------------------------
# High-level convenience functions
# ---------------------------------------------------------------------------


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
    from scipy.spatial.transform import Slerp

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
