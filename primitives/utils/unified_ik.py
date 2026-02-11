"""
Centralized IK solver with early termination.

Consolidates the repeated IK solving pattern found across multiple primitives
into a single module with consistent early-exit logic and best-result tracking.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

from primitives.utils.ik_solver import forward_kinematics, dh_params, ik_objective_quaternion


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


# Default L-BFGS-B options: relaxed tolerances sufficient for IK (cost threshold is 0.01).
# Default scipy ftol/gtol are ~2.2e-15 which wastes iterations converging far beyond needed.
# ftol=1e-6, gtol=1e-5 still yields cost ~3e-6 (well below 0.01) with ~25% fewer iterations.
_DEFAULT_SOLVER_OPTIONS = {'ftol': 1e-6, 'gtol': 1e-5}


@dataclass
class IKSolverConfig:
    """Configuration for IK solving."""
    cost_threshold: float = 0.01        # Immediate return threshold (tight tolerance)
    acceptable_cost: float = 0.1        # Fallback acceptance threshold
    early_termination: bool = True      # Stop on good solution
    objective_fn: Optional[Callable] = None  # Custom objective; defaults to ik_objective_quaternion
    joint_bounds: Optional[list] = None      # Custom joint bounds; defaults to [(-pi, pi)] * 6
    solver_options: Optional[Dict] = None    # L-BFGS-B options; defaults to relaxed tolerances
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
