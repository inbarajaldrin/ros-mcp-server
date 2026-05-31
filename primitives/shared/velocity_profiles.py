"""Velocity profile utilities for joint-space trajectory generation.

Provides:
- Single-point profile for simple joint-to-joint moves
- Trapezoidal velocity profile (accel / cruise / decel) for multi-waypoint trajectories
- S-curve velocity profile (sinusoidal acceleration ramps) for jerk-free trajectories
- Min-jerk velocity profile (5th-order polynomial) for maximally smooth trajectories

Reference: Biagiotti & Melchiorri, "Trajectory Planning for Automatic Machines
and Robots", Springer 2008, Ch. 3 — sinusoidal jerk variant and min-jerk.
"""

# TODO(offline-path-optimization — EXECUTION phase only):
#   During EXECUTION (replay / Phase 3) the FULL primitive sequence is known ahead of time,
#   so the per-primitive "decelerate to zero velocity, then re-accelerate from rest"
#   transitions between moves can be merged into continuous, blended motion.
#   Measured 2026-05-31 on a replayed fmb2 assembly: only ~44% of primitive wall-clock is
#   actual arm motion; ~56% is the in-between (subprocess spawn + subscription-ready + IK
#   solve + gripper settle + the stop/restart between every move). The arm motion itself is
#   already near its safe limits — the remaining win is removing the in-between, in execution.
#
#   DO NOT merge during DISCOVERY: the discrete stop-between-steps is a REQUIRED feature there
#   (the agent observes -> thinks -> decides the next primitive between steps). Execution only.
#
#   Approach: take the known sequence's waypoints; keep HARD zero-velocity stops ONLY at the
#   contact-sensitive points (gripper close on grasp, insert/place_down contact, release), and
#   blend the free-space transitions (retract -> traverse -> pre-grasp / pre-insert) into one
#   (or few) continuous FollowJointTrajectory goals with nonzero internal-waypoint velocities.
#   The UR scaled_joint_trajectory_controller only interpolates the points it is given, so the
#   merge must be computed host/server-side (this module generates the profiles to feed it).
#   Prereq: build the deferred intra-primitive timestamps (trajectory send -> action result)
#   to measure the exact mergeable in-between before/after. See Track-B session log 2026-05-31.
import numpy as np


def compute_duration(joint_distance=None, cartesian_distance=None, profile='s_curve',
                     joint_vel=None, joint_accel=None,
                     cart_vel=None, cart_accel=None,
                     min_duration=None, max_duration=None):
    """Compute trajectory duration from path distance and velocity/acceleration limits.

    Accepts joint-space distance, Cartesian distance, or both. When both are
    given, the slower (longer duration) constraint wins.

    The formula depends on the velocity profile:
      - trapezoidal / s_curve (with cruise phase):
            T = d / v_max + v_max / a_max
        This is the time-optimal duration for accelerating to v_max then cruising.
      - min_jerk (no cruise, peak = 1.875 * avg):
            T = 1.875 * d / v_max

    Args:
        joint_distance: Max joint displacement across all joints (rad).
            For multi-waypoint paths, pass the total arc-length or max
            single-joint delta — whichever is more conservative.
        cartesian_distance: Straight-line or path distance in meters.
        profile: 'trapezoidal', 's_curve', or 'min_jerk'.
        joint_vel: Joint velocity limit (rad/s). Defaults to config.JOINT_VEL_LIMIT.
        joint_accel: Joint acceleration limit (rad/s²). Defaults to config.JOINT_ACCEL_LIMIT.
        cart_vel: Cartesian velocity limit (m/s). Defaults to config.CART_VEL_LIMIT.
        cart_accel: Cartesian accel limit (m/s²). Defaults to config.CART_ACCEL_LIMIT.
        min_duration: Floor in seconds. Defaults to config.MIN_DURATION.
        max_duration: Cap in seconds. Defaults to config.MAX_DURATION.

    Returns:
        Duration in seconds (float).
    """
    from primitives.shared.config import (
        JOINT_VEL_LIMIT, JOINT_ACCEL_LIMIT,
        CART_VEL_LIMIT, CART_ACCEL_LIMIT,
        MIN_DURATION, MAX_DURATION,
    )
    if joint_vel is None:
        joint_vel = JOINT_VEL_LIMIT
    if joint_accel is None:
        joint_accel = JOINT_ACCEL_LIMIT
    if cart_vel is None:
        cart_vel = CART_VEL_LIMIT
    if cart_accel is None:
        cart_accel = CART_ACCEL_LIMIT
    if min_duration is None:
        min_duration = MIN_DURATION
    if max_duration is None:
        max_duration = MAX_DURATION

    def _time_for_distance(d, v_max, a_max):
        if d <= 0:
            return 0.0
        if profile == 'min_jerk':
            # Peak speed = 1.875 * avg, so avg = v_max / 1.875
            return 1.875 * d / v_max
        else:
            # Trapezoidal / s-curve: T = d/v + v/a
            # Check if distance is too short to reach v_max
            # (triangle profile: d < v²/a → T = 2*sqrt(d/a))
            if d < v_max * v_max / a_max:
                return 2.0 * np.sqrt(d / a_max)
            return d / v_max + v_max / a_max

    candidates = []
    if joint_distance is not None and joint_distance > 0:
        candidates.append(_time_for_distance(joint_distance, joint_vel, joint_accel))
    if cartesian_distance is not None and cartesian_distance > 0:
        candidates.append(_time_for_distance(cartesian_distance, cart_vel, cart_accel))

    if not candidates:
        return min_duration

    duration = max(candidates)
    return float(np.clip(duration, min_duration, max_duration))


def single_point(target_joints, duration):
    """Create a single-point trajectory for simple joint-space moves.

    Sends only the target position — the UR controller handles smooth
    interpolation from the current state internally.

    Args:
        target_joints: Target joint angles (list/array of 6 floats).
        duration: Time to reach target in seconds.

    Returns:
        List containing one (positions, velocities, time_from_start) tuple.
    """
    return [([float(x) for x in target_joints], [0.0] * len(target_joints), duration)]


def _compute_arc_lengths(all_joint_angles):
    """Compute cumulative arc-length along a joint-space path."""
    cumulative_s = [0.0]
    for i in range(1, len(all_joint_angles)):
        dist = np.linalg.norm(
            np.asarray(all_joint_angles[i]) - np.asarray(all_joint_angles[i - 1])
        )
        cumulative_s.append(cumulative_s[-1] + max(dist, 1e-6))
    return cumulative_s


def _map_arc_lengths_to_profile(all_joint_angles, cumulative_s, total_duration, s_and_v_func):
    """Map arc-length waypoints to (positions, velocities, time) using a speed profile.

    Args:
        all_joint_angles: Joint angle arrays (N waypoints).
        cumulative_s: Cumulative arc-lengths for each waypoint.
        total_duration: Total trajectory time in seconds.
        s_and_v_func: Callable(t) -> (arc_length, speed) defining the profile.

    Returns:
        List of (positions, velocities, time_from_start) tuples.
    """
    n_total = len(all_joint_angles)
    num_joints = len(all_joint_angles[0])

    def find_time_for_s(target_s):
        lo, hi = 0.0, total_duration
        for _ in range(50):
            mid = (lo + hi) / 2
            s_mid, _ = s_and_v_func(mid)
            if s_mid < target_s:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    # Map each waypoint's arc-length to a time, enforcing strictly increasing
    MIN_DT = 1e-4  # 0.1ms minimum gap between waypoints
    waypoint_times = [find_time_for_s(s) for s in cumulative_s]
    waypoint_times[0] = 0.0
    waypoint_times[-1] = total_duration
    for i in range(1, len(waypoint_times)):
        if waypoint_times[i] <= waypoint_times[i - 1] + MIN_DT:
            waypoint_times[i] = waypoint_times[i - 1] + MIN_DT

    result = []
    for i in range(n_total):
        t_i = waypoint_times[i]
        _, speed_scalar = s_and_v_func(t_i)

        if i == 0 or i == n_total - 1:
            velocities = [0.0] * num_joints
        else:
            delta = np.asarray(all_joint_angles[i + 1]) - np.asarray(all_joint_angles[i - 1])
            delta_norm = np.linalg.norm(delta)
            if delta_norm > 1e-8:
                direction = delta / delta_norm
                velocities = [float(speed_scalar * direction[j]) for j in range(num_joints)]
            else:
                velocities = [0.0] * num_joints

        positions = [float(x) for x in all_joint_angles[i]]
        result.append((positions, velocities, t_i))

    return result


def trapezoidal_profile(all_joint_angles, total_duration, accel_frac=0.2, decel_frac=0.2):
    """Compute trapezoidal velocity profile for a list of joint-space waypoints.

    Arc-length parameterizes the waypoints in joint space, then maps them onto
    a trapezoidal speed curve so the robot accelerates, cruises, and decelerates
    smoothly.  First and last waypoints get zero velocity; intermediate waypoints
    get velocities aligned with the local path tangent.

    Args:
        all_joint_angles: List/array of joint angle arrays (N waypoints, each
            with 6 joint values).  The first entry should be the current joint
            state, and the last the final target.
        total_duration: Total trajectory time in seconds.
        accel_frac: Fraction of total_duration spent accelerating (default 0.2).
        decel_frac: Fraction of total_duration spent decelerating (default 0.2).

    Returns:
        List of (positions, velocities, time_from_start) tuples, one per
        waypoint.  ``positions`` and ``velocities`` are lists of 6 floats;
        ``time_from_start`` is a float in seconds.
    """
    cumulative_s = _compute_arc_lengths(all_joint_angles)
    total_s = cumulative_s[-1]

    t_accel = accel_frac * total_duration
    t_decel = decel_frac * total_duration
    t_cruise = total_duration - t_accel - t_decel
    v_max = total_s / (0.5 * t_accel + t_cruise + 0.5 * t_decel)
    a_accel = v_max / t_accel
    a_decel = v_max / t_decel

    def s_and_v(t):
        if t <= t_accel:
            s = 0.5 * a_accel * t ** 2
            v = a_accel * t
        elif t <= t_accel + t_cruise:
            s_accel = 0.5 * v_max * t_accel
            s = s_accel + v_max * (t - t_accel)
            v = v_max
        else:
            s_accel = 0.5 * v_max * t_accel
            s_cruise = v_max * t_cruise
            t_in_decel = t - t_accel - t_cruise
            s = s_accel + s_cruise + v_max * t_in_decel - 0.5 * a_decel * t_in_decel ** 2
            v = v_max - a_decel * t_in_decel
        return s, max(v, 0.0)

    return _map_arc_lengths_to_profile(all_joint_angles, cumulative_s, total_duration, s_and_v)


def s_curve_profile(all_joint_angles, total_duration, accel_frac=0.2, decel_frac=0.2):
    """Compute S-curve velocity profile using raised-cosine velocity ramps.

    Same interface as trapezoidal_profile but uses raised-cosine velocity
    curves for acceleration and deceleration, yielding sinusoidal acceleration
    that is zero at all phase boundaries (continuous acceleration, zero jerk
    at transitions).

    During acceleration (0 <= t <= t_a):
        v(t) = v_max/2 * (1 - cos(pi * t / t_a))
        a(t) = v_max * pi / (2 * t_a) * sin(pi * t / t_a)
        s(t) = v_max/2 * (t - (t_a/pi) * sin(pi * t / t_a))

    a(0) = 0, a(t_a) = 0 — smooth at both boundaries.

    Args:
        all_joint_angles: List/array of joint angle arrays (N waypoints).
        total_duration: Total trajectory time in seconds.
        accel_frac: Fraction of total_duration for acceleration (default 0.2).
        decel_frac: Fraction of total_duration for deceleration (default 0.2).

    Returns:
        List of (positions, velocities, time_from_start) tuples.
    """
    cumulative_s = _compute_arc_lengths(all_joint_angles)
    total_s = cumulative_s[-1]

    t_a = accel_frac * total_duration
    t_d = decel_frac * total_duration
    t_c = total_duration - t_a - t_d
    # v_max from total displacement: s_accel + s_cruise + s_decel = total_s
    # s_accel = v_max*t_a/2, s_cruise = v_max*t_c, s_decel = v_max*t_d/2
    v_max = total_s / (0.5 * t_a + t_c + 0.5 * t_d)

    def s_and_v(t):
        if t <= t_a:
            # Raised-cosine velocity: v = v_max/2 * (1 - cos(pi*t/t_a))
            v = (v_max / 2.0) * (1.0 - np.cos(np.pi * t / t_a))
            s = (v_max / 2.0) * (t - (t_a / np.pi) * np.sin(np.pi * t / t_a))
        elif t <= t_a + t_c:
            s_accel = v_max * t_a / 2.0
            v = v_max
            s = s_accel + v_max * (t - t_a)
        else:
            s_accel = v_max * t_a / 2.0
            s_cruise = v_max * t_c
            td = t - t_a - t_c
            # Mirror: v = v_max/2 * (1 + cos(pi*td/t_d))
            v = (v_max / 2.0) * (1.0 + np.cos(np.pi * td / t_d))
            s = s_accel + s_cruise + (v_max / 2.0) * (td + (t_d / np.pi) * np.sin(np.pi * td / t_d))
        return s, max(v, 0.0)

    return _map_arc_lengths_to_profile(all_joint_angles, cumulative_s, total_duration, s_and_v)


def min_jerk_profile(all_joint_angles, total_duration):
    """Compute min-jerk (5th-order polynomial) velocity profile.

    Minimizes the integral of jerk squared, producing the smoothest possible
    motion with zero velocity, acceleration, and jerk at both endpoints.

    For normalized time tau = t / T:
        s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
        v(tau) = (total_s / T) * (30*tau^2 - 60*tau^3 + 30*tau^4)

    No accel/decel fractions — the profile is a single symmetric bell curve.
    Peak speed is 1.875 * average speed (at tau = 0.5).

    Args:
        all_joint_angles: List/array of joint angle arrays (N waypoints).
        total_duration: Total trajectory time in seconds.

    Returns:
        List of (positions, velocities, time_from_start) tuples.
    """
    cumulative_s = _compute_arc_lengths(all_joint_angles)
    total_s = cumulative_s[-1]

    def s_and_v(t):
        tau = t / total_duration
        s = total_s * (10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5)
        v = (total_s / total_duration) * (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4)
        return s, max(v, 0.0)

    return _map_arc_lengths_to_profile(all_joint_angles, cumulative_s, total_duration, s_and_v)
