"""Velocity profile utilities for joint-space trajectory generation.

Provides:
- Single-point profile for simple joint-to-joint moves
- Trapezoidal velocity profile (accel / cruise / decel) for multi-waypoint trajectories
"""

import numpy as np


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
    n_total = len(all_joint_angles)
    num_joints = len(all_joint_angles[0])

    # Compute arc-length along the joint-space path
    segment_dists = []
    for i in range(1, n_total):
        dist = np.linalg.norm(
            np.asarray(all_joint_angles[i]) - np.asarray(all_joint_angles[i - 1])
        )
        segment_dists.append(max(dist, 1e-6))

    cumulative_s = [0.0]
    for d in segment_dists:
        cumulative_s.append(cumulative_s[-1] + d)
    total_s = cumulative_s[-1]

    # Trapezoidal timing parameters
    t_accel = accel_frac * total_duration
    t_decel = decel_frac * total_duration
    t_cruise = total_duration - t_accel - t_decel
    v_max = total_s / (0.5 * t_accel + t_cruise + 0.5 * t_decel)
    a_accel = v_max / t_accel
    a_decel = v_max / t_decel

    def trapez_s_and_v(t_query):
        if t_query <= t_accel:
            s = 0.5 * a_accel * t_query ** 2
            v = a_accel * t_query
        elif t_query <= t_accel + t_cruise:
            s_accel = 0.5 * v_max * t_accel
            s = s_accel + v_max * (t_query - t_accel)
            v = v_max
        else:
            s_accel = 0.5 * v_max * t_accel
            s_cruise = v_max * t_cruise
            t_in_decel = t_query - t_accel - t_cruise
            s = s_accel + s_cruise + v_max * t_in_decel - 0.5 * a_decel * t_in_decel ** 2
            v = v_max - a_decel * t_in_decel
        return s, max(v, 0.0)

    def find_time_for_s(target_s):
        lo, hi = 0.0, total_duration
        for _ in range(50):
            mid = (lo + hi) / 2
            s_mid, _ = trapez_s_and_v(mid)
            if s_mid < target_s:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    # Map each waypoint's arc-length position to a time
    waypoint_times = [find_time_for_s(s) for s in cumulative_s]
    waypoint_times[0] = 0.0
    waypoint_times[-1] = total_duration

    # Build output tuples
    result = []
    for i in range(n_total):
        t_i = waypoint_times[i]
        _, speed_scalar = trapez_s_and_v(t_i)

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
