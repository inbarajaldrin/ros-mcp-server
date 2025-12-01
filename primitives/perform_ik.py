#!/usr/bin/env python3
"""
Inverse Kinematics Helper Functions
Extracted from JETANK GUI for use in primitives.

Provides forward and inverse kinematics functions for the JETANK robot arm.
"""

import math
import numpy as np

# Try to import scipy - required for IK functions
try:
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    minimize = None
    R = None
    print("Warning: scipy not available. IK functions will be disabled.")


def forward_kinematics(theta0, theta1, theta3):
    """
    Calculate forward kinematics for JETANK robot
    Returns transformation matrix and position
    
    Args:
        theta0: Base bearing angle (radians)
        theta1: Lower servo angle (radians)
        theta3: Upper servo angle (radians)
    
    Returns:
        T: 4x4 transformation matrix, or None if invalid
        position: (x, y, z) tuple in mm, or None if invalid
    """
    # Joint limits (from URDF)
    limits = [
        {'lower': -1.5708, 'upper': 1.5708},      # Joint 0
        {'lower': 0.0,     'upper': 1.570796},    # Joint 1
        {'lower': -3.1418, 'upper': 0.785594}     # Joint 3
    ]
    
    # Check joint limits
    if theta0 < limits[0]['lower'] or theta0 > limits[0]['upper']:
        return None, None
    if theta1 < limits[1]['lower'] or theta1 > limits[1]['upper']:
        return None, None
    if theta3 < limits[2]['lower'] or theta3 > limits[2]['upper']:
        return None, None
    
    # DH parameters (in meters)
    dh_params = [
        {'alpha': 1.57, 'a': -13.50/1000, 'd': 75.00/1000},  # Link 0
        {'alpha': -1.57,  'a': 0.00/1000,   'd': 0.00/1000},   # Link 1  
        {'alpha': 1.57, 'a': 0.00/1000,   'd': 95.00/1000},  # Link 2
        {'alpha': -1.57,  'a': -179.25/1000, 'd': 0.00/1000}   # Link 3
    ]
    
    # Joint angles (theta2 is fixed at 0)
    theta = [theta0, theta1, 0.0, theta3]
    
    def dh_matrix(alpha, a, d, theta):
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        cos_alpha = math.cos(alpha)
        sin_alpha = math.sin(alpha)
        
        return np.array([
            [cos_theta, -sin_theta*cos_alpha,  sin_theta*sin_alpha, a*cos_theta],
            [sin_theta,  cos_theta*cos_alpha, -cos_theta*sin_alpha, a*sin_theta],
            [0,          sin_alpha,            cos_alpha,           d],
            [0,          0,                    0,                   1]
        ])
    
    # Calculate transformation matrix
    T = np.eye(4)
    for i in range(4):
        dh = dh_params[i]
        T_i = dh_matrix(dh['alpha'], dh['a'], dh['d'], theta[i])
        T = np.dot(T, T_i)
    
    # Extract position and convert to mm
    position = T[:3, 3] * 1000
    
    return T, (position[0], position[1], position[2])


def rpy_to_matrix(rpy_degrees):
    """Convert roll-pitch-yaw in degrees to rotation matrix"""
    if not SCIPY_AVAILABLE:
        return None
    return R.from_euler('xyz', rpy_degrees, degrees=True).as_matrix()


def ik_objective(q, target_position, target_orientation=None, position_weight=1.0, orientation_weight=0.1):
    """
    Objective function for IK optimization
    q: joint angles [theta0, theta1, theta3]
    target_position: [x, y, z] in mm
    target_orientation: optional 3x3 rotation matrix
    """
    # Get forward kinematics
    T, current_pos = forward_kinematics(q[0], q[1], q[2])
    
    if T is None or current_pos is None:
        return 1e6  # Large penalty for invalid joint angles
    
    # Position error
    pos_error = np.linalg.norm(np.array(current_pos) - np.array(target_position))
    
    total_error = position_weight * pos_error
    
    # Orientation error (if specified)
    if target_orientation is not None:
        current_orientation = T[:3, :3]
        rot_error = np.linalg.norm(current_orientation - target_orientation, 'fro')
        total_error += orientation_weight * rot_error
    
    return total_error


def compute_ik(target_x, target_y, target_z, target_rpy=None, q_guess=None, max_tries=10, position_tolerance=1.0):
    """
    Compute inverse kinematics for JETANK robot
    
    Args:
        target_x, target_y, target_z: target position in mm
        target_rpy: optional target orientation [rx, ry, rz] in degrees
        q_guess: initial guess for joint angles
        max_tries: maximum number of optimization attempts
        position_tolerance: acceptable position error in mm
    
    Returns:
        joint_angles: [theta0, theta1, theta3] in radians, or None if failed
    """
    if not SCIPY_AVAILABLE:
        return None
    
    target_position = [target_x, target_y, target_z]
    target_orientation = None
    
    if target_rpy is not None:
        target_orientation = rpy_to_matrix(target_rpy)
    
    # Joint bounds
    joint_bounds = [
        (-1.5708, 1.5708),   # theta0
        (0.0, 1.570796),     # theta1  
        (-3.1418, 0.785594)  # theta3
    ]
    
    # Default initial guess if none provided
    if q_guess is None:
        q_guess = [0.0, 0.785, -1.57]  # Middle-ish values
    
    best_result = None
    best_error = float('inf')
    
    for attempt in range(max_tries):
        # Add small random perturbation to initial guess for each attempt
        if attempt > 0:
            current_guess = q_guess + np.random.normal(0, 0.1, 3)
            # Ensure guess is within bounds
            for i in range(3):
                current_guess[i] = np.clip(current_guess[i], joint_bounds[i][0], joint_bounds[i][1])
        else:
            current_guess = q_guess
        
        try:
            # Use different optimization methods
            methods = ['L-BFGS-B', 'SLSQP', 'trust-constr']
            method = methods[attempt % len(methods)]
            
            result = minimize(
                ik_objective, 
                current_guess, 
                args=(target_position, target_orientation),
                method=method,
                bounds=joint_bounds,
                options={'maxiter': 1000}
            )
            
            if result.success:
                error = ik_objective(result.x, target_position, target_orientation)
                if error < best_error:
                    best_error = error
                    best_result = result
                
                # Check if solution is good enough
                if error < position_tolerance:
                    return result.x
        
        except Exception as e:
            continue
    
    # Return best result found, even if not optimal
    if best_result is not None and best_error < 10.0:  # Accept if within 10mm
        return best_result.x
    
    return None


def verify_solution(joint_angles, target_position, target_rpy=None):
    """Verify the IK solution by computing forward kinematics"""
    if not SCIPY_AVAILABLE:
        return False
    
    T, actual_pos = forward_kinematics(joint_angles[0], joint_angles[1], joint_angles[2])
    
    if actual_pos is None:
        return False
    
    pos_error = np.linalg.norm(np.array(actual_pos) - np.array(target_position))
    
    return pos_error < 5.0  # Accept if within 5mm

