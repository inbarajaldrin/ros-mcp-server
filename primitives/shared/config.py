import numpy as np

TABLE_HEIGHT = -0.08       # Table surface Z in robot base frame (meters)
TABLE_COLLISION_MARGIN_SIDEWAYS = 0.08  # Safety margin for sideways/horizontal EE orientations
TABLE_COLLISION_MARGIN_FACEDOWN = 0.08  # Safety margin for face_down EE orientation 
GRIPPER_CENTER_TOOL_OFFSET = np.array([0.0, 0.0, 0.2286])  # 3D offset from EE flange to gripper center in tool frame (meters)
SAFE_HEIGHT_ABOVE_TABLE = 0.4      # Safe/hover height above table surface (meters)
SAFE_HEIGHT = TABLE_HEIGHT + SAFE_HEIGHT_ABOVE_TABLE  # Absolute safe height Z in robot base frame
ROTATE_ABOUT_GRIPPER_CENTER = False # If True, rotate about gripper center; if False, rotate about flange
DEFAULT_BASE_ORIENTATION = [0.0, 0.0, 0.0, 1.0]  # Base object orientation [x, y, z, w] quaternion (identity = no rotation)
DEFAULT_BASE_POSITION = [0.0, -0.4, -0.0625]     # Default base object position [x, y, z] in robot base frame (meters)

# UR5e motion limits — two independent constraints, compute_duration takes the max.
# Cartesian limits govern translation speed (reference: safe_height ↔ table = 0.32m → 5.0s).
# Joint limits govern rotation speed independently (2 rad rotation → 5.0s).
JOINT_VEL_LIMIT = 0.80      # rad/s  (hardware max: pi rad/s)
JOINT_ACCEL_LIMIT = 0.80    # rad/s²
CART_VEL_LIMIT = 0.0798     # m/s    (hardware max: 1.0 m/s)
CART_ACCEL_LIMIT = 0.0798   # m/s²
MIN_DURATION = 2.5          # seconds — floor to avoid jerky micro-moves
MAX_DURATION = 10.0         # seconds — cap for very long moves

HOME_POSE = [0.065, -0.385, 0.481, 0, 180, 0]  # EE tool frame
PICK_STATION_POSE = [-0.330, -0.385, 0.404, 0, 180, 0]  # EE tool frame 

# UR5e DH parameters: (theta_offset, d, a, alpha)
DH_PARAMS = [
    (0,  0.1625,  0,     np.pi/2),
    (0,  0,      -0.425,  0),
    (0,  0,      -0.3922, 0),
    (0,  0.1333,  0,     np.pi/2),
    (0,  0.0997,  0,    -np.pi/2),
    (0,  0.0996,  0,     0)
]
