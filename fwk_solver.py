import numpy as np

BASE_OFFSET_Y = 0.00023
base_T = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, BASE_OFFSET_Y],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

dh_params = [
    {"alpha": 0, "a": 0.01867, "d": 0.33002, "theta": 0},
    {"alpha": -90, "a": 0.04, "d": 0, "theta": -90},
    {"alpha": 180, "a": 0.345, "d": 0, "theta": 0},
    {"alpha": -90, "a": 0.04, "d": -0.34, "theta": 180},
    {"alpha": -90, "a": 0, "d": 0.0, "theta": 0},
    {"alpha": 90, "a": 0, "d": 0, "theta": 0},
    {"alpha": 0, "a": 0, "d": -0.241, "theta": 180}
]

joint_limits_deg = {
    0: {'min': -170, 'max': 170},
    1: {'min': -65, 'max': 150},
    2: {'min': -70, 'max': 190},
    3: {'min': -190, 'max': 190},
    4: {'min': -135, 'max': 135},
    5: {'min': -360, 'max': 360}
}

def dh_transform_modified(a, alpha, d, theta):
    alpha_rad = np.radians(alpha)
    theta_rad = np.radians(theta)
    
    T = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad), 0, a],
        [np.sin(theta_rad)*np.cos(alpha_rad), np.cos(theta_rad)*np.cos(alpha_rad), -np.sin(alpha_rad), -d*np.sin(alpha_rad)],
        [np.sin(theta_rad)*np.sin(alpha_rad), np.cos(theta_rad)*np.sin(alpha_rad), np.cos(alpha_rad), d*np.cos(alpha_rad)],
        [0, 0, 0, 1]
    ])
    return T

def forward_kinematics(joint_angles):
    T = base_T.copy()
    transformations = [T.copy()]
    for i, (param, angle) in enumerate(zip(dh_params, joint_angles)):
        modified_param = param.copy()
        modified_param["theta"] += angle
        T_i = dh_transform_modified(
            modified_param["a"],
            modified_param["alpha"],
            modified_param["d"],
            modified_param["theta"]
        )
        T = T @ T_i
        transformations.append(T.copy())
    return T, transformations

def rotation_matrix_to_euler_angles(R):
    if abs(R[0, 2]) > 0.99999:
        roll = np.arctan2(-R[1, 0], R[1, 1])
        pitch = np.pi/2 if R[0, 2] > 0 else -np.pi/2
        yaw = 0
    else:
        pitch = np.arcsin(R[0, 2])
        roll = np.arctan2(-R[1, 2], R[2, 2])
        yaw = np.arctan2(-R[0, 1], R[0, 0])
    return [roll, pitch, yaw]

def get_end_effector_pose(joint_angles):
    T, _ = forward_kinematics(joint_angles)
    position = T[0:3, 3].tolist()
    R = T[0:3, 0:3]
    euler = rotation_matrix_to_euler_angles(R)
    orientation = [np.degrees(angle) for angle in euler]
    return position, orientation

def validate_joint_angles(joint_angles):
    if len(joint_angles) != 6:
        return False
        
    for i, angle in enumerate(joint_angles):
        limits = joint_limits_deg[i]
        if angle < limits['min'] or angle > limits['max']:
            return False
    
    return True