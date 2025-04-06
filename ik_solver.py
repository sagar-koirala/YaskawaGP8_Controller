import numpy as np
from scipy.optimize import minimize
import sympy as sp

def inverse_kinematics(X, Y, Z, Roll, Pitch, Yaw, desired_wrist="any", current_joints=None):
    # Define joint limits in degrees (Yaskawa GP8)
    joint_limits = [
        (-170, 170),  # q1
        (-65, 145),   # q2
        (-70, 190),   # q3
        (-190, 190),  # q4
        (-135, 135),  # q5
        (-360, 360)   # q6
    ]
    
    # Define DH transformation matrix function for numeric calculations
    def DH_matrix(theta, d, a, alpha):
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        return np.array([
            [ct, -st, 0, a],
            [st*ca, ct*ca, -sa, -d*sa],
            [st*sa, ct*sa, ca, d*ca],
            [0, 0, 0, 1]
        ])
    
    # Define DH transformation matrix function for symbolic calculations
    def DH_matrix_sym(theta, d, a, alpha):
        ct = sp.cos(theta)
        st = sp.sin(theta)
        ca = sp.cos(alpha)
        sa = sp.sin(alpha)
        return sp.Matrix([
            [ct, -st, 0, a],
            [st*ca, ct*ca, -sa, -d*sa],
            [st*sa, ct*sa, ca, d*ca],
            [0, 0, 0, 1]
        ])
    
    # Define the DH parameters as constants
    d1 = 0.33002  # base height
    a1 = 0.01867  # link 0 length
    a2 = 0.04     # link 1 length
    a3 = 0.345    # link 2 length
    d4 = -0.34    # link 3 offset
    
    # Convert Euler angles to radians
    roll_rad = np.radians(Roll)
    pitch_rad = np.radians(Pitch)
    yaw_rad = np.radians(Yaw)
    
    # Construct the rotation matrix using XYZ convention
    Rx_mat = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])
    
    Ry_mat = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    
    Rz_mat = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    
    R_target = Rx_mat @ Ry_mat @ Rz_mat
    
    T_target = np.eye(4)
    T_target[0:3, 0:3] = R_target
    T_target[0:3, 3] = [X, Y, Z]
    
    T6_7 = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.241],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    T0_6 = T_target @ np.linalg.inv(T6_7)
    
    # POSITION INVERSE KINEMATICS (FIRST 3 JOINTS)
    wc_x = T0_6[0, 3]
    wc_y = T0_6[1, 3]
    wc_z = T0_6[2, 3]
    wc_pos = np.array([wc_x, wc_y, wc_z])
    
    q1_calculated = np.arctan2(wc_y, wc_x - a1)
    q1_deg = np.degrees(q1_calculated)
    
    # Normalize q1 to fit within joint limits
    while q1_deg > 180:
        q1_deg -= 360
    while q1_deg < -180:
        q1_deg += 360
    if not (joint_limits[0][0] <= q1_deg <= joint_limits[0][1]):
        return None
    
    q2, q3 = sp.symbols('q2 q3', real=True)
    
    T01_sym = DH_matrix_sym(q1_calculated, d1, a1, 0)
    T12_sym = DH_matrix_sym(q2 - sp.pi/2, 0, a2, -sp.pi/2)
    T23_sym = DH_matrix_sym(q3, 0, a3, sp.pi)
    T34_sym = DH_matrix_sym(sp.pi, d4, 0.04, -sp.pi/2)
    
    T04_sym = T01_sym * T12_sym * T23_sym * T34_sym
    p_sym = T04_sym[0:3, 3]
    
    px_func = sp.lambdify((q2, q3), p_sym[0], "numpy")
    py_func = sp.lambdify((q2, q3), p_sym[1], "numpy")
    pz_func = sp.lambdify((q2, q3), p_sym[2], "numpy")
    
    def wrist_error(q):
        return np.linalg.norm(np.array([px_func(q[0], q[1]), 
                                        py_func(q[0], q[1]), 
                                        pz_func(q[0], q[1])]) - wc_pos)
    
    q0 = [0, 0]
    bounds = [
        (np.radians(joint_limits[1][0]), np.radians(joint_limits[1][1])),  # q2
        (np.radians(joint_limits[2][0]), np.radians(joint_limits[2][1]))   # q3
    ]
    
    result = minimize(wrist_error, q0, bounds=bounds, method='SLSQP')
    
    if not result.success:
        return None
    
    q2_calculated = result.x[0]
    q3_calculated = result.x[1]
    
    q2_deg = np.degrees(q2_calculated)
    q3_deg = np.degrees(q3_calculated)
    
    if not (joint_limits[1][0] <= q2_deg <= joint_limits[1][1]):
        return None
    if not (joint_limits[2][0] <= q3_deg <= joint_limits[2][1]):
        return None
    
    # ORIENTATION INVERSE KINEMATICS (LAST 3 JOINTS)
    T01 = DH_matrix(q1_calculated, d1, a1, 0)
    T12 = DH_matrix(q2_calculated - np.pi/2, 0, a2, -np.pi/2)
    T23 = DH_matrix(q3_calculated, 0, a3, np.pi)
    T03 = T01 @ T12 @ T23
    
    T3_6 = np.linalg.inv(T03) @ T0_6
    R3_6 = T3_6[0:3, 0:3]
    
    def orientation_error(q):
        T34 = DH_matrix(q[0] + np.pi, d4, 0.04, -np.pi/2)
        T45 = DH_matrix(q[1], 0, 0, -np.pi/2)
        T56 = DH_matrix(q[2], 0, 0, np.pi/2)
        T36 = T34 @ T45 @ T56
        R36 = T36[0:3, 0:3]
        return np.linalg.norm(R36 - R3_6)
    
    def joint_distance(q, current_q=None):
        if current_q is None:
            return 0
        return np.sum(np.square(np.array(q) - np.array(current_q)))
    
    def combined_objective(q):
        ori_err = orientation_error(q)
        if current_joints is not None:
            current_q456 = np.radians(current_joints[3:6])
            return ori_err + 0.1 * joint_distance(q, current_q456)
        return ori_err
    
    base_guesses = 6
    num_initial_guesses = base_guesses
    if current_joints is not None and len(current_joints) >= 6:
        num_initial_guesses += 1
    
    all_solutions = np.zeros((num_initial_guesses, 3))
    all_errors = np.ones(num_initial_guesses) * float('inf')
    all_distances = np.ones(num_initial_guesses) * float('inf')
    valid_solutions = 0
    
    initial_guesses = [
        [0, 0, 0],
        [np.pi/2, np.pi/4, 0],
        [-np.pi/2, -np.pi/4, 0],
        [0, np.pi/2, 0],
        [0, -np.pi/2, 0],
        [np.pi/2, 0, np.pi/2]
    ]
    
    if current_joints is not None and len(current_joints) >= 6:
        initial_guesses.append(np.radians(current_joints[3:6]))
    
    bounds = [
        (np.radians(joint_limits[3][0]), np.radians(joint_limits[3][1])),  # q4
        (np.radians(joint_limits[4][0]), np.radians(joint_limits[4][1])),  # q5
        (np.radians(joint_limits[5][0]), np.radians(joint_limits[5][1]))   # q6
    ]
    
    for i in range(len(initial_guesses)):
        result = minimize(combined_objective, initial_guesses[i], 
                         bounds=bounds, method='SLSQP')
        if result.success and orientation_error(result.x) < 0.01:
            all_solutions[valid_solutions] = result.x
            all_errors[valid_solutions] = orientation_error(result.x)
            
            if current_joints is not None and len(current_joints) >= 6:
                all_distances[valid_solutions] = joint_distance(result.x, np.radians(current_joints[3:6]))
            else:
                all_distances[valid_solutions] = 0
                
            valid_solutions += 1
    
    q4_calculated = 0.0
    q5_calculated = 0.0
    q6_calculated = 0.0
    
    best_idx = -1
    best_score = float('inf')
    
    for i in range(valid_solutions):
        q5_val_current = all_solutions[i, 1]
        is_wrist_up = q5_val_current > 0
        
        wrist_match = ((is_wrist_up and desired_wrist == "up") or 
                      (not is_wrist_up and desired_wrist == "down") or 
                      (desired_wrist == "any"))
        
        if wrist_match:
            error_weight = 0.7
            distance_weight = 0.3
            
            if current_joints is not None:
                score = error_weight * all_errors[i] + distance_weight * all_distances[i]
            else:
                score = all_errors[i]
                
            if score < best_score:
                best_idx = i
                best_score = score
    
    if best_idx == -1 and valid_solutions > 0:
        if current_joints is not None:
            normalized_errors = all_errors[:valid_solutions] / np.max(all_errors[:valid_solutions]) if np.max(all_errors[:valid_solutions]) > 0 else all_errors[:valid_solutions]
            normalized_distances = all_distances[:valid_solutions] / np.max(all_distances[:valid_solutions]) if np.max(all_distances[:valid_solutions]) > 0 else all_distances[:valid_solutions]
            composite_scores = 0.7 * normalized_errors + 0.3 * normalized_distances
            best_idx = np.argmin(composite_scores)
        else:
            best_idx = np.argmin(all_errors[:valid_solutions])
    
    if best_idx != -1:
        q4_calculated = all_solutions[best_idx, 0]
        q5_calculated = all_solutions[best_idx, 1]
        q6_calculated = all_solutions[best_idx, 2]
    else:
        initial_guess = np.radians(current_joints[3:6]) if current_joints is not None else [0, 0, 0]
        result = minimize(combined_objective, initial_guess, 
                          bounds=bounds, method='SLSQP')
        if result.success:
            q4_calculated = result.x[0]
            q5_calculated = result.x[1]
            q6_calculated = result.x[2]
        else:
            return None
    
    q1_deg = np.degrees(q1_calculated)
    q2_deg = np.degrees(q2_calculated)
    q3_deg = np.degrees(q3_calculated)
    q4_deg = np.degrees(q4_calculated)
    q5_deg = np.degrees(q5_calculated)
    q6_deg = np.degrees(q6_calculated)
    
    joint_angles = [q1_deg, q2_deg, q3_deg, q4_deg, q5_deg, q6_deg]
    for i, (angle, limit) in enumerate(zip(joint_angles, joint_limits)):
        if not (limit[0] <= angle <= limit[1]):
            return None
    
    T01 = DH_matrix(np.radians(q1_deg), d1, a1, 0)
    T12 = DH_matrix(np.radians(q2_deg) - np.pi/2, 0, a2, -np.pi/2)
    T23 = DH_matrix(np.radians(q3_deg), 0, a3, np.pi)
    T34 = DH_matrix(np.radians(q4_deg) + np.pi, d4, 0.04, -np.pi/2)
    T45 = DH_matrix(np.radians(q5_deg), 0, 0, -np.pi/2)
    T56 = DH_matrix(np.radians(q6_deg), 0, 0, np.pi/2)
    T07 = T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T6_7
    
    pos_error = np.linalg.norm(T07[0:3, 3] - np.array([X, Y, Z]))
    ori_error = np.linalg.norm(T07[0:3, 0:3] - R_target)
    
    pos_error_threshold = 0.02
    ori_error_threshold = 0.12
    
    if pos_error > 0.05 or ori_error > 0.25:
        return None
    
    return joint_angles