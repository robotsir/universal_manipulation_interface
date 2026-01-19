import numpy as np
import math
from scipy.spatial.transform import Rotation as R

class AR4Kinematics:
    def __init__(self):
        # Ref: Craig's book page 80, PUMA 560 example
        # Modified DH Parameters (Craig Convention)
        # alpha(i-1), a(i-1), d(i), theta_offset
        self.DH_PARAMS = [
            # alpha(deg), a(mm),  d(mm),    theta_offset(deg)
            (0,           0,      169.77,   0),    # Joint 1
            (-90,         64.2,   0,        -90),    # Joint 2
            (0,           305,    0,        0),  # Joint 3 (Offset -90)
            (-90,         0,      222.63,   0),    # Joint 4
            (90,          0,      0,        0),    # Joint 5
            (-90,         0,      38+41.0,  0)     # Joint 6
        ]

        # # Joint Limits (Degrees)
        # self.LIMITS = np.array([
        #     [-170, 170], [-131, 0], [1, 147],
        #     [-168, 168], [-106, 106], [-152, 152]

        self.LIMITS = np.array([
            [-170, 170], [-42, 90], [-89.01, 52],
            [-165, 165], [-105, 105], [-155, 155]
        ])

        # Initialize Base and Tool Transforms (Identity by default)
        self.T_world_base = np.eye(4)
        self.T_flange_tool = np.eye(4)
        self.T_base_world = np.eye(4)
        self.T_tool_flange = np.eye(4)
        
        self._apply_custom_transforms()
        
    def _apply_custom_transforms(self):
        """
        Applies the specific base/tool logic for the AR4 robot.
        """
        # 1. Base Correction (Z -90)
        # Rotate the 'base' to align with the 'world'
        # Map vectors FROM base TO world
        R_world_base = R.from_euler('z', -90, degrees=True).as_matrix()
        self.T_world_base[:3, :3] = R_world_base

        # 2. Tool Transform chain: RotZ(90) -> RotX(60) -> TransZ(235+41)
        tool_rot1 = R.from_euler('z', 90, degrees=True).as_matrix()
        tool_rot2 = R.from_euler('x', 60, degrees=True).as_matrix()
        
        t1 = np.eye(4); t1[:3, :3] = tool_rot1
        t2 = np.eye(4); t2[:3, :3] = tool_rot2
        t3 = np.eye(4); t3[:3, 3] = [0, 0, 225 + 41] # Tool Length

        # Combine them into one single Tool Matrix
        # Logic: Flange @ T1 @ T2 @ T3
        self.T_flange_tool = t1 @ t2 @ t3

        # Note: Inverting matrices is computationally expensive. 
        # If static, cache the inverses in __init__. For now, we compute on fly.
        self.T_base_world = np.linalg.inv(self.T_world_base)
        self.T_tool_flange = np.linalg.inv(self.T_flange_tool)
        

    def _check_limits(self, joints):
        """Returns True if all joints are within limits."""
        if joints is None: return False
        for i, val in enumerate(joints):
            if not (self.LIMITS[i, 0] <= val <= self.LIMITS[i, 1]):
                return False
        return True

    def _modified_dh_transform(self, theta_rad, idx):
        """
        Calculates the Modified DH Transformation Matrix (Craig Eq 3.6).
        Matches the logic of the original 'transJ' function.
        """
        alpha_deg, a, d, offset_deg = self.DH_PARAMS[idx]
        
        # Apply Offsets
        theta = theta_rad + np.radians(offset_deg)
        alpha = np.radians(alpha_deg)

        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        # Modified DH Matrix
        # Row 1: [     ct,       -st,    0,      a   ]
        # Row 2: [ st*ca,     ct*ca,  -sa,  -d*sa  ]
        # Row 3: [ st*sa,     ct*sa,   ca,   d*ca  ]
        # Row 4: [      0,         0,    0,      1   ]
        
        return np.array([
            [ct,          -st,       0,     a],
            [st * ca,   ct * ca,   -sa,   -d * sa],
            [st * sa,   ct * sa,    ca,    d * ca],
            [0,             0,       0,     1]
        ])

    def forward_kinematics(self, joints):
        """
        Input: joints [J1...J6] in degrees.
        Output: (x, y, z, roll, pitch, yaw)
        """
        # 1. Handle J1 Direction (my original code uses -J1)
        j1_corrected = -joints[0]
        
        # Create list of radians, applying the J1 fix
        thetas = [np.radians(j1_corrected)] + [np.radians(j) for j in joints[1:]]

        # 2. Compute Chain
        T_base_flange = np.eye(4)
        
        for i in range(6):
            T_i = self._modified_dh_transform(thetas[i], i)
            T_base_flange = T_base_flange @ T_i

        # 2. Apply Base and Tool: Base @ Flange @ Tool
        T_world_tool = self.T_world_base @ T_base_flange @ self.T_flange_tool
        
        # 3. Extract Position
        x, y, z = T_world_tool[:3, 3]

        # 4. Extract Rotation (Matches the Fanuc/Motoman logic)
        # Using scipy 'xyz' (extrinsic) matches the RPY logic in most cases
        r = R.from_matrix(T_world_tool[:3, :3])
        rx, ry, rz = r.as_euler('xyz', degrees=True)

        return np.round([x, y, z, rx, ry, rz], 4)

    def _solve_spherical_wrist(self, R_0_3, R_0_6, wrist_flip):
        """
        Solves J4, J5, J6 given orientation matrices.
        Returns [J4, J5, J6] in degrees.
        """
        # R_3_6 = inv(R_0_3) * R_0_6
        R_3_6 = R_0_3.T @ R_0_6

        r13 = R_3_6[0, 2]
        r23 = R_3_6[1, 2]
        r33 = R_3_6[2, 2]
        r21 = R_3_6[1, 0]
        r22 = R_3_6[1, 1]

        # Calculate J5 (Pitch)
        # Sqrt term is always positive, so atan2 returns positive J5 (Non-flipped)
        # Note: We clamp the input to sqrt to avoid numerical errors slightly < 0
        val = max(0, 1 - r23**2)
        j5_rad = math.atan2(np.sqrt(val), r23)

        if wrist_flip:
            j5_rad = -j5_rad

        # Singularity check (J5 near 0)
        if abs(j5_rad) < 1e-4:
            j4_rad = 0
            # J4+J6 = atan2(...)
            j6_rad = math.atan2(-R_3_6[1, 0], R_3_6[0, 0]) 
        else:
            if wrist_flip:
                j4_rad = math.atan2(-r33, r13)
                j6_rad = math.atan2(r22, -r21)
            else:
                j4_rad = math.atan2(r33, -r13)
                j6_rad = math.atan2(-r22, r21)

        return np.degrees([j4_rad, j5_rad, j6_rad])

    def inverse_kinematics(self, target_pose, wrist_flip=False):
        """
        target_pose: The desired TCP position in WORLD coordinates (User Frame).
        """
        # 1. Strip Base and Tool to get the Flange Target
        # Flange = inv(Base) @ Target @ inv(Tool)
        
        T_base_flange = self.T_base_world @ target_pose @ self.T_tool_flange

        # 2. Now run the standard Geometric Solver on the FLANGE target
        return self._solve_ik_core(T_base_flange, wrist_flip)
    
    def _solve_ik_core(self, target_pose, wrist_flip):
        """
        Fast Analytical IK.
        Input: 4x4 Homogeneous Matrix (Target)
        Output: [J1...J6] in degrees, or None if unreachable.
        """
        # Unpack Geometry
        d1 = self.DH_PARAMS[0][2]
        a1 = self.DH_PARAMS[1][1]
        a2 = self.DH_PARAMS[2][1]
        d4 = self.DH_PARAMS[3][2]
        d6 = self.DH_PARAMS[5][2]

        # Wrist Center (WC)
        tcp = target_pose[:3, 3]
        z_axis = target_pose[:3, 2]
        wc = tcp - (d6 * z_axis)

        # --- ATTEMPT 1: Standard Solution ---
        sol1 = self._solve_geometric(wc, target_pose, a1, a2, d1, d4, wrist_flip, rear_config=False)
        if self._check_limits(sol1):
            return sol1
        else:
            print(f"sol1 joint out of limits: {sol1}")

        # --- ATTEMPT 2: "Rear" Solution (Flip Waist 180) ---
        # Only try this if solution 1 failed limits (common on AR4 for "reaching back")
        sol2 = self._solve_geometric(wc, target_pose, a1, a2, d1, d4, wrist_flip, rear_config=True)
        if self._check_limits(sol2):
            return sol2
        else:
            print(f"sol2 joint out of limits: {sol2}")

        return None # Unreachable

    def _solve_geometric(self, wc, target_pose, a1, a2, d1, d4, wrist_flip, rear_config):
        # 0. Load Offsets directly from Class Configuration
        j2_off = np.radians(self.DH_PARAMS[1][3])
        j3_off = np.radians(self.DH_PARAMS[2][3] + 90)       
        
        
        # 1. Solve J1
        j1_rad = math.atan2(wc[1], wc[0])
        
        if rear_config:
            if j1_rad > 0: j1_rad -= np.pi
            else: j1_rad += np.pi

        # 2. Solve J2/J3 (Planar)
        r = np.sqrt(wc[0]**2 + wc[1]**2)
        if rear_config: r = -r
        
        r_prime = r - a1 
        s = wc[2] - d1
        
        len_sq = r_prime**2 + s**2
        len_h = np.sqrt(len_sq)

        # Cosine rule for Elbow (J3)
        cos_c = (a2**2 + d4**2 - len_sq) / (2 * a2 * d4)
        
        if abs(cos_c) > 1.0: return None 

        angle_c = math.acos(cos_c)
 
        # FIX: Matches Original Code "O27 = 180 - O23"
        # J3 = 180 - InteriorAngle
        j3_rad = np.pi - angle_c

        # Shoulder (J2)
        angle_a = math.atan2(s, r_prime)
        cos_b = (a2**2 + len_sq - d4**2) / (2 * a2 * len_h)
        angle_b = math.acos(cos_b)
        
        # FIX: Matches Original Code "O26 = -(O21+O22)"
        # J2 = -(AngleA + AngleB)
        j2_rad = -(angle_a + angle_b)

        # 3. Solve Wrist (J4, J5, J6)
        # CRITICAL FIX: Use the class's OWN DH Transform to ensure consistency
        # We must use the angles as the DH chain expects them.
        # J1 in DH is just j1_rad (because we output -j1_rad later)
        T01 = self._modified_dh_transform(j1_rad, 0)
        T12 = self._modified_dh_transform(j2_rad, 1)
        T23 = self._modified_dh_transform(j3_rad, 2)

        R_0_3 = (T01 @ T12 @ T23)[:3, :3]
        R_0_6 = target_pose[:3, :3]

        wrist_angs = self._solve_spherical_wrist(R_0_3, R_0_6, wrist_flip)
        
        # Combine
        joints = np.array([
            np.degrees(j1_rad), 
            np.degrees(j2_rad - j2_off), 
            np.degrees(j3_rad - j3_off), 
            wrist_angs[0], 
            wrist_angs[1], 
            wrist_angs[2]
        ])
        
        # Apply AR4 J1 direction correction for final output
        joints[0] = -joints[0] 
        

        # print(f"joints: {joints}")
        
        return joints




    # --- UMI INTERFACE METHODS ---

    def compute_joint_angles(self, target_pose_6d):
        """ 
        IK: UMI Pose (Meters, RotVec Radians) -> Joint Angles (Degrees) 
        """
        # 1. Convert UMI 6D -> 4x4 Matrix
        pos_m = target_pose_6d[:3] * 1000.0 # m to mm
        rot_vec = target_pose_6d[3:]
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        target_mat = np.eye(4)
        target_mat[:3, :3] = rot_mat
        target_mat[:3, 3] = pos_m

        # 2. Call Analytical Solver
        # Note: wrist_flip=False is standard. Change if we need "Elbow Down".
        joints = self.inverse_kinematics(target_mat, wrist_flip=False)

        if joints is None:
            raise Exception("Analytical IK Failed (Out of reach or limits)")
        
        return joints

    def compute_fk(self, joints_deg):
        """ 
        FK: Joint Angles (Degrees) -> UMI Pose (Meters, RotVec Radians)
        """
        # 1. Call Analytical FK -> Returns [x,y,z, rx,ry,rz] (Euler Deg)
        pose_euler = self.forward_kinematics(joints_deg)
        
        pos_m = pose_euler[:3] * 0.001 # mm to m
        euler_deg = pose_euler[3:]

        # 2. Convert Euler Deg -> RotVec Radians
        rot_vec = R.from_euler('xyz', euler_deg, degrees=True).as_rotvec()

        # 3. Concatenate
        return np.concatenate([pos_m, rot_vec])


# --- Usage Example ---
if __name__ == "__main__":
    # --- SPEED TEST ---
    import time
    np.set_printoptions(suppress=True, precision=3)
    
    bot = AR4Kinematics()

    joints = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    print(f"Testing Joints (Deg): {np.round(joints, 2)}")

    pose = bot.forward_kinematics(joints)

    position = pose[:3]
    rpy = pose[3:]

    target = np.eye(4)
    target[:3, 3] = position
    target[:3, :3] = R.from_euler('xyz', rpy, degrees=True).as_matrix()

    print(f"calculated position: {position}, rpy: {rpy}")
    print(f"calculated matrix: \n {target}")



    start = time.perf_counter()
    sol = bot.inverse_kinematics(target, wrist_flip=False)
    end = time.perf_counter()

    if sol is not None:
        print(f"Solved IK:       {np.round(sol, 3)}")
        error = np.linalg.norm(sol - joints)
        print(f"Difference:      {error:.4f}")
    else:
        print("IK Failed to find solution")
        
    print(f"Time: {(end-start)*1000:.4f} ms") # Expect ~0.05ms