import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import scipy.spatial.transform as st
import serial
import re

# --- UMI / Diffusion Policy Imports ---
from umi.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait

# --- KINEMATICS IMPORT ---
# This expects 'ar4_kinematics.py' to be in the same folder
try:
    from umi.real_world.ar4_kinematics import AR4Kinematics
except ImportError:
    print("[ERROR] Could not import 'AR4Kinematics'. Make sure 'ar4_kinematics.py' is in the same folder.")
    raise

# --- UTILS FOR SERIAL PARSING ---
_RP_RE = re.compile(r"([A-Z])(-?\d+(?:\.\d+)?)")

def parse_rp_line(line: str) -> dict:
    """ Parses the AR4 'RP' response string into a dictionary. """
    d = {}
    for k, v in _RP_RE.findall(line):
        d[k] = float(v)
    return d

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

class AR4InterpolationController(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager, 
            port="/dev/serial/by-id/usb-Teensyduino_USB_Serial_11564850-if00", 
            frequency=30, 
            lookahead_time=0.05,   # Look 50ms into future to compensate for robot lag
            max_pos_speed=0.25, # 5% of max speed, copied from ur5
            max_rot_speed=0.16, # 5% of max speed, copied from ur5
            launch_timeout=3,
            verbose=False,
            get_max_k=None,
            receive_latency=0.015
            ):
        
        super().__init__(name="AR4PositionalController")
        self.port = port
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.verbose = verbose
        self.lookahead_time = lookahead_time
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.receive_latency = receive_latency

        if get_max_k is None:
            get_max_k = int(frequency * 10)

        # 1. Input Queue (Matches UR5 structure)
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=example, buffer_size=1024
        )

        # 2. Ring Buffer (Matches UMI observation keys)
        example_state = {
            'ActualTCPPose': np.zeros((6,), dtype=np.float64),
            'TargetTCPPose': np.zeros((6,), dtype=np.float64),
            'ActualQ': np.zeros((6,), dtype=np.float64),
            'ActualQd': np.zeros((6,), dtype=np.float64),
            'robot_receive_timestamp': time.time(),
            'robot_timestamp': time.time()
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example_state,
            get_max_k=get_max_k,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()

    def run(self):
        # Initialize Serial INSIDE the process
        try:
            np.set_printoptions(suppress=True, precision=6)

            ser = serial.Serial(self.port, 115200, timeout=1.0)
            time.sleep(0.5) # Wait for Teensy reset
            # clear error state just in case
            ser.write(b"ER\n")
            line = ser.readline().decode("ascii", errors="ignore")

            print(f"[AR4Controller] ER Response: {line.strip()}")


            # Move to initial position (optional)
            # ser.write(b"MJX0Y0Z0Rz0Ry0Rx0Sp10\n")
            # line = ser.readline().decode("ascii", errors="ignore")
            cmd = "RJA0.000B-20.016C30.024D0.000E90.000F0.000J70J80J90Sp25Ac15Dc20Rm100WNLm000000\n"

            ser.reset_input_buffer()
            ser.write(cmd.encode())    
            deadline = time.time() + 30.0
            required_joints = ["B", "C", "D", "E", "F"]
            
            while time.time() < deadline:
                response = ser.readline().decode("utf-8").strip()
                if not response:
                    continue
                if response.startswith('E'):
                    break
                if response.startswith("A") and all(char in response for char in required_joints):
                    break

        except Exception as e:
            print(f"[AR4Controller] Serial Error: {e}")
            self.ready_event.set() # Release wait to fail gracefully
            return

        # 2. Initialize Kinematics Solver INSIDE the process
        # (ikpy is not picklable, so it must be created here, not in __init__)
        kinematics = AR4Kinematics()

        try:
            # --- HELPER: Serial I/O Wrapper ---
            def get_current_pose_and_joints():
                # [NEW] Sweep the floor: Clear the buffer of old responses (like "OK")
                # from the PREVIOUS loop iteration before asking for new data.
                if ser.in_waiting:
                    ser.read(ser.in_waiting)

                ser.write(b"RP\n")
                # This readline is now guaranteed* to get the RP response,
                # not the leftover response from the last move command.
                line = ser.readline().decode("ascii", errors="ignore")

                #print(f"[AR4Controller] <<<< RP Response: {line.strip()}")

                # CHECK FOR ESTOP MESSAGE FROM FIRMWARE
                if "ESTOP ACTIVE" in line or "ESTOP LATCHED" in line:
                    print("[AR4Controller] WARNING: Robot is in Soft E-Stop! Release button and restart.")
                    return None, None

                match = _RP_RE.search(line)
                if match:
                    # We need to extract A-F for joints and G-L for pose
                    # Let's assume your regex/parser gets all letters:
                    vals = parse_rp_line(line) # Use your dict parser
                    


                    # pos_m = np.array([vals['G'], vals['H'], vals['I']]) / 1000.0
                    # rot = st.Rotation.from_euler('zyx', [vals['J'], vals['K'], vals['L']], degrees=True)
                    # pose = np.concatenate([pos_m, rot.as_rotvec()])

                    # 2. Joint Positions (A-F)
                    # Note: UMI usually expects Radians for joints. 
                    # If AR4 gives Degrees, convert them here.
                    joints_deg = np.array([vals['A'], vals['B'], vals['C'], vals['D'], vals['E'], vals['F']])
                    joints_rad = np.deg2rad(joints_deg)

                                        
                    # 1. TCP Pose (m and rad)
                    # 2. Compute FK to get Cartesian Pose (In UMI Frame)
                    # This returns the 6D pose vector [x, y, z, rx, ry, rz]
                    pose_umi = kinematics.compute_fk(joints_deg)


                    return pose_umi, joints_rad
                return None, None

            # Init loop variables
            dt = 1. / self.frequency
            print(f"[AR4Controller] Starting control loop at {self.frequency} Hz on port {self.port} dt={dt:.4f}s")

            # 1. Properly handle the initial pose fetch
            curr_pose, actual_q = get_current_pose_and_joints()

            if curr_pose is None:
                print("[AR4Controller] Warning: Could not fetch initial pose, using zeros.")
                curr_pose = np.zeros(6, dtype=np.float64)
            else:
                print(f"[AR4Controller] Initial pose fetched: {curr_pose}")


            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t # <--- ADD THIS

            # 2. Setup interpolator with the validated pose
            pose_interp = PoseTrajectoryInterpolator(times=[curr_t], poses=[curr_pose])
     
            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True

            last_q = None
            #last_t = None

            #time.sleep(1.0) # Wait a moment before sending RP command again
            last_mj_send_time = 0
            last_cmd_angles = np.zeros(6) # Initialize storage for last sent angles
            MIN_MOVE_THRESHOLD = 0.05     # Degrees (Deadband) - Ignore moves smaller than this

            while keep_running:
                t_loop_start = time.monotonic()
                
                # A. Get interpolated command
                # 1. This is the pose the interpolator calculated (where the robot SHOULD be)
                target_pose = pose_interp(t_loop_start + self.lookahead_time) # <--- ADD LOOKAHEAD
                

                # --- STEP D: Send MJ Command (Throttled to 1Hz) ---
                # Only send if 1.0 second has passed since the last command



                # --- B. Calculate IK (Python Side) ---
                # Calculate angles for the 6 motors
                try:
                    motor_angles_deg = kinematics.compute_joint_angles(target_pose)
                except Exception as e:
                    print(f"[AR4Controller] IK Error: {e}")
                    # Do NOT send [0,0,0,0,0,0]. Just stay where we are.
                    motor_angles_deg = last_cmd_angles

                # 3. SAFETY CHECKS (Deadband + Rate Limit)
                # Check A: Is the move too small? (Fixes "Slow Crawl")
                change_magnitude = np.max(np.abs(motor_angles_deg - last_cmd_angles))
                is_tiny_move = change_magnitude < MIN_MOVE_THRESHOLD
                
                # Check B: Are we sending too fast? (Fixes "Buffer Saturation" during catch-up spikes)
                # We use 0.99 * dt to allow slight timing jitter but block 200Hz bursts
                is_too_fast = (t_loop_start - last_mj_send_time) < (dt * 0.99)


                # 2. If the move is microscopic (noise/catch-up), DO NOT SEND IT.
                #    0.05 degrees is a safe threshold for AR4 accuracy.
                if is_tiny_move or is_too_fast:
                    # SKIP this command. 
                    # This effectively drops the "catch-up" frames that flood the buffer.
                    pass
                else:
                    # Update our memory
                    last_cmd_angles = motor_angles_deg
                    # Send command as normal

                    # --- C. Send Joint Command (RJA) ---
                    # We construct the RJ string directly with the calculated angles
                    # A=J1, B=J2 ... F=J6
                    cmd = "RJ"
                    cmd += f"A{motor_angles_deg[0]:.2f}"
                    cmd += f"B{motor_angles_deg[1]:.2f}"
                    cmd += f"C{motor_angles_deg[2]:.2f}"
                    cmd += f"D{motor_angles_deg[3]:.2f}"
                    cmd += f"E{motor_angles_deg[4]:.2f}"
                    cmd += f"F{motor_angles_deg[5]:.2f}"

                    # Add motion params (Speed/Accel)
                    # Note: For trajectory tracking, speed should be handled by the update rate (frequency),
                    # but we set a high speed limit here so the motor controller doesn't throttle us.
                    cmd += "J70J80J90Sp25Ac15Dc20Rm100WNLm000000\n"

                    #cmd = f"MJX{x_mm:.3f}Y{y_mm:.3f}Z{z_mm:.3f}Rz{rz:.3f}Ry{ry:.3f}Rx{rx:.3f}Sp10\n"

                    if self.verbose and iter_idx % 30 == 0:
                        print(f"[AR4Controller] >>>>>> Sending Command: {cmd.strip()}")
                    # don't move robot for now
                    ser.write(cmd.encode("ascii"))

                    # don't wait for response, it slows down the loop
                    # line = ser.readline().decode("ascii", errors="ignore")

                    # print(f"[AR4Controller] <<<< MJ Response: {line.strip()}")

                    # # Read minimal response to keep buffer clean (Optional: Non-blocking read recommended)
                    # if ser.in_waiting:
                    #     ser.read(ser.in_waiting)


                    last_mj_send_time = t_loop_start


                # C. Feedback to Ring Buffer
                # 2. This is the pose you got from the RP command (where the robot ACTUALLY is)
                actual_pose, actual_q = get_current_pose_and_joints()


                if self.verbose and iter_idx % 30 == 0 and actual_q is not None:
                    print(f"[AR4Controller] Current UMI Pose: {actual_pose}, Joints (degree): {np.rad2deg(actual_q)}")

                t_wall_clock = time.time()

                actual_qd = np.zeros(6, dtype=np.float64)
                if last_q is not None and actual_q is not None:
                    # Use monotonic for dt calculation to avoid system clock jump issues
                    dt_actual = t_loop_start - last_monotonic_t
                    if dt_actual > 0:
                        actual_qd = (actual_q - last_q) / dt_actual

                last_q = actual_q
                last_monotonic_t = t_loop_start

                if actual_pose is not None:
                    self.ring_buffer.put({
                        'ActualTCPPose': actual_pose,
                        'TargetTCPPose': target_pose,
                        'ActualQ': actual_q,
                        'ActualQd': actual_qd,
                        'robot_receive_timestamp': t_wall_clock, # Use wall clock here
                        'robot_timestamp': t_wall_clock - self.receive_latency # <--- SUBTRACT LATENCY
                    })

                # D. Handle Input Queue
                # Note: Ensure you use monotonic for the interpolator logic here too!
                try:
                    # process at most 1 command per cycle to maintain frequency
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                    commands = None # Safety

                # Only try to loop if we actually have commands
                if n_cmd > 0:
                    # execute commands
                    for i in range(n_cmd):
                        cmd_type = commands['cmd'][i]
                        if cmd_type == Command.STOP.value:
                            keep_running = False
                            # stop immediately, ignore later commands
                            break
                        elif cmd_type == Command.SCHEDULE_WAYPOINT.value:
                            target_time_mono = time.monotonic() - time.time() + commands['target_time'][i]
                            
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=commands['target_pose'][i],
                                time=target_time_mono,
                                max_pos_speed=self.max_pos_speed, 
                                max_rot_speed=self.max_rot_speed, 
                                curr_time=t_loop_start + dt,
                                last_waypoint_time=last_waypoint_time # <--- PASS THIS
                            )
                            last_waypoint_time = target_time_mono # <--- UPDATE THIS 
                            
                        else:
                            keep_running = False
                            break


                # E. Precise timing - use the MONOTONIC start time


                # t_loop_end = time.monotonic()
            
                # NOTE: duration is aournd 0.01s for AR4
                # if self.verbose:
                    # print(f"[AR4Controller] Loop {iter_idx}: Start {t_loop_start:.4f}, End {t_loop_end:.4f}, Duration {t_loop_end - t_loop_start:.4f}s")
                
                # regulate frequency
                # t_wait_until = t_loop_start + dt
                # if t_wait_until > t_loop_end:
                #     precise_wait(t_wait_until, time_func=time.monotonic)

                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                if iter_idx == 0: 
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose and iter_idx % 30 == 0:
                    print(f"[AR4Controller] Actual frequency {1/(time.monotonic() - t_loop_start):.2f} Hz")





        finally:# <--- This runs even if the loop crashes or stops
            # 2. Mandatory Cleanup
            ser.close() 
            self.ready_event.set() # Release anything waiting on this process
            if self.verbose:
                print(f"[AR4Controller] Serial port {self.port} closed and process exiting.")

    # Required interface methods for UMI
    @property
    def is_ready(self):
        # The controller is ready if:
        # 1. The process is still alive
        # 2. The 'ready_event' has been set by the run() loop
        # 3. The ring buffer has at least one data point
        # print(f"[AR4Controller] is_alive: {self.is_alive()}, ready_event: {self.ready_event.is_set()}, ring_buffer count: {self.ring_buffer.count}")
        return self.is_alive() and self.ready_event.is_set() and (self.ring_buffer.count > 0)

    def start(self, wait=True):
        # This triggers the mp.Process magic
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[AR4Controller] Process spawned at {self.pid}")

    def stop(self, wait=True):
        # Sends the STOP signal to your while loop
        self.input_queue.put({'cmd': Command.STOP.value})
        if wait:
            self.stop_wait()

    def start_wait(self):
        # self.launch_timeout should be set in your __init__ (usually 3 seconds)
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive(), "AR4 Controller process failed to start!"
    
    def stop_wait(self):
        self.join()

    def get_state(self): return self.ring_buffer.get()
    def get_all_state(self): return self.ring_buffer.get_all()
    def schedule_waypoint(self, pose, target_time):
        self.input_queue.put({'cmd': Command.SCHEDULE_WAYPOINT.value, 'target_pose': pose, 'target_time': target_time})