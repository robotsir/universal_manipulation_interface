import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import serial

from umi.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait

class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1

class GripperCorrector:
    def __init__(self):
        # 1. Define the raw data points you measured
        # "Command" = What you sent to the gripper
        # "Actual"  = What the caliper measured (in mm)
        self.map_command = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
        self.map_actual  = np.array([5, 10, 20, 25, 35, 45, 55, 66, 79, 90, 100, 110])

    def get_corrected_command(self, desired_opening_mm):
        """
        Input: The opening you WANT (e.g., 30mm)
        Output: The command you must SEND (e.g., 35.0)
        """
        # np.interp(x, xp, fp)
        # x  = The value we want to find (Desired Opening)
        # xp = The known x-coordinates (Actual Openings from data)
        # fp = The known y-coordinates (Commands corresponding to those actuals)
        
        command_float = np.interp(
            desired_opening_mm, 
            self.map_actual,   # Look up the value in the "Actual" list
            self.map_command   # Interpolate the corresponding "Command"
        )
        
        # 1. Round to nearest whole number (e.g. 34.6 -> 35.0)
        # 2. Cast to integer (35.0 -> 35)
        return int(round(command_float))


class ServoGripperController(mp.Process): # Inherit from mp.Process
    def __init__(self,
            shm_manager: SharedMemoryManager,
            port="/dev/serial/by-id/usb-Arduino__www.arduino.cc__0042_758303330383513060F0-if00",
            frequency=30,
            init_width=0.08,
            launch_timeout=3,
            verbose=False):
        
        super().__init__(name="ServoGripperController")
        self.port = port
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.verbose = verbose
        self.init_width = init_width

        # 1. Input Queue for commands
        example_cmd = {'cmd': Command.SCHEDULE_WAYPOINT.value, 'target_width': 0.0, 'target_time': 0.0}
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=example_cmd, buffer_size=256)

        # 2. Ring Buffer for observations (matches UMI keys)
        example_state = {
            'gripper_position': np.float64(0.0),
            'gripper_timestamp': time.time()
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager, examples=example_state, get_max_k=int(frequency*10))

        self.ready_event = mp.Event()

    def run(self):
        # Open serial INSIDE the process
        ser = serial.Serial(self.port, 115200, timeout=1.0)
        time.sleep(2.0) # Wait for reboot
        
        # Initialize interpolator to prevent sudden jumps
        curr_t = time.monotonic()
        pose_interp = PoseTrajectoryInterpolator(times=[curr_t], poses=[[self.init_width,0,0,0,0,0]])
        
        keep_running = True
        iter_idx = 0
        dt = 1.0 / self.frequency

        corrector = GripperCorrector()

        try:
            while keep_running:
                t_now = time.monotonic()
                
                # A. Get Target from Interpolator
                target_width = pose_interp(t_now)[0]
                
                # B. Send to Arduino (Meters -> MM)
                # To do: need to fine tune the gripper opening width, as it's not accurate

                target_width_mm = int(round(np.clip(target_width * 1000.0, 0, 110)))

                gripper_cmd = corrector.get_corrected_command(target_width_mm)

                ser.write(f"SV0P{gripper_cmd}\n".encode("ascii"))

                # C. Feedback to Ring Buffer
                self.ring_buffer.put({
                    'gripper_position': target_width, 
                    'gripper_timestamp': time.time()
                })

                # D. Handle Command Queue
                try:
                    commands = self.input_queue.get_all()
                    for i in range(len(commands['cmd'])):
                        if commands['cmd'][i] == Command.SHUTDOWN.value:
                            keep_running = False
                        elif commands['cmd'][i] == Command.SCHEDULE_WAYPOINT.value:
                            # Convert wall time to monotonic for the interpolator
                            target_t_mono = time.monotonic() - time.time() + commands['target_time'][i]
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=[commands['target_width'][i],0,0,0,0,0],
                                time=target_t_mono,
                                max_pos_speed=0.2, curr_time=t_now
                            )
                except Empty:
                    pass

                if iter_idx == 0: self.ready_event.set()
                iter_idx += 1
                precise_wait(t_now + dt)
        finally:
            ser.close()
            self.ready_event.set()

    # --- UMI Interface Methods ---
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[ServoGripper] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        # Wait for the background process to signal it is ready
        self.ready_event.wait(self.launch_timeout)
        # Verify the process didn't crash immediately
        assert self.is_alive(), "ServoGripperController process failed to start!"
    
    def stop_wait(self): 
        # Ensure the process has fully joined before continuing
        self.join()


    @property
    def is_ready(self): return self.ready_event.is_set()
    def get_state(self): return self.ring_buffer.get()
    def get_all_state(self): return self.ring_buffer.get_all()
    
    def schedule_waypoint(self, *args, **kwargs):
        """
        Flexible waypoint scheduler that handles different UMI naming conventions.
        """
        width = None
        target_time = None

        # 1. Handle positional arguments (width, target_time)
        if len(args) >= 1:
            width = args[0]
        if len(args) >= 2:
            target_time = args[1]

        # 2. Handle keyword arguments (pos, width, width_m, target_time)
        for key in ('pos', 'width', 'width_m', 'gripper_width'):
            if width is None and key in kwargs:
                width = kwargs[key]
        
        if target_time is None and 'target_time' in kwargs:
            target_time = kwargs['target_time']

        # 3. Validation and Queueing
        if width is not None and target_time is not None:
            # Flatten if width is passed as a numpy array
            width = float(np.asarray(width).reshape(-1)[0])
            
            self.input_queue.put({
                'cmd': Command.SCHEDULE_WAYPOINT.value, 
                'target_width': width, 
                'target_time': target_time
            })