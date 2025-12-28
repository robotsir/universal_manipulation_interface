import time
import numpy as np


class DummyInterpolationController:
    def __init__(self, *args, **kwargs):
        self._t0 = time.time()
        # UR-style TCP pose: [x, y, z, rx, ry, rz]
        self._tcp_pose = np.zeros(6, dtype=np.float64)
        self._q = np.zeros(6, dtype=np.float64)
        self._gripper_width = np.float64(0.08)
        self._qd = np.zeros(6, dtype=np.float64)

    @property
    def is_ready(self): return True
    def start(self, wait=False): pass
    def stop(self, wait=False): pass
    def start_wait(self): pass
    def stop_wait(self): pass
    def schedule_waypoint(self, *args, **kwargs): pass

    def get_state(self):
        # minimal
        return self.get_all_state()


    def get_all_state(self):
        t_now = np.float64(time.time() - self._t0)

        t = np.array([t_now - 1e-3, t_now], dtype=np.float64)   # (2,)

        tcp = self._tcp_pose.astype(np.float64)
        tcp_hist = np.stack([tcp, tcp], axis=0)                # (2,6)

        q = self._q.astype(np.float64)
        q_hist = np.stack([q, q], axis=0)                       # (2,6)

        qd  = self._qd.astype(np.float64)
        qd_hist  = np.stack([qd, qd], axis=0)  

        return {
            "robot_timestamp": t,        # (2,)
            "ActualTCPPose": tcp_hist,   # (2,6)
            "ActualQ": q_hist,           # (2,6)
            "ActualQd": qd_hist,    
        }
    

    def get_state(self):
        # current (not history)
        tcp = self._tcp_pose.astype(np.float64)   # (6,)
        q   = self._q.astype(np.float64)          # (6,)
        qd  = self._qd.astype(np.float64)

        return {
            "robot_timestamp": np.float64(time.time() - self._t0),
            "ActualTCPPose": tcp.copy(),
            "ActualQ": q.copy(),
            "ActualQd": qd.copy(),
            # Add these for eval_real.py
            "TargetTCPPose": tcp.copy(),
            "TargetQ": q.copy(),
        }