import time
import numpy as np

class DummyGripperController:

    def __init__(self, *args, **kwargs):
        self._t0 = time.time()
        self._width = np.float64(kwargs.get("init_width", 0.08))

    @property
    def is_ready(self):
        return True

    def start(self, wait=False): pass
    def stop(self, wait=False): pass
    def start_wait(self): pass
    def stop_wait(self): pass
    def schedule_waypoint(self, *args, **kwargs): pass

    def get_state(self):
        return self.get_all_state()

    def get_all_state(self):
        t_now = np.float64(time.time() - self._t0)
        t = np.array([t_now - 1e-3, t_now], dtype=np.float64)   # (2,)

        # constant dummy "position" (whatever units the env expects)
        # If you want, set this to self._width, but keep it simple for now.
        pos = np.array([self._width, self._width], dtype=np.float64)            # (2,)

        return {
            "gripper_timestamp": t,
            "gripper_position": pos,   # <-- required
        }