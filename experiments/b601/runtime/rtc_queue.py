"""Single-owner EE-action FIFO; holding never consumes a policy action."""

import math
import numpy as np


def elapsed_steps(seconds, dt):
    if not math.isfinite(seconds) or seconds < 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid RTC elapsed time")
    return max(0, math.ceil(seconds / dt - 1e-9))


class ActionQueue:
    def __init__(self, horizon, execution_horizon=None):
        self.horizon = horizon
        self.execution_horizon = horizon if execution_horizon is None else execution_horizon
        if type(self.execution_horizon) is not int or not 1 <= self.execution_horizon <= horizon:
            raise ValueError("RTC execution horizon must be within the prediction horizon")
        self.consumed = 0
        self.clear()

    def clear(self):
        self.actions = np.empty((0, 7), dtype=np.float32)
        self.index = self.offset = 0

    def __len__(self):
        return len(self.actions) - self.index

    def leftover(self):
        return self.actions[self.index :].copy()

    def replace(self, actions, delay):
        actions = np.asarray(actions)
        if (
            actions.shape != (self.horizon, 7)
            or actions.dtype.kind not in "fiu"
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Invalid RTC action chunk")
        if type(delay) is not int or delay < 0:
            raise ValueError("Invalid RTC merge delay")
        if delay >= self.execution_horizon:
            return False
        # The unused prediction tail must not leak back through the next prefix.
        self.actions = actions[delay : self.execution_horizon].copy()
        self.index, self.offset = 0, delay
        return True

    def peek(self):
        if not len(self):
            return None
        return self.actions[self.index].copy(), self.offset + self.index

    def commit(self):
        """Called only after the actor successfully sends this policy target."""
        if not len(self):
            raise RuntimeError("Cannot consume an empty RTC queue")
        self.index += 1
        self.consumed += 1
