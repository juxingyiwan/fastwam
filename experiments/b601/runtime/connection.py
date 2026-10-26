"""In-process FastWAM calls; this module never owns hardware."""

import time
import uuid

import numpy as np

from .contracts import InferenceTiming, Prediction, ProtocolError


class FastWAMConnection:
    def __init__(self, policy):
        self.policy = policy
        self.closed = False

    def open(self):
        return {
            "backend": "local://fastwam",
            "checkpoint_step": self.policy.checkpoint_step,
            "checkpoint": str(self.policy.checkpoint),
            "action_representation": "base_frame_ee_delta_same_row_measured_pose",
            "horizon": 32,
            "clip": False,
            "joint_limit_processing": "native_driver_clip",
            "tasks": list(self.policy.contexts),
        }

    def rtc_capability(self):
        model = self.policy.model.model_rtc_capability()
        contract = self.policy.adapter.contract
        return {
            **model,
            "supported": model["supported"] and contract["fps"] == 30,
            "representation": contract["representation"],
            "model_dt_s": 1 / contract["fps"],
            "horizon": self.policy.horizon,
            "flow_times": self.policy.rtc_flow_times(),
        }

    def predict(self, observation, task, generation=0, *, rtc_request=None):
        if self.closed:
            raise RuntimeError("Model connection closed")
        observation.validate_payload(("front", "side"))
        started = time.monotonic()
        inputs = dict(
            zip(
                self.policy.adapter.codec.config["recorded_feature_names"],
                observation.state,
            )
        )
        inputs.update(observation.views)
        dispatched = time.monotonic()
        ack = None
        if rtc_request is None:
            actions = self.policy.predict(inputs, task)
        else:
            from .rtc_contract import acknowledgement, validate_request

            validate_request(rtc_request)
            if rtc_request.generation != generation:
                raise ProtocolError("RTC generation mismatch")
            actions = self.policy.predict(inputs, task, rtc_request=rtc_request)
            ack = acknowledgement(rtc_request)
        finished = time.monotonic()
        if actions.shape != (32, 7) or not np.isfinite(actions).all():
            raise ProtocolError(
                "FastWAM must produce 32 finite seven-dimensional EE actions"
            )
        return Prediction(
            uuid.uuid4().hex,
            generation,
            actions,
            observation,
            finished,
            finished - dispatched,
            {},
            timing=InferenceTiming(started, dispatched, finished),
            rtc=ack,
        )

    def close(self):
        # An active CUDA call is not cancellable. The bounded worker reports this;
        # it cannot issue motor commands, and the launcher owns the process lifetime.
        self.closed = True
