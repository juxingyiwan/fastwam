"""RTC inference in a spawned process, isolated from control/recording Python work.

The child owns the only model/CUDA instance. The parent retains exclusive robot
ownership. RPC is single-flight, bounded, and never replays a failed request.
"""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import multiprocessing as mp
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

from .contracts import ProtocolError


def _load_connection(policy_args, task):
    from ..fastwam_policy.policy import B601Policy
    from .connection import FastWAMConnection

    policy = B601Policy(*policy_args)
    policy.load_context(task)
    return FastWAMConnection(policy)


def _serve(pipe, factory, policy_args, task, log_path):
    # Ctrl+C belongs to the attended parent, which shuts hardware down first.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from ..deploy import guard_offline_hardware

    sys.addaudithook(guard_offline_hardware)
    with ExitStack() as stack:
        if log_path is not None:
            stream = stack.enter_context(Path(log_path).open("a", buffering=1))
            stack.enter_context(redirect_stdout(stream))
            stack.enter_context(redirect_stderr(stream))
        try:
            connection = factory(policy_args, task)
            stack.callback(connection.close)
            pipe.send({"ready": True, "model": connection.open(),
                       "capability": connection.rtc_capability()})
            while True:
                message = pipe.recv()
                if message is None:
                    break
                request_id, args, kwargs = message
                prediction = connection.predict(*args, **kwargs)
                pipe.send({"request_id": request_id, "prediction": prediction})
        except (EOFError, BrokenPipeError):
            pass
        except BaseException:
            error = traceback.format_exc()
            print(error, file=sys.stderr, flush=True)
            try:
                pipe.send({"error": error})
            except (EOFError, OSError):
                pass
        finally:
            pipe.close()


class FastWAMProcessConnection:
    def __init__(self, checkpoint, device="cuda", num_inference_steps=10, seed=42,
                 *, task, log_path=None, inference_timeout_s=15.0,
                 load_timeout_s=180.0, _factory=_load_connection):
        self.timeout = float(inference_timeout_s)
        if self.timeout <= 0 or load_timeout_s <= 0:
            raise ValueError("Model process deadlines must be positive")
        self._state_lock = threading.Lock()
        self._call_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = self._busy = False
        self._sequence = 0
        self._close_status = None
        context = mp.get_context("spawn")  # Never fork a CUDA or hardware owner.
        self._pipe, child = context.Pipe()
        self._process = context.Process(
            target=_serve,
            args=(child, _factory, (str(checkpoint), device, num_inference_steps, seed),
                  task, str(log_path) if log_path is not None else None),
            name="fastwam-rtc-model", daemon=True,
        )
        self._process.start()
        child.close()
        try:
            ready = self._receive(float(load_timeout_s))
            if not ready.get("ready"):
                raise ProtocolError("Model process did not complete initialization")
            self._model = {**ready["model"], "backend": "local://fastwam-process",
                           "inference_pid": self._process.pid}
            self._capability = ready["capability"]
        except BaseException:
            self.close_with_status(2.0)
            raise

    def _receive(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            if self._closed:
                raise ProtocolError("Model process connection closed")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProtocolError("Model process response deadline exceeded")
            try:
                if self._pipe.poll(min(0.05, remaining)):
                    response = self._pipe.recv()
                    if "error" in response:
                        raise ProtocolError("Model process failed:\n" + response["error"])
                    return response
            except (EOFError, OSError) as exc:
                raise ProtocolError("Model process disconnected") from exc
            if not self._process.is_alive():
                raise ProtocolError(f"Model process exited: {self._process.exitcode}")

    def open(self):
        return dict(self._model)

    def rtc_capability(self):
        return dict(self._capability)

    def predict(self, observation, task, generation=0, *, rtc_request=None):
        observation.validate_payload(("front", "side"))
        if not self._call_lock.acquire(blocking=False):
            raise ProtocolError("Model process already has an outstanding request")
        try:
            with self._state_lock:
                if self._closed:
                    raise ProtocolError("Model process connection closed")
                self._busy = True
                self._sequence += 1
                request_id = self._sequence
            self._pipe.send((request_id, (observation, task, generation),
                             {"rtc_request": rtc_request}))
            response = self._receive(self.timeout)
            if response.get("request_id") != request_id:
                raise ProtocolError("Model process response ID mismatch")
            return response["prediction"]
        except BaseException:
            # A timed-out response must never be admitted by a later request.
            self.close_with_status(1.0)
            raise
        finally:
            with self._state_lock:
                self._busy = False
            self._call_lock.release()

    def close_with_status(self, timeout):
        with self._close_lock:
            if self._close_status is not None:
                return dict(self._close_status)
            started = time.monotonic()
            timeout = max(0.0, float(timeout))
            with self._state_lock:
                self._closed = True
                busy = self._busy
            forced = False
            if self._process.is_alive() and not busy:
                try:
                    self._pipe.send(None)
                except (EOFError, OSError):
                    pass
                self._process.join(timeout * 0.4)
            if self._process.is_alive():
                forced = True
                self._process.terminate()
                self._process.join(max(0, timeout * 0.8 - (time.monotonic()-started)))
            if self._process.is_alive():
                self._process.kill()
                self._process.join(max(0, timeout - (time.monotonic()-started)))
            alive = self._process.is_alive()
            self._pipe.close()
            status = {"kind": "process", "pid": self._process.pid,
                      "status": "unconfirmed" if alive else ("terminated" if forced else "joined"),
                      "timed_out": alive}
            if not alive:
                self._close_status = status
            return dict(status)

    def close(self):
        return self.close_with_status(2.0)
