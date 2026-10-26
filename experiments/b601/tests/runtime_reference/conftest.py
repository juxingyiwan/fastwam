import time
from dataclasses import replace
import numpy as np
import pytest
from experiments.b601.runtime.config import Config, ModelConfig, RobotConfig
from experiments.b601.runtime.contracts import Observation


@pytest.fixture
def cfg(tmp_path):
    return Config(
        task='fixture', output_root=str(tmp_path/'runs'), save_rollout=False,
        model=ModelConfig(checkpoints={'5epoch':'/fixture/checkpoint.pt'}),
        robot=RobotConfig(camera_devices={'front':'/fixture/front','side':'/fixture/side'}, lock_paths=(str(tmp_path/'lock'),)),
    )


@pytest.fixture
def observation():
    now=time.monotonic()
    return Observation(np.arange(7,dtype=np.float32), {k:np.full((48,64,3),[255,30,15],dtype=np.uint8) for k in ('front','side')}, now,time.time(),now)
