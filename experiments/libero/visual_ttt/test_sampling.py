"""Check collection/evaluation RNG behavior without loading a model or simulator."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
from torch import nn

from experiments.libero import run_visual_ttt_worker as worker


class SamplingTest(unittest.TestCase):
    def test_evaluation_rejects_a_false_ttt_label(self):
        model = nn.Sequential(worker.LoRALinear(nn.Linear(2, 2), 1, 1))
        cfg = NS(EVALUATION=NS(task_id=0))
        with self.assertRaisesRegex(RuntimeError, "activation disagree"):
            worker.evaluate(cfg, None, [], model, None, None, None, "test", "visual_ttt_101")
        model[0].enabled = True
        with self.assertRaisesRegex(RuntimeError, "activation disagree"):
            worker.evaluate(cfg, None, [], model, None, None, None, "test", "frozen")

    def test_collection_and_evaluation_use_the_configured_seed(self):
        for configured_seed in (42, 317):
            with self.subTest(seed=configured_seed), tempfile.TemporaryDirectory() as temporary:
                calls = []
                environment_seeds = []
                class Env:
                    def seed(self, seed):
                        environment_seeds.append(seed)
                        np.random.seed(seed)
                    def reset(self): pass
                    def set_init_state(self, state):
                        self.state = int(state[0])
                        return self.state
                    def step(self, action): return self.state, 0, False, {}
                    def close(self): pass
                def make_env(task, resolution, seed):
                    env = Env()
                    env.seed(seed)
                    return env, task.language
                def predict(obs, task, cfg, *args):
                    calls.append((obs, cfg.seed, np.random.random()))
                    return np.zeros((32, 7), dtype=np.float32)
                def episode(**kwargs):
                    obs = kwargs['env'].set_init_state(kwargs['initial_state'])
                    for _ in range(4):
                        predict(obs, task, kwargs['cfg'])
                    return False, [], [], None
                cfg = NS(seed=configured_seed,
                         data=NS(train=NS(num_frames=33, action_video_freq_ratio=4, video_size=[224, 224])),
                         EVALUATION=NS(task_id=0, num_steps_wait=0, replan_steps=10),
                         TTT=NS(collection_steps=40, windows_per_rollout=1, save_eval_videos=False))
                ev = NS(get_libero_env=make_env, LIBERO_ENV_RESOLUTION=256, run_single_episode=episode)
                states = np.array([[0], [1]], dtype=np.float64)
                task = NS(language='test')
                root = Path(temporary)
                with patch.object(worker, 'predict', predict), patch.object(worker, 'snapshot',
                     lambda obs, *a: (np.zeros((2, 2, 3), dtype=np.uint8), np.zeros(8), np.array([obs]))):
                    worker.collect(cfg, task, states, None, None, ev, root/'collection', 'test')
                    collection = list(calls)
                    calls.clear()
                    worker.evaluate(cfg, task, states, None, None, ev, root/'evaluation.json', 'test', 'frozen')
                self.assertEqual(collection, calls)
                self.assertEqual([seed for _, seed, _ in calls], [configured_seed]*8)
                self.assertEqual(environment_seeds, [configured_seed]*2)
                self.assertEqual(cfg.seed, configured_seed)


if __name__ == '__main__':
    unittest.main()
