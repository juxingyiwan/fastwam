"""CPU tests of temporal alignment and adapter state handling."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

from experiments.libero.visual_ttt.core import (
    LoRALinear, activate, adapter_state, frame_indices, install_lora,
    load_adapter_state, reset_adapters, window_starts,
)


class ContractsTest(unittest.TestCase):
    def test_time_alignment(self):
        starts = window_starts(61, 32, 8)
        self.assertEqual(len(set(starts)), 8)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 28)
        self.assertEqual(frame_indices(28, 32, 4).tolist(), list(range(28, 61, 4)))
        with self.assertRaises(ValueError):
            window_starts(33, 32, 8)
        with self.assertRaises(ValueError):
            frame_indices(0, 30, 4)

    def test_lora_freezes_base_resets_and_roundtrips(self):
        model = nn.Module()
        model.video_expert = nn.Module()
        model.video_expert.layers = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
        model.action_expert = nn.Linear(2, 2)
        x = torch.randn(2, 3)
        before = model.video_expert.layers(x).detach()
        base = {k: p.detach().clone() for k, p in model.named_parameters()}
        adapters = install_lora(model, 2, 2, r"layers\.\d+")
        reset_adapters(adapters, 123)
        torch.testing.assert_close(model.video_expert.layers(x), before, rtol=0, atol=0)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        model.video_expert.layers(x).square().mean().backward()
        optimizer.step()
        trained = model.video_expert.layers(x).detach()
        self.assertFalse(torch.equal(trained, before))
        self.assertIsNone(model.action_expert.weight.grad)
        for name, module in adapters.items():
            torch.testing.assert_close(module.base.weight, base[f"video_expert.{name}.weight"], rtol=0, atol=0)
        activate(adapters, True, False)
        merged_trained = model.video_expert.layers(x).detach()
        torch.testing.assert_close(merged_trained, trained)
        saved = adapter_state(adapters)
        activate(adapters, False)
        torch.testing.assert_close(model.video_expert.layers(x), before, rtol=0, atol=0)
        reset_adapters(adapters, 456)
        torch.testing.assert_close(model.video_expert.layers(x), before, rtol=0, atol=0)
        load_adapter_state(adapters, saved)
        torch.testing.assert_close(model.video_expert.layers(x), merged_trained, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            load_adapter_state(adapters, {})

    def test_compiled_forward_observes_adapter_switches_and_updates(self):
        layer = LoRALinear(nn.Linear(3, 4), 2, 2)
        x = torch.randn(2, 3)
        # Trace the complete graph and exercise Dynamo's guards on enabled.
        compiled = torch.compile(layer, backend="aot_eager", fullgraph=True)
        original_weight = layer.base.weight.detach().clone()
        adapters = {"layer": layer}
        with torch.no_grad():
            frozen = compiled(x).clone()
            activate(adapters, True, False)
            torch.testing.assert_close(compiled(x), frozen, rtol=0, atol=0)
            layer.lora_B.fill_(0.2)
            activate(adapters, True, False)
            first = compiled(x).clone()
            self.assertFalse(torch.equal(first, frozen))
            layer.lora_B.fill_(0.4)
            activate(adapters, True, False)
            second = compiled(x).clone()
            self.assertFalse(torch.equal(first, second))
            activate(adapters, False)
            torch.testing.assert_close(compiled(x), frozen, rtol=0, atol=0)
            torch.testing.assert_close(layer.base.weight, original_weight, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
