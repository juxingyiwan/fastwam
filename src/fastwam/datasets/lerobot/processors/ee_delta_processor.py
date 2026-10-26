"""Physical EE-delta preprocessing with frozen, unclipped affine statistics."""

import torch

from .fastwam_processor import FastWAMProcessor


class EEAffineNormalizer:
    """Apply exactly the saved FP32 affine map in both directions."""

    def __init__(self, stats):
        self.stats = stats
        self.parameters = {}
        for field in ("action", "state"):
            self.parameters[field] = {}
            for key, values in stats[field].items():
                lower = torch.as_tensor(values["global_lower"], dtype=torch.float32)
                span = torch.as_tensor(values["global_span"], dtype=torch.float32)
                if (
                    lower.shape != (7,)
                    or span.shape != lower.shape
                    or not torch.isfinite(lower).all()
                    or not torch.isfinite(span).all()
                    or (span <= 0).any()
                ):
                    raise ValueError("Invalid saved EE affine parameters")
                self.parameters[field][key] = (lower, span)

    def transform(self, value, field, key="default", inverse=False):
        lower, span = self.parameters[field][key]
        value = value.float()
        lower, span = lower.to(value.device), span.to(value.device)
        if value.shape[-1] != len(lower):
            raise ValueError("Normalizer dimension does not match the data")
        return (
            (value + 1) * span / 2 + lower
            if inverse
            else 2 * (value - lower) / span - 1
        )

    def _apply(self, batch, inverse):
        for field in ("action", "state"):
            for key, value in batch.get(field, {}).items():
                batch[field][key] = self.transform(value, field, key, inverse)
        return batch

    def forward(self, batch):
        return self._apply(batch, False)

    def backward(self, batch):
        return self._apply(batch, True)

    def get_stats(self):
        return self.stats


class EEDeltaProcessor(FastWAMProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        masks = self.delta_action_dim_mask
        expected = torch.tensor([True] * 6 + [False])
        if (
            masks is None
            or set(masks) != {"default"}
            or not torch.equal(masks["default"], expected)
            or self.use_stepwise_action_norm
            or self.norm_exception_mode is not None
            or self.action_state_transforms
            or self.norm_default_mode not in {"min/max", "q01/q99"}
            or self.action_output_dim != 7
            or self.proprio_output_dim != 7
        ):
            raise ValueError(
                "EE deltas require a 6+1 padding mask, frozen global normalization and no extra coordinate transform"
            )

    def set_normalizer_from_stats(self, dataset_stats=None):
        contract = dataset_stats["ee_delta_contract"]
        if (
            contract["representation"] != "base_frame_ee_delta_v2"
            or contract["normalization"] != self.norm_default_mode
            or contract["horizon"] != self.num_obs_steps - 1
            or contract["statistics_scope"] != "training_episodes_valid_rows_once"
            or contract["clip"] is not False
        ):
            raise ValueError("Frozen EE statistics and processor disagree")
        self._normalizer = EEAffineNormalizer(dataset_stats)

    def postprocess(self, data):
        # This model predicts a future chunk; retain all H actions on decode.
        data = dict(data)
        data["state"] = data.pop("proprio")
        return self.normalizer.backward(self.action_state_merger.backward(data))

    def decode_actions(self, action):
        """Return physical deltas; compose each with fresh feedback downstream."""
        return self.normalizer.transform(action, "action", inverse=True)
