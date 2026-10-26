"""Map Wan's decreasing sigma and noise-minus-action field into RTC flow time."""

from fastwam.models.rtc.guidance import guided_velocity


def guided_wan_velocity(
    denoise, x, timestep, num_train_timesteps, condition, **context
):
    # Use the actual shifted scheduler time, never the denoising loop index.
    t = 1 - timestep.float().reshape(()) / num_train_timesteps

    def forward_velocity(_time, value):
        return -denoise(latents_action=value, timestep_action=timestep, **context)

    return -guided_velocity(forward_velocity, x, t, condition)
