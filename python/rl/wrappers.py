import gymnasium as gym
from typing import Any


class TimeLimitTerminateWrapper(gym.Wrapper):
    """Limits the number of steps for an environment by setting terminated=True when the limit is reached."""

    def __init__(self, env: gym.Env, max_episode_steps: int):
        """Initializes the wrapper with an environment and the number of steps after which termination will occur.

        Args:
            env: The environment to apply the wrapper
            max_episode_steps: the environment step after which the episode is terminated
        """
        assert (
            isinstance(max_episode_steps, int) and max_episode_steps > 0
        ), f"Expect the `max_episode_steps` to be positive, actually: {max_episode_steps}"
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0 

    def step(self, action):
        """Steps through the environment and if the number of steps elapsed exceeds max_episode_steps then terminate.

        Args:
            action: The environment step action

        Returns:
            The environment step (observation, reward, terminated, truncated, info) with terminated=True
            if the number of steps elapsed >= max episode steps
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            terminated = True

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Resets the environment and sets the number of steps elapsed to zero.

        Args:
            seed: Seed for the environment
            options: Options for the environment

        Returns:
            The reset environment
        """
        self._elapsed_steps = 0
        return super().reset(seed=seed, options=options)