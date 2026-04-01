import gymnasium as gym
import numpy as np
from env.firefighting_env import FireFightingEnv

"""
This file demonstrates how to create a custom Gymnasium ObservationWrapper for the
existing `FireFightingEnv`.

The purpose of this wrapper, `SimpleObservationWrapper`, is to drastically simplify
the complex, dictionary-based observation space provided by the base environment.
Many standard reinforcement learning algorithms work best with a simple, flat
vector of numbers. This process of manually selecting and processing key information
from a complex state is a form of "feature engineering."

Our goal is to reduce the entire environment state down to only the most critical
information for a simple "seek and find" task: the agent's own position and the
position of its closest known fire. The wrapper takes the original, large
observation dictionary as input and outputs a small, 4-element NumPy array
that an agent can more easily learn from.

Note that we do not need to implement `reset()` or `step()` methods ourselves;
the `gymnasium.ObservationWrapper` base class handles this boilerplate for us,
ensuring our `observation()` method is called at the correct times.

For more details on Gymnasium wrappers, refer to the official documentation:
https://gymnasium.farama.org/api/wrappers/
"""

class SimpleObservationWrapper(gym.ObservationWrapper):
    """
    Wraps the FireFightingEnv to simplify its complex observation dictionary into a
    compact, 1D NumPy array.

    This wrapper extracts only the position of the first uav and the position of the
    closest known fire, creating a focused observation space suitable for simpler
    RL agents or for testing basic navigation policies.

    The new observation space is a 4-element array:
    `[uav_x, uav_a, closest_fire_x, closest_fire_y]`

    If no fires are currently known, the fire position will be `[-1.0, -1.0]`.
    """
    def __init__(self, env):
        """
        Initializes the SimpleObservationWrapper.

        Args:
            env (gym.Env): The `F` instance to wrap.
        """
        # Initialize the parent class with the environment.
        super().__init__(env)

        # Define the new, simplified observation space. It's a Box of 4 continuous
        # values (x/y for uav, x/y for fire) ranging from the grid boundaries.
        grid_size = self.env.config.GRID_SIZE
        self.observation_space = gym.spaces.Box(
            low=0,
            high=grid_size,
            shape=(4,),
            dtype=np.float32
        )

    def observation(self, obs):
        """
        Transforms the original complex observation dictionary into a simplified
        4-element NumPy array.

        This method is automatically called by the wrapper after every `env.step()`
        and `env.reset()` call.

        Args:
            obs (dict): The original observation dictionary from the `FireFightingEnv`.

        Returns:
            np.ndarray: The new, simplified 4-element observation array.
        """
        # --- Original `FireFightingEnv` observation structure for reference ---
        # {
        #     "uav_positions":      Box(shape=(N, 2)),
        #     "uav_orientations":   Box(shape=(N,)),
        #     "uav_water_drops":    Box(shape=(N,)),
        #     "obstacle_positions": Box(shape=(MAX_OBSTACLES, 2)),
        #     "fire_positions":     Box(shape=(MAX_fireS, 2)),
        #     "fire_hps":           Box(shape=(MAX_fireS,)),
        #     "fires_known":        MultiBinary(MAX_fireS),
        #     "smoke_positions":      Box(shape=(MAX_dense_smokeS, 2)),
        # }

        # --- 1. Get the uav's information ---
        uav_pos = obs['uav_positions'][0]

        # --- 2. Find the closest known fire ---
        closest_fire_pos = np.array([-1.0, -1.0])
        min_dist = float('inf')

        for i, pos in enumerate(obs['fire_positions']):
            if obs['fires_known'][i] == 1 and pos[0] != -1:
                dist = np.linalg.norm(uav_pos - pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_fire_pos = pos

        # --- 3. Create and return the new observation ---
        new_obs = np.concatenate([
            uav_pos,
            closest_fire_pos
        ]).astype(np.float32)

        return new_obs
