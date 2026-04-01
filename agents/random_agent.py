"""
This script defines a simple RandomAgent for the Unmanned Aerial Vehicle (UAV)
environment. This agent serves as a baseline or a placeholder, demonstrating
the basic interaction with the environment's action space.

The RandomAgent makes decisions by selecting actions purely at random,
without any internal state, learning, or strategic consideration. It's
useful for:
- Verifying environment functionality and action space compatibility.
- Providing a baseline performance metric against which more sophisticated
  agents can be compared.
- Initial testing or debugging of the simulation.
"""

class RandomAgent:
    """
    A basic agent that selects actions randomly for each UAV in the environment.
    This agent does not implement any learning, planning, or state management,
    making it suitable for baseline comparisons or initial environment testing.
    """
    def __init__(self, env):
        """
        Initializes the RandomAgent.

        Args:
            env (gym.Env): The Gymnasium environment object, from which the
                           action space and number of UAVs are obtained.
        """
        self.action_space = env.unwrapped.action_space
        self.num_UAVS = env.unwrapped.config.N_UAVS

    def get_action(self, obs):
        """
        Generates a list of random actions, one for each Unmanned Aerial Vehicle (UAV).

        This method samples an action from the environment's action space independently
        for every UAV controlled by this agent, without considering the current
        observation or any internal state.

        Args:
            obs (dict): The current observation from the environment.
                        (Note: This agent ignores the observation as it acts randomly).

        Returns:
            list: A list containing `self.num_UAVs` randomly sampled actions.
                  Each action is an integer corresponding to an action in `FireFightingEnv.ACTION_SPACE`.
        """
        return [self.action_space.sample() for _ in range(self.num_UAVS)]

    def reset(self):
        """
        Resets the agent's internal state.

        For the RandomAgent, there is no internal state to reset, so this method
        simply passes. It's included for API consistency with other agents.
        """
        pass