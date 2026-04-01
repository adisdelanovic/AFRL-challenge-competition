class Scoring:
    """
    A dedicated class to manage the human-readable game score,
    which is separate from the reinforcement learning agent's reward.

    This class tracks various in-game events and applies predefined score
    changes based on the values configured in the `Config` class.
    """
    def __init__(self, config):
        """
        Initializes the Scoring system with a reference to the global configuration.

        Args:
            config (Config): An instance of the Config class containing all scoring values.
        """
        self.config = config
        self.current_score = 0.0

    def reset(self):
        """
        Resets the current score to zero for a new episode.

        This method should be called at the beginning of every new episode
        to ensure scores are not carried over.
        """
        self.current_score = 0.0

    def time_step(self):
        """
        Applies a penalty for every time step that passes in the environment.

        This encourages agents to complete their objectives efficiently.
        The penalty value is defined by `config.SCORE_TIME_DECAY`.
        """
        self.current_score += self.config.SCORE_TIME_DECAY

    def fire_found(self):
        """
        Adds points to the score for discovering a new fire.

        This encourages exploration and fire identification.
        The score value is defined by `config.SCORE_fire_FOUND`.
        """
        self.current_score += self.config.SCORE_FIRE_FOUND

    def fire_extinguished(self):
        """
        Adds points to the score for extinguishing a fire.

        This is typically a major objective and thus provides a significant score boost.
        The score value is defined by `config.SCORE_FIRE_EXTINGUISHED`.
        """
        self.current_score += self.config.SCORE_FIRE_EXTINGUISHED

    def fire_doused(self):
        """
        Adds points to the score for hitting a fire with a water drop.

        This rewards successful hit, even if the fire is not yet extinguished.
        The score value is defined by `config.SCORE_FIRE_DOUSED`.
        """
        self.current_score += self.config.SCORE_FIRE_DOUSED

    def crashed(self):
        """
        Applies a penalty to the score when a UAV crashes

        Crashing includes collisions with walls, obstacles, or other UAVs'
        The penalty value is defined by `config.SCORE_CRASHED`.
        """
        self.current_score += self.config.SCORE_CRASHED

    def impaired_by_smoke(self):
        """
        Applies a penalty to the score when a UAV is impaired by dense smoke

        This penalizes dangerous maneuvering within smoke range.
        The penalty value is defined by `config.SCORE_IMPAIRED_BY_SMOKE`.
        """
        self.current_score += self.config.SCORE_IMPAIRED_BY_SMOKE

