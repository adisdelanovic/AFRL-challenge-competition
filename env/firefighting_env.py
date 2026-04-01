from env.config import Config
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.renderer import Renderer
from env.game_object import Obstacle, DenseSmoke, Fire, UAV
from env.scorer import Scoring
import os
import csv
from collections import OrderedDict

"""
This script defines the core Fire Fighting environment, `FireFightingEnv`,
which simulates a problem space for training and evaluating control agents.

The environment is built using the Gymnasium (formerly OpenAI Gym) API and features:
- A configurable grid-based world.
- Multiple agent-controlled UAVs.
- Stationary hazards (smoke sites) and neutral obstacles that need to be avoided.
- Extinguishable fires that can be discovered and hit with water.
- A curriculum learning system where the complexity of the environment (number of entities)
  is determined by the `curriculum_stage`.
- A detailed, dictionary-based observation space.
- A multi-discrete action space for controlling multiple UAVs.
- An optional Pygame-based renderer for human visualization.
- Built-in metrics tracking and saving to a CSV file.
"""

class FireFightingEnv(gym.Env):
    """
    A Gymnasium environment for simulating Unmanned Aerial Vehicle (UAV) missions fore firefighting.

    In this environment, an agent controls one or more UAVs' with the primary objectives of
    finding and extinguishing fires while avoiding hazards like smoke sites and collisions.
    The environment's complexity is adjusted via a curriculum staging parameter, allowing
    for progressive learning from simple navigation to complex combat scenarios.
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}
    ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT, ACTION_DOUSE = 0, 1, 2, 3


    def __init__(self, render_mode=None, curriculum_stage=3, n_uavs=1, all_fires_known=False, agent_type='random', record=None, config_obj=None):

        """
        Initializes the uav Environment.

        Args:
            render_mode (str, optional): The mode for rendering. 'human' enables visualization.
                                         Defaults to None.
            curriculum_stage (int, optional): The stage of learning, which determines the
                                              complexity of the environment setup. Defaults to 3.
            N_UAVS (int, optional): The number of UAVs to be controlled by the agent.
                                    This overrides the value in the config file. Defaults to 1.
            all_fires_known (bool, optional): If True, all fires are visible to the agent
                                                from the start of the episode. Defaults to False.
            agent_type (str, optional): A string identifier for the agent being used, saved in metrics.
                                        Defaults to 'random'.
        """
        super().__init__()
        self.config = config_obj if config_obj is not None else Config()
        self.config.N_UAVS = n_uavs  # Allow overriding config
        self.agent_type = agent_type
        self.render_mode = render_mode
        self.all_fires_known = all_fires_known
        self.curriculum_stage = curriculum_stage
        self.current_episode = 0


        if render_mode:
            self.renderer = Renderer(self.config)

        # Scoring
        self.scorer = Scoring(self.config)

        # Action/Observation Spaces
        self.action_space = spaces.MultiDiscrete([4] * self.config.N_UAVS)
        self.observation_space = self._define_observation_space()

        # Environment State
        self.uavs = []
        self.fires = []
        self.obstacles = []
        self.dense_smoke_areas = []
        self.effects = []
        self.step_count = 0
        self.total_reward = 0
        self.initialize_metrics()

    def _define_observation_space(self):
        """
        Defines the structure of the observation space using `gymnasium.spaces`.

        The observation is a dictionary containing padded numpy arrays for all entities
        in the environment. Padding ensures a consistent shape regardless of the
        number of entities, which is crucial for RL models. A value of -1 is used for padding.

        Returns:
            gymnasium.spaces.Dict: The structured observation space.
        """
        return spaces.Dict({
            "uav_positions": spaces.Box(0, self.config.GRID_SIZE - 1, shape=(self.config.N_UAVS, 2), dtype=np.int64),
            "uav_orientations": spaces.Box(0, 3, shape=(self.config.N_UAVS,), dtype=np.int64),
            "uav_water_drops": spaces.Box(0, self.config.UAV_WATER_COUNT, shape=(self.config.N_UAVS,), dtype=np.int64),
            "obstacle_positions": spaces.Box(-1, self.config.GRID_SIZE - 1, shape=(self.config.MAX_OBSTACLES, 2),
                                             dtype=np.int64),
            "fire_positions": spaces.Box(-1, self.config.GRID_SIZE - 1, shape=(self.config.MAX_FIRES, 2),
                                           dtype=np.int64),
            "fire_hps": spaces.Box(-1, self.config.FIRE_HP_EACH, shape=(self.config.MAX_FIRES,), dtype=np.int64),
            "fires_known": spaces.MultiBinary(self.config.MAX_FIRES),
            "smoke_positions": spaces.Box(-1, self.config.GRID_SIZE - 1, shape=(self.config.MAX_SMOKE_AREAS, 2), dtype=np.int64),
        })

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state for a new episode.

        This involves clearing all entities, resetting the step count and reward,
        and then placing new entities based on the curriculum stage.

        Args:
            seed (int, optional): The seed for the random number generator. Defaults to None.
            options (dict, optional): Additional options for resetting the environment (not used).

        Returns:
            tuple: A tuple containing the initial observation (`obs`) and information (`info`).
        """
        super().reset(seed=seed)

        # Reset state
        self.step_count = 0
        self.total_reward = 0
        self.effects = []
        self.seed = seed if seed is not None else int(self.np_random.integers(0, 2 ** 31))
        self.uavs = []
        self.fires = []
        self.obstacles = []
        self.dense_smoke_areas = []
        self.current_episode = seed if seed is not None else self.current_episode + 1

        # Reset score
        self.scorer.reset()

        # Place entities
        self._place_entities()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def _place_entities(self):
        """
        Places uavs, fires, obstacles, and smoke sites on the grid.

        The placement logic ensures that entities do not spawn too close to the uavs'
        starting positions. The number of entities placed is determined by the
        `self.curriculum_stage`.
        """

        # --- Part 1: Place uavs First ---
        self.uavs = []
        # Keep track of where uavs are placed to avoid overlap
        uav_positions_set = set()

        while len(self.uavs) < self.config.N_UAVS:
            # Place uavs away from the edges of the map
            pos = self.np_random.integers(1, self.config.GRID_SIZE - 2, size=2)
            if tuple(pos) not in uav_positions_set:
                self.uavs.append(UAV(pos, self.config.UAV_WATER_COUNT))
                uav_positions_set.add(tuple(pos))

        # --- Part 2: Determine Valid Spawn Locations for Other Entities ---

        # Start with all possible coordinates on the grid
        all_coords = [(x, y) for x in range(self.config.GRID_SIZE) for y in range(self.config.GRID_SIZE)]
        uav_positions = [uav.pos for uav in self.uavs]

        # This creates the "no-spawn zone".
        valid_spawn_coords = [
            coord for coord in all_coords
            if all(np.linalg.norm(np.array(coord) - np.array(uav_pos)) >= self.config.MIN_SPAWN_DISTANCE_FROM_UAV
                   for uav_pos in uav_positions)
        ]

        # Determine entity counts based on curriculum stage

        if self.curriculum_stage == 1:
            n_smokes = 0
            n_fires = 0
            n_obstacles = 0
        elif self.curriculum_stage == 2:
            n_smokes = 0
            n_fires = 1
            n_obstacles = 0
        elif self.curriculum_stage == 3:
            n_smokes = 1
            n_fires = 1
            n_obstacles = 0
        elif self.curriculum_stage == 4:
            n_smokes = 1
            n_fires = 3
            n_obstacles = 1
        else:
            n_smokes = self.np_random.integers(self.config.MIN_SMOKE_AREAS, self.config.MAX_SMOKE_AREAS + 1)
            n_obstacles = self.np_random.integers(self.config.MIN_OBSTACLES, self.config.MAX_OBSTACLES + 1)
            n_fires = self.np_random.integers(self.config.MIN_FIRES, self.config.MAX_FIRES + 1)

        total_entities_to_place = n_obstacles + n_fires + n_smokes
        if len(valid_spawn_coords) < total_entities_to_place:
            raise ValueError(
                f"Not enough valid spawn locations away from uavs. "
                f"Required: {total_entities_to_place}, Available: {len(valid_spawn_coords)}. "
                f"Consider reducing MIN_SPAWN_DISTANCE_FROM_uav or the number of entities."
            )

        # Choose randomly from the *valid* list of coordinates
        chosen_indices = self.np_random.choice(len(valid_spawn_coords), size=total_entities_to_place, replace=False)
        chosen_coords = [valid_spawn_coords[i] for i in chosen_indices]

        # Create and place entities at the chosen coordinates
        self.obstacles = [Obstacle(pos) for pos in chosen_coords[:n_obstacles]]
        self.fires = [Fire(pos, self.config.FIRE_HP_EACH) for pos in
                        chosen_coords[n_obstacles:n_obstacles + n_fires]]
        self.dense_smoke_areas = [DenseSmoke(pos, self.config.SMOKE_RANGE, self.config.SMOKE_IMPAIRMENT_FACTOR) for pos in
                     chosen_coords[n_obstacles + n_fires:]]

        # Set fires to known if specified by the config
        if self.all_fires_known:
            for fire in self.fires:
                fire.known = True

    def step(self, actions):
        """
        Executes one time step in the environment based on the given actions.

        The step logic proceeds in a fixed order:
        1. Process agent movement actions and check for collisions.
        2. Process smoke area impairment against UAVs'.
        3. Process fire discovery by UAVs'.
        4. Process agent firing actions against fires.
        5. Check for episode termination or truncation conditions.

        Args:
            actions (list or np.ndarray): A list of actions, one for each uav.

        Returns:
            tuple: A tuple containing the new observation (`obs`), the reward (`reward`),
                   a boolean indicating if the episode is terminated (`terminated`), a boolean
                   indicating if the episode is truncated (`truncated`), and an info dict (`info`).
        """
        self.step_count += 1
        reward = self.config.PENALTY_STEP
        terminated = False

        self.scorer.time_step()

        self.effects = [fx for fx in self.effects if fx['timer'] > 0 and fx['type'] != 'smoke_impairment']
        for fx in self.effects: fx['timer'] -= 1

        # --- 1. Process Agent Actions (Movement) ---
        new_positions = [uav.pos.copy() for uav in self.uavs]
        for i, (uav, action) in enumerate(zip(self.uavs, actions)):
            # Process turns first
            if action == self.ACTION_LEFT:
                uav.turn(-1)
            elif action == self.ACTION_RIGHT:
                uav.turn(1)

            # A single move is executed for STRAIGHT, LEFT, and RIGHT actions.
            # The move is based on the orientation *after* the turn.
            if action != self.ACTION_DOUSE:
                new_positions[i] += uav.get_move_vector()

        # --- 2. Check for Collisions ---
        # Wall/Obstacle collisions
        for i, pos in enumerate(new_positions):
            if not (0 <= pos[0] < self.config.GRID_SIZE and 0 <= pos[1] < self.config.GRID_SIZE) or \
                    any(np.array_equal(pos, obs.pos) for obs in self.obstacles):
                reward += self.config.PENALTY_WALL_COLLISION
                self.scorer.crashed()
                terminated = True
                break

        # Inter-uav collisions
        if not terminated and len(set(map(tuple, new_positions))) < len(self.uavs):
            reward += self.config.PENALTY_UAV_COLLISION
            self.scorer.crashed()
            terminated = True

        if terminated:
            self.total_reward += reward
            if self.render_mode == "human": self.render()
            return self._get_obs(), reward, terminated, False, self._get_info()

        # If no collisions, update positions
        for i, uav in enumerate(self.uavs):
            uav.pos = new_positions[i]

        # --- 3. Dense Smoke impairment ---
        for smoke in self.dense_smoke_areas:
            for uav in self.uavs:
                if smoke.in_area(uav.pos) and self.np_random.random() < smoke.impairment_factor:
                    reward += self.config.PENALTY_IMPAIRED_BY_SMOKE
                    terminated = True
                    self.effects.append({'type': 'smoke_impairment', 'start': smoke.pos, 'end': uav.pos, 'timer': 4})
                    self.scorer.impaired_by_smoke()
                    break
            if terminated: break

        if terminated:
            self.total_reward += reward
            if self.render_mode == "human": self.render()
            return self._get_obs(), reward, terminated, False, self._get_info()

        # --- 4. fire Discovery ---
        for fire in self.fires:
            if not fire.known and not fire.is_extinguished:
                for uav in self.uavs:
                    if np.linalg.norm(uav.pos - fire.pos) <= self.config.UAV_VISION_RANGE:
                        fire.known = True
                        reward += self.config.REWARD_FIND_FIRE
                        self.scorer.fire_found()
                        break

        # --- 5. Dousing Logic (Intent-Based with Miss Visualization) ---
        for i, (uav, action) in enumerate(zip(self.uavs, actions)):
            if action == self.ACTION_DOUSE:
                # First, check if the agent can even fire.
                if not uav.has_water_drops():
                    # Agent tried to DOUSE with no ammo. This is a bad decision.
                    reward += self.config.PENALTY_WASTED_DOUSE_CMD
                    continue  # Move to the next agent

                # If we are here, a water will be expended.
                uav.expend_water()

                # Now, determine if that water hits or misses.
                intended_fire = None
                min_dist = float('inf')
                for fire in self.fires:
                    if fire.known and not fire.is_extinguished:
                        dist = np.linalg.norm(uav.pos - fire.pos)
                        if dist <= self.config.UAV_WATER_RANGE and dist < min_dist:
                            min_dist = dist
                            intended_fire = fire

                # --- OUTCOME 1: HIT ---
                if intended_fire is not None:
                    # douse a fire then give reward for hitting it
                    intended_fire.douse()
                    reward += self.config.REWARD_DOUSE_FIRE

                    # update the score for the hit
                    self.scorer.fire_doused()

                    # if the fire is extinguished give a larger reward
                    if intended_fire.is_extinguished:
                        self.scorer.fire_extinguished()

                        # Update the reward for extinguishing
                        reward += self.config.REWARD_EXTINGUISH_FIRE

                    self.effects.append({'type': 'water_splash', 'pos': intended_fire.pos, 'timer': 5})
                    douse_effect = {
                        'type': 'douse',
                        'start_pos': uav.pos.copy(),
                        'end_pos': intended_fire.pos.copy(),
                        'timer': self.config.DOUSE_DURATION,
                        'duration': self.config.DOUSE_DURATION
                    }
                    self.effects.append(douse_effect)


                # --- OUTCOME 2: MISS ---
                else:
                    # The decision to douse was bad. Apply penalty and create a "miss" visual.
                    reward += self.config.PENALTY_WASTED_DOUSE_CMD

                    forward_vector = np.array(Config.ORIENTATION_TO_VECTOR[uav.orientation])

                    # Calculate the end point for missing
                    miss_end_pos = uav.pos + forward_vector * self.config.UAV_WATER_RANGE

                    # Create the visual effect for the miss
                    douse_effect = {
                        'type': 'douse',
                        'start_pos': uav.pos.copy(),
                        'end_pos': miss_end_pos,
                        'timer': self.config.DOUSE_DURATION,
                        'duration': self.config.DOUSE_DURATION
                    }
                    self.effects.append({'type': 'water_splash', 'pos': miss_end_pos, 'timer': 5})
                    self.effects.append(douse_effect)

        # --- 6. Check End Conditions ---
        if self.fires and self.get_fires_extinguished() == len(self.fires):
            terminated = True

        truncated = not terminated and self.step_count >= self.config.MAX_EPISODE_STEPS
        self.total_reward += reward

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def get_fires_extinguished(self):
        """Counts the number of fires that have been extinguished."""
        return sum(1 for fire in self.fires if fire.is_extinguished)

    def get_fires_found(self):
        """Counts the number of fires that have been discovered."""
        return sum(1 for fire in self.fires if fire.known)

    def render(self):
        """Renders the current state of the environment using the Renderer."""
        if self.render_mode == "human":
            return self.renderer.render_human(self)
        elif self.render_mode == "rgb_array":
            return self.renderer.render_rgb_array(self)

    def close(self):
        """Closes the renderer window if it was created."""
        if self.render_mode and self.renderer:
            self.renderer.close()

    def _get_obs(self):
        """
        Constructs the observation dictionary from the current environment state.

        This method gathers data from all entities and formats it into padded numpy arrays
        that match the defined `observation_space`. The use of `OrderedDict` ensures a
        consistent key order.

        Returns:
            OrderedDict: The observation dictionary.
        """

        # Start with an empty OrderedDict
        obs = OrderedDict()

        # Add items in the desired, explicit order
        obs["uav_positions"] = np.array([uav.pos for uav in self.uavs], dtype=np.int64)
        obs["uav_orientations"] = np.array([uav.orientation for uav in self.uavs], dtype=np.int64)
        obs["uav_water_drops"] = np.array([uav.water_drops for uav in self.uavs], dtype=np.int64)

        # Pad obstacle positions
        obstacle_pos_array = np.full((self.config.MAX_OBSTACLES, 2), -1, dtype=np.int64)
        if self.obstacles:
            obstacle_pos_array[:len(self.obstacles)] = np.array([o.pos for o in self.obstacles])
        obs["obstacle_positions"] = obstacle_pos_array

        # Pad fire info
        fire_pos_array = np.full((self.config.MAX_FIRES, 2), -1, dtype=np.int64)
        fire_hp_array = np.full(self.config.MAX_FIRES, -1, dtype=np.int64)
        fires_known_array = np.zeros(self.config.MAX_FIRES, dtype=np.int64)
        for i, fire in enumerate(self.fires):
            if fire.known:
                fire_pos_array[i] = fire.pos
                fires_known_array[i] = 1
                fire_hp_array[i] = fire.hp
        obs["fire_positions"] = fire_pos_array
        obs["fire_hps"] = fire_hp_array
        obs["fires_known"] = fires_known_array

        # Pad smoke positions
        smoke_pos_array = np.full((self.config.MAX_SMOKE_AREAS, 2), -1, dtype=np.int64)
        if self.dense_smoke_areas:
            smoke_pos_array[:len(self.dense_smoke_areas)] = np.array([s.pos for s in self.dense_smoke_areas])
        obs["smoke_positions"] = smoke_pos_array

        return obs

    def _get_info(self):
        """
        Returns a dictionary with auxiliary diagnostic information.

        This is part of the standard Gymnasium API and can be used to pass
        extra information from the `step` function that is not part of the
        observation. Here, it provides the current score.

        Returns:
            dict: A dictionary containing the current score.
        """
        return {
            "score": self.scorer.current_score
        }

    def initialize_metrics(self):
        """
        Initializes a CSV file for storing episode metrics if it doesn't exist.
        Creates the 'metrics' directory and writes the header row to the CSV file.
        """
        csv_file_path = 'metrics/metrics.csv'
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

        file_exists = os.path.isfile(csv_file_path)
        if not file_exists:
            with open(csv_file_path, mode='w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(
                    ['episode', 'seed', 'agent_type', 'outcome', 'final_reward', 'steps_taken', 'num_obstacles',
                     'num_fires',
                     'fires_extinguished', 'num_dense_smoke_areas', 'num_uavs', 'fires_found', 'final_score'])

    def save_metrics(self, terminated, truncated):
        """
        Saves the results of the completed episode to the metrics CSV file.

        Args:
            terminated (bool): Whether the episode ended due to a terminal state (e.g., crash, success).
            truncated (bool): Whether the episode ended due to a time limit.
        """
        csv_file_path = 'metrics/metrics.csv'

        fires_extinguished = self.get_fires_extinguished()

        if not self.fires:
            if terminated:
                outcome = "CRASH"
            elif truncated:
                outcome = "TIMEOUT"
            else:
                outcome = "SUCCESS"
        elif fires_extinguished < len(self.fires):
            outcome = "CRASH" if terminated else "TIMEOUT"
        else:
            outcome = "SUCCESS"

        data_row = [
            self.current_episode,
            self.seed,
            self.agent_type,
            outcome,
            self.total_reward,
            self.step_count,
            len(self.obstacles),
            len(self.fires),
            fires_extinguished,
            len(self.dense_smoke_areas),
            self.config.N_UAVS,
            self.get_fires_found(),
            self.scorer.current_score
        ]

        with open(csv_file_path, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(data_row)

        print(f"--- Episode {self.current_episode} Finished ---")
        print(
            f"Result: {outcome} in {self.step_count} steps. ({fires_extinguished}/{len(self.fires)} fires extinguished)")
        print(f"Reward for this Episode: {self.total_reward}. Score: {self.scorer.current_score}")
