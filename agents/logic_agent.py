import numpy as np
from collections import deque
from env.firefighting_env import FireFightingEnv

"""
This script defines the LogicAgent, a heuristic-based AI for controlling
an Unmanned Aerial Vehicle (UAV) in the FireFighting simulation environment.

The agent's primary goal is to explore the map to find and extinguish fires
while actively avoiding known hazards (smoke areas, obstacles) and walls.

It operates on a simple rule-based task system:
  - 'search' : Navigate toward unexplored cells to discover hidden fires.
  - 'douse'  : Navigate toward a known fire and extinguish it.

Key design decisions:
  - Waypoint scoring uses fully vectorised numpy operations (no Python loops
    over every grid cell) for efficiency.
  - Orientation arithmetic replaces floating-point angle math, avoiding
    edge cases and being easier to read.
  - The stuck-escape direction alternates left/right so the UAV doesn't
    repeatedly escape into the same wall.
  - Dousing does not require the UAV to be facing the fire, consistent
    with UAV_WATER_RANGE = 1 in the environment.
"""


class LogicAgent:
    # Movement vectors indexed by orientation (matches Config.ORIENTATION_TO_VECTOR)
    _FORWARD = {
        0: np.array([0, -1]),   # North
        1: np.array([1,  0]),   # East
        2: np.array([0,  1]),   # South
        3: np.array([-1, 0]),   # West
    }

    # Scoring constants for move evaluation
    _SCORE_OFF_GRID       = -2000
    _SCORE_UAV_COLLISION  = -1500
    _SCORE_HAZARD         = -1000   # smoke range or obstacle
    _SCORE_TURN_PENALTY   =    -1   # slight bias toward going straight

    # Scoring weights for waypoint selection
    _WEIGHT_UNVISITED =  1000
    _WEIGHT_DISTANCE  =   -10
    _WEIGHT_HAZARD    = -5000

    def __init__(self, env):
        """
        Initialises the LogicAgent.

        Args:
            env: The FireFighting environment instance. Used to read config values.
        """
        self.config = env.unwrapped.config
        self.reset()

    def reset(self):
        """Resets all internal state for a new episode."""
        n = self.config.N_UAVS

        self.uav_tasks           = ['search'] * n
        self.uav_assigned_fires  = [None] * n
        self.recent_positions    = [deque(maxlen=4) for _ in range(n)]
        self.stuck_counters      = [0] * n

        # Alternates each time a UAV escapes, so it doesn't always turn the
        # same way and immediately get stuck again.
        self.stuck_escape_dir    = [FireFightingEnv.ACTION_RIGHT] * n

        self.search_waypoints    = [None] * n
        self.visited_grid        = np.zeros((self.config.GRID_SIZE, self.config.GRID_SIZE), dtype=int)
        self.hazard_grid         = np.zeros((self.config.GRID_SIZE, self.config.GRID_SIZE), dtype=int)

    def get_action(self, obs):
        """
        Determines the best action for each UAV given the current observation.

        The decision pipeline runs in this order every step:
          1. Drop any fire assignments that are no longer valid.
          2. Assign unassigned UAVs to newly discovered fires.
          3. Update the internal world model (visited cells, hazard map).
          4. Generate one action per UAV based on its current task.

        Args:
            obs (dict): The observation dictionary from the environment.

        Returns:
            list[int]: One action per UAV.
        """
        actions = [FireFightingEnv.ACTION_STRAIGHT] * self.config.N_UAVS

        # Parse observation into convenient structures
        real_smoke_areas = [pos for pos in obs['smoke_positions'] if pos[0] != -1]
        known_fires = [
            {'id': i, 'pos': obs["fire_positions"][i], 'hp': obs["fire_hps"][i]}
            for i in range(len(obs["fire_hps"]))
            if obs["fire_hps"][i] > 0 and obs["fires_known"][i] == 1
        ]
        valid_fire_pos = {tuple(f['pos']) for f in known_fires}

        # --- 1. Maintain Fire Lock ---
        # Release any UAV whose assigned fire has been extinguished or is unknown.
        for i in range(self.config.N_UAVS):
            if self.uav_tasks[i] == 'douse':
                if (self.uav_assigned_fires[i] is None or
                        tuple(self.uav_assigned_fires[i]) not in valid_fire_pos):
                    self.uav_tasks[i] = 'search'
                    self.uav_assigned_fires[i] = None

        # --- 2. Assign New Fires ---
        # Sort unassigned fires by distance to the first available UAV so the
        # nearest fire is always targeted first. HP sort is skipped because
        # FIRE_HP_EACH = 1 makes all fires equal in health.
        unassigned_uavs = [i for i, task in enumerate(self.uav_tasks) if task == 'search']
        assigned_fire_pos = {tuple(pos) for pos in self.uav_assigned_fires if pos is not None}
        unassigned_fires = [f for f in known_fires if tuple(f['pos']) not in assigned_fire_pos]

        if unassigned_fires and unassigned_uavs:
            # Sort fires by distance to the first searching UAV
            ref_pos = obs['uav_positions'][unassigned_uavs[0]]
            unassigned_fires.sort(key=lambda f: np.linalg.norm(ref_pos - f['pos']))

            for fire in unassigned_fires:
                if not unassigned_uavs:
                    break
                # Pick the closest UAV with water remaining
                available = sorted(
                    [(np.linalg.norm(obs['uav_positions'][i] - fire['pos']), i)
                     for i in unassigned_uavs if obs['uav_water_drops'][i] > 0]
                )
                if not available:
                    continue
                _, uav_id = available[0]
                self.uav_tasks[uav_id] = 'douse'
                self.uav_assigned_fires[uav_id] = fire['pos']
                unassigned_uavs.remove(uav_id)

        # --- 3. Update Internal World Model ---
        self._update_hazard_grid(obs)
        self._update_visited_grid(obs)

        # --- 4. Generate Actions ---
        for i in range(self.config.N_UAVS):
            my_pos      = obs['uav_positions'][i]
            my_orient   = obs['uav_orientations'][i]
            other_uavs  = {tuple(pos) for j, pos in enumerate(obs['uav_positions']) if j != i}

            # Stuck detection — if the UAV barely moved over its last few steps,
            # increment the stuck counter.
            if len(self.recent_positions[i]) == self.recent_positions[i].maxlen:
                positions = np.array(self.recent_positions[i])
                max_dist  = np.max(np.linalg.norm(positions - positions[0], axis=1))
                self.stuck_counters[i] = self.stuck_counters[i] + 1 if max_dist < 1.5 else 0
            self.recent_positions[i].append(my_pos.copy())

            if self.uav_tasks[i] == 'search':
                actions[i] = self._search_action(i, my_pos, my_orient, real_smoke_areas, other_uavs, obs)
            else:
                actions[i] = self._douse_action(i, my_pos, my_orient, real_smoke_areas, other_uavs, obs)

        return actions

    def _search_action(self, uav_id, my_pos, my_orient, smoke_areas, other_uavs, obs):
        """Returns the best action for a UAV currently in 'search' mode."""
        # Escape maneuver — alternate turn direction so the UAV doesn't
        # repeatedly escape into the same obstacle.
        if self.stuck_counters[uav_id] > 3:
            escape = self.stuck_escape_dir[uav_id]
            self.stuck_escape_dir[uav_id] = (
                FireFightingEnv.ACTION_LEFT
                if escape == FireFightingEnv.ACTION_RIGHT
                else FireFightingEnv.ACTION_RIGHT
            )
            self.stuck_counters[uav_id] = 0
            self.recent_positions[uav_id].clear()
            return escape

        # Assign a new waypoint if we've reached the current one or don't have one
        if (self.search_waypoints[uav_id] is None or
                np.array_equal(my_pos, self.search_waypoints[uav_id])):
            self._assign_search_waypoint(uav_id, my_pos)

        return self._get_best_move(my_pos, my_orient, self.search_waypoints[uav_id],
                                   smoke_areas, other_uavs)

    def _douse_action(self, uav_id, my_pos, my_orient, smoke_areas, other_uavs, obs):
        """Returns the best action for a UAV currently in 'douse' mode."""
        fire_pos    = self.uav_assigned_fires[uav_id]
        is_in_range = np.linalg.norm(my_pos - fire_pos) <= self.config.UAV_WATER_RANGE
        has_water   = obs['uav_water_drops'][uav_id] > 0

        # No facing requirement — the environment accepts a douse from any
        # adjacent cell regardless of orientation (UAV_WATER_RANGE = 1).
        if is_in_range and has_water:
            return FireFightingEnv.ACTION_DOUSE

        # Not in range yet — navigate toward the fire
        return self._get_best_move(my_pos, my_orient, fire_pos, smoke_areas, other_uavs)

    def _update_visited_grid(self, obs):
        """
        Marks all cells within each UAV's vision radius as visited.
        Uses a bounding-box pre-filter then a circular distance check.
        """
        r    = self.config.UAV_VISION_RANGE
        size = self.config.GRID_SIZE

        for uav_pos in obs['uav_positions']:
            x0, y0 = uav_pos
            min_x, max_x = max(0, x0 - r), min(size - 1, x0 + r)
            min_y, max_y = max(0, y0 - r), min(size - 1, y0 + r)

            for x in range(int(min_x), int(max_x) + 1):
                for y in range(int(min_y), int(max_y) + 1):
                    if np.linalg.norm(np.array([x, y]) - uav_pos) <= r:
                        self.visited_grid[y, x] = 1

    def _update_hazard_grid(self, obs):
        """
        Rebuilds the hazard map from scratch each step using vectorised numpy
        operations. Marks cells within smoke radius and obstacle cells.
        """
        self.hazard_grid.fill(0)
        size = self.config.GRID_SIZE

        # Build coordinate grid once
        xs, ys = np.meshgrid(np.arange(size), np.arange(size))  # both (size, size)
        coords = np.stack([xs, ys], axis=-1).astype(float)       # (size, size, 2)

        # Mark smoke areas
        for smoke_pos in obs['smoke_positions']:
            if smoke_pos[0] == -1:
                continue
            dists = np.linalg.norm(coords - smoke_pos, axis=-1)
            self.hazard_grid[dists <= self.config.SMOKE_RANGE] = 1

        # Mark obstacles
        for obs_pos in obs['obstacle_positions']:
            if obs_pos[0] == -1:
                continue
            x, y = int(obs_pos[0]), int(obs_pos[1])
            if 0 <= x < size and 0 <= y < size:
                self.hazard_grid[y, x] = 1

    def _assign_search_waypoint(self, uav_id, my_pos):
        """
        Picks the highest-scoring unvisited, safe cell as the next waypoint.
        Fully vectorised — no Python loop over grid cells.

        Score = WEIGHT_UNVISITED * (not visited)
              + WEIGHT_DISTANCE  * distance
              + WEIGHT_HAZARD    * (is hazardous)
        """
        size = self.config.GRID_SIZE
        xs, ys = np.meshgrid(np.arange(size), np.arange(size))
        coords  = np.stack([xs, ys], axis=-1).astype(float)

        distances = np.linalg.norm(coords - my_pos, axis=-1)

        # visited_grid and hazard_grid are indexed [y, x], matching ys/xs above
        scores = (
            (self.visited_grid == 0) * self._WEIGHT_UNVISITED +
            distances                * self._WEIGHT_DISTANCE  +
            (self.hazard_grid == 1)  * self._WEIGHT_HAZARD
        )

        best_yx = np.unravel_index(np.argmax(scores), scores.shape)
        self.search_waypoints[uav_id] = np.array([best_yx[1], best_yx[0]])


    def _get_best_move(self, current_pos, current_orient, target_pos, smoke_areas, other_uavs):
        """
        Scores all three movement actions (straight, left, right) and returns
        the highest-scoring one. Considers distance to target, wall proximity,
        UAV collisions, and hazards (smoke + obstacles).

        Even invalid moves are scored so the agent always picks the "least bad"
        option rather than defaulting arbitrarily.
        """
        action_scores = {}

        for action in (FireFightingEnv.ACTION_STRAIGHT,
                       FireFightingEnv.ACTION_LEFT,
                       FireFightingEnv.ACTION_RIGHT):

            if action == FireFightingEnv.ACTION_LEFT:
                next_orient = (current_orient - 1 + 4) % 4
            elif action == FireFightingEnv.ACTION_RIGHT:
                next_orient = (current_orient + 1) % 4
            else:
                next_orient = current_orient

            move_delta = self._FORWARD[next_orient]
            next_pos   = current_pos + move_delta
            next_tuple = tuple(next_pos)

            # Base score: reward for closing distance to target
            score = (np.linalg.norm(current_pos - target_pos) -
                     np.linalg.norm(next_pos    - target_pos)) * 10

            # Slight penalty for turning to prefer straight-line travel
            if action != FireFightingEnv.ACTION_STRAIGHT:
                score += self._SCORE_TURN_PENALTY

            # Wall / out-of-bounds penalty
            size = self.config.GRID_SIZE
            if not (0 <= next_pos[0] < size and 0 <= next_pos[1] < size):
                score += self._SCORE_OFF_GRID
            else:
                # UAV collision penalty
                if next_tuple in other_uavs:
                    score += self._SCORE_UAV_COLLISION

                # Hazard penalty — covers both smoke areas and obstacles
                nx, ny = int(next_pos[0]), int(next_pos[1])
                if self.hazard_grid[ny, nx] == 1:
                    score += self._SCORE_HAZARD

            action_scores[action] = score

        return max(action_scores, key=action_scores.get)

    def _get_turn_to_face(self, uav_pos, uav_orient, target_pos):
        """
        Returns the turn action that most efficiently rotates the UAV toward
        a target position using integer orientation arithmetic — no trig needed.

        Returns ACTION_STRAIGHT if already facing the target (or on top of it).
        """
        delta = target_pos - uav_pos
        if np.array_equal(delta, [0, 0]):
            return FireFightingEnv.ACTION_STRAIGHT

        # Find which orientation has the highest dot product with delta
        best_orient = max(range(4), key=lambda o: np.dot(self._FORWARD[o], delta))

        diff = (best_orient - uav_orient) % 4
        if diff == 0:
            return FireFightingEnv.ACTION_STRAIGHT
        elif diff == 1:
            return FireFightingEnv.ACTION_RIGHT
        else:
            # diff == 3: one left turn is faster than three right turns
            return FireFightingEnv.ACTION_LEFT