import unittest
import numpy as np
from env.config import Config as config
from env.firefighting_env import FireFightingEnv

"""
This script contains the unit test suite for the `FireFightingEnv` environment.

It uses Python's built-in `unittest` framework to verify the core mechanics
and logic of the simulation. The tests are divided into two distinct suites:

1.  TestDefaultEnvironment: This suite tests general environment properties
    that do not require a controlled setup, such as determinism (seeding)
    and action space integrity. It uses a standard, randomized environment
    for each test.

2.  TestControlledEnvironment: This suite tests specific game logic, such
    as fire extinguish and reward calculation. To ensure predictable and
    repeatable outcomes, it temporarily modifies the global environment
    configuration to create a simple, non-random world with a single fire
    and no other entities. The original configuration is restored after
    the tests are complete.
"""


# --- Test Suite 1: For tests that can use the default, random environment ---
class TestDefaultEnvironment(unittest.TestCase):
    """
    Tests general environment mechanics using a standard, randomized setup.

    Each test in this class gets a fresh instance of the default `FireFightingEnv`,
    allowing for tests of properties like seeding and action handling without
    needing to control the specific placement of game objects.
    """

    def setUp(self):
        """
        Creates a fresh, default environment before each individual test case.
        """
        self.env = FireFightingEnv(render_mode=None, all_fires_known=True)

    def tearDown(self):
        """
        Cleans up resources by closing the environment after each test case.
        """
        self.env.close()

    def test_episode_is_deterministic_with_seed(self):
        """
        Verifies that when the environment is reset with the same seed, it produces
        the exact same sequence of observations and rewards for a fixed set of actions.
        This ensures that experiments are repeatable.
        """
        seed = 42
        fixed_actions = [FireFightingEnv.ACTION_RIGHT, FireFightingEnv.ACTION_STRAIGHT, FireFightingEnv.ACTION_DOUSE]

        # First Run
        obs1, info1 = self.env.reset(seed=seed)
        history1 = {'obs': [obs1], 'rewards': []}
        for action in fixed_actions:
            obs, reward, _, _, _ = self.env.step([action])
            history1['obs'].append(obs)
            history1['rewards'].append(reward)

        # Second Run
        obs2, info2 = self.env.reset(seed=seed)
        history2 = {'obs': [obs2], 'rewards': []}
        for action in fixed_actions:
            obs, reward, _, _, _ = self.env.step([action])
            history2['obs'].append(obs)
            history2['rewards'].append(reward)

        self.assertListEqual(history1['rewards'], history2['rewards'], "Rewards should be identical for a seeded run.")
        for i in range(len(history1['obs'])):
            obs_a = history1['obs'][i]
            obs_b = history2['obs'][i]
            for key in obs_a:
                np.testing.assert_array_equal(obs_a[key], obs_b[key],
                                              f"Observation '{key}' at step {i} should be identical.")

    def test_all_actions_run_without_error(self):
        """
        Ensures that the environment can process every possible valid action
        from the action space without crashing. This is a basic sanity check.
        """
        self.env.reset()
        all_actions = [FireFightingEnv.ACTION_STRAIGHT, FireFightingEnv.ACTION_LEFT, FireFightingEnv.ACTION_RIGHT, FireFightingEnv.ACTION_DOUSE]
        for action in all_actions:
            try:
                self.env.step([action])
            except Exception as e:
                self.fail(f"Environment crashed when stepping with action {action}. Error: {e}")


# --- Test Suite 2: For tests that require a simple, controlled environment ---
class TestControlledEnvironment(unittest.TestCase):
    """
    Tests specific game logic in a highly controlled, non-random environment.

    This class temporarily modifies the global `Config` object to create a
    minimalist world (e.g., 1 Fire, 0 Obstacles, 0 Dense Smoke). This allows for
    predictable testing of specific mechanics like fire extinguish and
    reward calculation. The original config is restored after all tests in this
    class are complete.
    """
    @classmethod
    def setUpClass(cls):
        """
        Runs ONCE before any tests in this class.

        It saves the original configuration values and then overrides them to
        create a simple, predictable test world. This prevents random entity
        placement from interfering with tests of specific game logic.
        """
        cls.original_config = {
            'MIN_FIRES': config.MIN_FIRES,
            'MAX_FIRE': config.MAX_FIRES,
            'FIRE_HP_EACH': config.FIRE_HP_EACH,
            'MIN_OBSTACLES': config.MIN_OBSTACLES,
            'MAX_OBSTACLES': config.MAX_OBSTACLES,
            'MIN_SMOKE_AREAS': config.MIN_SMOKE_AREAS,
            'MAX_SMOKE_AREAS': config.MAX_SMOKE_AREAS,
            'UAV_WATER_RANGE': config.UAV_WATER_RANGE
        }

        # Apply overrides for a simple, predictable world
        config.MIN_FIRES = 1
        config.MAX_FIRES = 1
        config.FIRE_HP_EACH = 1
        config.MIN_OBSTACLES = 0
        config.MAX_OBSTACLES = 0
        config.MIN_SMOKE_AREAS = 0
        config.MAX_SMOKE_AREAS = 0
        config.UAV_WATER_RANGE = 8

    @classmethod
    def tearDownClass(cls):
        """
        Runs ONCE after all tests in this class are complete.

        It restores the global configuration to its original state to ensure
        these modifications do not affect other tests in the suite.
        """
        for key, value in cls.original_config.items():
            setattr(config, key, value)

    def setUp(self):
        """
        Creates a new environment instance before each test, which will use
        the MODIFIED (controlled) configuration.
        """
        self.env = FireFightingEnv(render_mode=None)

    def tearDown(self):
        """
        Cleans up resources by closing the environment after each test case.
        """
        self.env.close()

    def test_fire_extinguish_logic(self):
        """
        Verifies that rewards and state changes are correct when a fire is
        discovered, hit, and extinguished in a single step.

        This test manually places the UAV and the fire in a perfect arrangement
        to guarantee the outcome and test the associated logic.
        """
        obs, _ = self.env.reset()

        # Manually place objects for a perfect test case
        self.env.uavs[0].pos = np.array([5, 5])
        self.env.uavs[0].orientation = 2  # Face Down

        fire_object = self.env.fires[0]
        fire_object.pos = np.array([5, 6])  # fire in front

        fire_object.known = False

        initial_hp = fire_object.hp

        # Execute the DOUSE action
        action = [FireFightingEnv.ACTION_DOUSE]
        obs, reward, terminated, _, info = self.env.step(action)

        # 1. Assert game state changes
        final_hp = fire_object.hp
        self.assertEqual(final_hp, initial_hp - 1, "Fire HP should decrease by 1.")
        self.assertTrue(terminated, "Episode should terminate when the only fire is extinguished.")

        # 2. Assert correct step reward
        expected_step_reward = (
                self.env.config.REWARD_DOUSE_FIRE + self.env.config.REWARD_FIND_FIRE +
                self.env.config.REWARD_EXTINGUISH_FIRE +
                self.env.config.PENALTY_STEP
        )
        self.assertAlmostEqual(reward, expected_step_reward,
                               msg="Step reward did not match DAMAGE + FIND + STEP penalties.")


