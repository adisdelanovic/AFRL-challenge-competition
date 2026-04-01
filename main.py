import os
import csv
import argparse
from agents.logic_agent import LogicAgent
from agents.random_agent import RandomAgent
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from utils.parser_action import RecordAction
from env.firefighting_env import FireFightingEnv


"""
This script serves as an example entry point for running simulations in the 
environment.

It allows for the selection of different autonomous agents (e.g., 'logic', 'random') 
and simulation parameters via command-line arguments. The script initializes the 
environment and the chosen agent, runs a specified number of episodes, and saves 
the results of each episode to a metrics file.

This is primarily used for testing agents, evaluating performance, or generating
simulation data without involving a reinforcement learning training loop.
"""

def main(agent_type, num_episodes, render_mode, fires_known, record):
        """
        Initializes and runs the main simulation loop.

        This function sets up the environment and the specified agent, then iterates
        through a series of episodes. In each episode, it resets the environment and agent,
        then steps through the simulation until a termination or truncation condition is met.
        Finally, it logs the results of each episode.

        Args:
            agent_type (str): The type of agent to use ('logic' or 'random').
            num_episodes (int): The total number of simulation episodes to run.
            render_mode (str or None): The mode for rendering. 'human' enables visualization,
                                       while None runs the simulation headlessly.
            fires_known (bool): If True, all fires are visible to the agent from the
                                  start of each episode. If False, fires must be discovered.
        """

       # Initialize the UAV environment with the specified configuration
        env = FireFightingEnv(render_mode=render_mode, n_uavs=1, curriculum_stage=5,  all_fires_known=fires_known, agent_type=agent_type, record=record)

        video_folder = "videos"
        if render_mode == "rgb_array":
            env = RecordVideo(env, video_folder=video_folder, name_prefix=record['prefix'], episode_trigger=lambda x: x % record['interval'] == 0)
        

        
        agent = None
        if agent_type == "logic":
            agent = LogicAgent(env)
        elif agent_type == "random":
            agent = RandomAgent(env)

        print(f"--- Running Simulation with {agent_type.upper()} AGENT for {env.unwrapped.config.N_UAVS} UAV(s) ---")
        print(f"Competition will run for {num_episodes} episodes.")

        for episode in range(num_episodes):
            obs, info = env.reset(seed=episode)

            if agent: agent.reset()

            terminated, truncated = False, False
            print(
                f"\n--- Starting Ep {episode} (Obstacles: {len(env.unwrapped.obstacles)}, Fires: {len(env.unwrapped.fires)}, Dense Smoke areas: {len(env.unwrapped.dense_smoke_areas)}) ---")

            while not terminated and not truncated:
                action = agent.get_action(obs) if agent else [env.action_space.sample() for _ in
                                                              range(env.unwrapped.config.N_UAVS)]
                obs, reward, terminated, truncated, info = env.step(action)
            
            # Save the metrics for each episode
            env.unwrapped.save_metrics(terminated, truncated)
            
            if render_mode == "rgb_array":
                if env.recording:
                    # capture last frame
                    print(f"\t*** Saved {record['prefix']} episode {episode} in {video_folder} ***")
                    env._capture_frame() 

        env.close()


if __name__ == '__main__':
    # --- Command-Line Argument Parsing ---
    # This section allows the script to be configured and run from the command line.
    parser = argparse.ArgumentParser(description="Run the UAV Simulation Environment")

    parser.add_argument(
        '--agent',
        type=str,
        choices=['logic', 'random'],
        default='logic',
        help="The type of agent to run ('logic' or 'random')."
    )

    parser.add_argument(
        '--render',
        action='store_true',
        help="Enable graphical rendering of the simulation."
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help="The number of episodes to run."
    )

    parser.add_argument(
        '--fires_known',
        action='store_true',
        help="If present, all fires are visible on the map from the start."
    )

    parser.add_argument(
        '--record',
        action=RecordAction,  # Use our custom action
        nargs='*',            # Allow zero or more arguments for this flag
        default=None,         # If --record is NOT present, the value will be None
        help="Enable recording. \n"
             "Usage examples:\n"
             "  --record                (records with prefix 'test' every 1 episode)\n"
             "  --record eval           (records with prefix 'eval' every 1 episode)\n"
             "  --record 5              (records with prefix 'test' every 5 episodes)\n"
             "  --record eval 5         (records with prefix 'eval' every 5 episodes)\n"
             "  --record 5 eval         (order does not matter)"
    )

    args = parser.parse_args()

    if args.record:
        if args.episodes / args.record['interval'] > 10:
            user = input("Warning going to record more than 10 episodes; continue? [Y/y/yes] [N/n/No]:")
            if user.lower() in ["n", "no"]:
                print("try again")
                exit()

    render_mode_arg = "human" if args.render and not args.record else None

    if args.record:
        render_mode_arg = "rgb_array"


    # Call the main function with the parsed arguments
    main(agent_type=args.agent, num_episodes=args.episodes, render_mode=render_mode_arg, fires_known=args.fires_known, record=args.record)
