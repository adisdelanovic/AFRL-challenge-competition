Distribution Statement A: Approved for Public Release; Distribution Unlimited: Case Number: AFRL-2026-1349, CLEARED on 19 Mar 2026 

# Unmanned Aerial Vehicle (UAV)-Based Autonomous Firefighting: A Reinforcement Learning Environment


This project provides a 2D grid-world simulation for the AFRL 2026 Challenge Competition. In this environment, Unmanned Aerial Vehicles (UAVs) must navigate a map to find and extinguish fires while avoiding obstacles and dense smoke. The simulation logs performance metrics to a CSV file for analysis.

<div style="text-align: center;">
    <img src="afrl_challenge.gif", height=550>
</div>

## 1. Environment Setup

This project uses `conda` to manage its dependencies. You will need to have Anaconda or Miniconda installed to proceed.

### Steps

1.  **Clone the Repository:**
    If you haven't already, get the code on your local machine.
    ```
    git clone ...
    cd ...
    ```

2.  **Create the Conda Environment:**
    Use the provided `environment.yml` file to create a new conda environment with all the necessary libraries (like `numpy` and `pygame`). This command creates an environment named `afrl_challenge_env`.
    ```
    conda env create -f environment.yml
    ```

3.  **Activate the Environment:**
    Before running any scripts, you must activate the newly created environment. You will need to do this every time you open a new terminal for this project.
    ```
    conda activate afrl_challenge_env
    ```
4. **Run unit tests to confirm everything works properly**
    ```
    python -m unittest discover
    ```

## 2. How to Run the Simulation

The primary script for running the simulation is `main.py`. It accepts command-line arguments to control which agent to use, the number of episodes, and whether to display the graphical user interface.

### Command-Line Arguments

*   `--agent [logic|random]`: (Optional) Specifies which agent to use. Defaults to `logic`.
*   `--episodes <number>`: (Optional) Sets the number of episodes to run. Defaults to `10`.
*   `--render`: (Optional) A flag to enable the Pygame-based graphical rendering. If omitted, the simulation runs in headless mode.
*   `--fires_known`: (Optional Flag) If present, all fires will be visible on the map at the start of the episode. Defaults to False.
*   `--record <prefix> <episode interval>` (Optional) Records agent. Optional prefix name for the videos and optional episode interval to record an epsiode after an interval between, defaults to "eval" and 1.


### Example Usage

#### Running with the Logic Agent (Default)
To run the simulation with the `LogicAgent` and see the graphical output, use the `--render` flag.
```
python main.py --agent logic --render --episodes 5
```

To run the simulation with the 'LogicAgent', see the graphical output, and also have all fires visible on the map add the `--fires_known` flag.

```
python main.py --agent logic --render --episodes 100 --fires_known
```

## 3. How to view metrics and generate graphs.

Each simulation run generates metrics output to metrics/metrics.csv. You can use `utils/generate_graphs.py` to quickly create graphs and visualize the results.

```
python utils/generate_graphs.py
```
All graphs will be generated and placed within the metrics directory.

## 4. How to create your own first agent.

Insert detailed guideline on how to create and run your first agent.

## 5. How is an agent scored.

Insert detailed information about the scoring function.

## 6. How to submit your agent on the Kaggle platform.

Insert detailed information on Kaggle submissions.

## Authors

*   **Corryn Collins**
*   **David Shoukr**
*   **Alana Li**
*   **Elizabeth Andreas**
*   **Zane Kitchen-Lipski**
*   **Adis Delanovic**

Development of this environment was assisted by the Google Gemini large language model (LLM) on the genai.mil platform.

Distribution Statement A: Approved for Public Release; Distribution Unlimited: Case Number: AFRL-2026-1349, CLEARED on 19 Mar 2026 