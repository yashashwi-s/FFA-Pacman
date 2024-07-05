# Free-for-All Pacman Environment

This project implements a custom multi-agent Pacman environment using PettingZoo. Agents compete to collect energy balls and eliminate each other, with the last agent standing declared the winner. Agents are trained using the DQN algorithm from Stable Baselines3.

## Table of Contents

1. [Environment Overview](#environment-overview)
2. [Environment Components](#environment-components)
3. [Environment Class](#environment-class)
4. [PettingZoo Wrapper Class](#pettingzoo-wrapper-class)
5. [Training Agents](#training-agents)
6. [Simulation](#simulation)
7. [Edge Cases and Interactions](#edge-cases-and-interactions)
8. [Running the Project Locally](#running-the-project-locally)

## Environment Overview

### Grid and Entities

- **Grid Size**: The environment is a 2D grid of size `grid_size x grid_size`.
- **Agents**: Each agent is represented by a unique identifier and starts at a random position on the grid.
- **Energy Balls**: Energy balls appear randomly on the grid and increase the strength of the agent that collects them.

## Environment Components

### States

The state of the environment is represented by a 2D grid. Each cell in the grid can be:

- An empty cell (`0`).
- An energy ball (`1`).
- An agent, with each agent represented by a unique integer starting from `2`.

### Actions

Agents can move in four directions:

- `0`: Move left
- `1`: Move right
- `2`: Move up
- `3`: Move down

### Rewards

Agents receive rewards based on their interactions:

- Collecting an energy ball gives a reward of `+1`.
- Eliminating another agent gives a reward of `+10` to the attacker and a penalty of `-10` to the eliminated agent.
- The last standing agent receives a reward of `+100`.

### Done Condition

The game ends when only one agent remains active or the maximum number of timesteps is reached.

## Environment Class

### Initialization

The environment is initialized with a specified grid size, number of agents, and number of energy balls. Agents and energy balls are placed randomly on the grid.

### Reset

The `reset` method resets the environment to its initial state, placing agents and energy balls randomly on the grid. It also resets agent attributes like strength, points, and kills.

### Step

The `step` method executes the given actions for each agent and updates the environment state. It handles agent movements, interactions, and updates rewards. If multiple agents move to the same cell, the agent with the highest strength survives.

### Render

The `render` method displays the current state of the environment, including the grid and the status of each agent.

## PettingZoo Wrapper Class

### Initialization

The PettingZoo wrapper converts the multi-agent environment into a single-agent environment compatible with Gym. This allows training each agent individually using Gym-compatible algorithms.

### Reset

The `reset` method resets the environment and returns the observation for the specified agent.

### Step

The `step` method executes the action for the specified agent and returns the result, including the new observation, reward, done status, and additional info.

### Render

The `render` method displays the environment.

### Close

The `close` method closes the environment.

## Training Agents

### Training Process

Agents are trained using the DQN algorithm. Each agent is trained individually in the environment. The training involves multiple games, with each game consisting of a specified number of timesteps.

### Training Steps

1. Initialize the environment and the DQN model for each agent.
2. Train each agent individually by resetting the environment and learning from the observations, rewards, and actions.
3. Save the trained models for each agent.

## Simulation

### Simulation Process

The trained models are used to simulate games, where agents interact based on their learned policies. The simulation runs for a specified number of timesteps or until only one agent remains active.

### Simulation Steps

1. Reset the environment and initialize the done status for each agent.
2. For each timestep, predict actions for each agent using their trained models.
3. Execute the actions and update the environment state.
4. Render the environment and update the rewards and done status for each agent.

## Edge Cases and Interactions

### Agent Interactions

- **Collision Handling**: When multiple agents move to the same cell, the agent with the highest strength survives.
- **Strength Tie**: If there is a tie in strength, all involved agents are eliminated.

### Wrappers

- **PettingZooWrapper**: Converts the multi-agent environment into a single-agent environment for training with Gym-compatible algorithms.

## Running the Project Locally

### Prerequisites

Ensure you have Python installed on your system. Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Running Random Walk

To run a random walk simulation:

1. **Initialize the Environment:**
   The environment is created with a specified grid size, number of agents, and number of energy balls.

2. **Reset the Environment:**
   The environment is reset, initializing the grid, placing agents and energy balls randomly, and setting up initial values for agents' strength, points, and positions.

3. **Execute Random Actions:**
   Each agent selects a random action (move left, right, up, or down). These actions are executed, and the environment is updated accordingly:

   - Agents collect energy balls to increase their strength.
   - Agents may engage in interactions if they move to the same cell.
   - Interactions are resolved based on agents' strength, leading to potential eliminations.

4. **Render the Environment:**
   After each step, the environment's current state is rendered to the console, showing the grid, agents' statuses, and the current timestep. The process continues until only one agent remains or the maximum number of timesteps is reached.

### Training Agents

To train agents (this will take a lot of time, so I'll suggest running on google colab):

1. **Initialize the Environment and DQN Model:**
   For each agent, a `PettingZooWrapper` is created to adapt the environment for single-agent training. A DQN model is initialized with the environment.

2. **Training Process:**
   Each agent is trained individually for a specified number of games and timesteps per game. The agent learns by interacting with the environment, receiving rewards, and updating its policy based on the observations.

3. **Save Trained Models:**
   After training, the models for each agent are saved to disk for later use in simulations.

### Simulating a Game with Trained Agents

To simulate a game with trained agents:

1. **Load Trained Models:**
   The saved models for each agent are loaded from disk.

2. **Reset the Environment:**
   The environment is reset to start a new game.

3. **Simulate the Game:**
   For each timestep:

   - Each agent uses its trained model to predict the best action based on the current observation.
   - The predicted actions are executed, and the environment is updated.
   - The rewards and statuses of agents are tracked.

4. **Render the Environment:**
   The environment is rendered at each step to visualize the game progress. The simulation continues until all agents are done or the maximum number of timesteps is reached.

5. **Record Results:**
   The total rewards for each agent are recorded and displayed at the end of the simulation.
