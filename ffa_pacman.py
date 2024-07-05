import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gym.spaces import Discrete, Box
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gym

class FFAPacman(ParallelEnv):
    metadata = {'render_modes': ['human'], 'is_parallelizable': True}

    def __init__(self, grid_size, num_agents, num_balls):
        self._grid_size = grid_size
        self._num_agents = num_agents
        self._num_balls = num_balls

        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.pos = {agent: None for agent in self.agents}
        self.strength = {agent: 1 for agent in self.agents}
        self.points = {agent: 0 for agent in self.agents}
        self.kills = {agent: 0 for agent in self.agents}
        self.active_agents = set(self.agents)
        self.timesteps = 0

        self.action_spaces = {agent: Discrete(4) for agent in self.agents}
        self.observation_spaces = {agent: Box(low=0, high=3, shape=(grid_size, grid_size), dtype=np.int32) for agent in self.agents}

    def reset(self, **kwargs):
        self.timesteps = 0 
        self.grid = np.zeros((self._grid_size, self._grid_size), dtype=np.int32)
        self.balls = []

        for _ in range(self._num_balls):
            x, y = np.random.randint(0, self._grid_size, size=2)
            while self.grid[x, y] != 0:
                x, y = np.random.randint(0, self._grid_size, size=2)
            self.grid[x, y] = 1 # Energy ball
            self.balls.append((x, y))

        for agent in self.agents:
            x, y = np.random.randint(0, self._grid_size, size=2)
            while self.grid[x, y] != 0:
                x, y = np.random.randint(0, self._grid_size, size=2)
            self.grid[x, y] = 2 + self.agents.index(agent)  # Agents represented by 2, 3, 4, ...
            self.pos[agent] = (x, y)
            self.strength[agent] = 1  # Reset strength
            self.points[agent] = 0  # Reset points
            self.kills[agent] = 0  # Reset kills
            self.active_agents.add(agent)  # Ensure all agents are active at the start

        observations = {agent: self.grid.copy() for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos  # Returning only the observations, infos is empty (for debugging)

    def step(self, actions):
        self.timesteps += 1
        rewards = {agent: 0 for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        new_positions = {}
        for agent, action in actions.items():
            if agent not in self.active_agents:
                continue
            x, y = self.pos[agent]
            if action == 0: x = (x - 1) % self._grid_size  # Move left with wrapping
            elif action == 1: x = (x + 1) % self._grid_size  # Move right with wrapping
            elif action == 2: y = (y - 1) % self._grid_size  # Move up with wrapping
            elif action == 3: y = (y + 1) % self._grid_size
            new_positions[agent] = (x, y)

        interaction_zones = {}
        for agent, new_pos in new_positions.items():
            x, y = new_pos
            if self.grid[x, y] == 1:
                self.strength[agent] += 1
                self.grid[x, y] = 0
                self.balls.remove((x, y))
                rewards[agent] += 1
                self.points[agent] += 1

            if new_pos not in interaction_zones:
                interaction_zones[new_pos] = []
            interaction_zones[new_pos].append(agent)

        for pos, agents in interaction_zones.items():
            if len(agents) > 1:
                max_strength = max(self.strength[agent] for agent in agents)
                strongest_agents = [agent for agent in agents if self.strength[agent] == max_strength]
                if len(strongest_agents) == 1:
                    survivor = strongest_agents[0]
                    for agent in agents:
                        if agent != survivor:
                            dones[agent] = True
                            rewards[survivor] += 10
                            self.points[survivor] += 10
                            self.kills[survivor] += 1
                            rewards[agent] -= 10
                            self.points[agent] -= 10
                            self.grid[self.pos[agent]] = 0
                            self.pos[agent] = None
                            self.active_agents.remove(agent)
                else:
                    for agent in agents:
                        dones[agent] = True
                        self.points[agent] -= 5
                        self.grid[self.pos[agent]] = 0
                        self.pos[agent] = None
                        self.active_agents.remove(agent)

        for agent in self.agents:
            if not dones[agent] and agent in new_positions:
                self.grid[self.pos[agent]] = 0
                self.pos[agent] = new_positions[agent]
                x, y = self.pos[agent]
                self.grid[x, y] = 2 + self.agents.index(agent)

        if len(self.active_agents) == 1:
            remaining_agent = next(iter(self.active_agents))
            rewards[remaining_agent] += 100  # Reward the last standing agent
            dones.update({agent: True for agent in self.agents})

        observations = {agent: self.grid.copy() for agent in self.agents}
        return observations, rewards, dones, infos

    def render(self, mode='human'):
        print("Grid:")
        for row in self.grid:
            row_str = []
            for cell in row:
                if cell == 1:
                    row_str.append("*") # Energy balls
                elif cell == 0:
                    row_str.append("#") # Empty cells
                elif 2 <= cell < 2 + self._num_agents:
                    row_str.append(str(cell - 1)) # Agents
                else:
                    row_str.append(" ")
            print(" ".join(row_str))
        print(" ")
        for i, agent in enumerate(self.agents):
            status = "alive" if agent in self.active_agents else "dead"
            print(f"Agent {i + 1}: Strength: {self.strength[agent]}, Points: {self.points[agent]}, Kills: {self.kills[agent]}, Status: {status}")
        print(" ")
        print(f"Timestep: {self.timesteps}")
        print("="*40)

class PettingZooWrapper(gym.Env):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.reset()
        self.action_space = self.env.action_spaces[agent]
        self.observation_space = self.env.observation_spaces[agent]

    def reset(self):
        observations, _ = self.env.reset()
        return observations[self.agent]

    def step(self, action):
        actions = {self.agent: action}
        obs, rewards, dones, infos = self.env.step(actions)
        return obs[self.agent], rewards[self.agent], dones[self.agent], infos[self.agent]

    def render(self, mode='human'):
        self.env.render(mode)

    def close(self):
        self.env.close()

env = FFAPacman(grid_size=10, num_agents=10, num_balls=90)

# Random Walk

if __name__ == "__main__":
    env.reset()
    env.render()
    done = {agent: False for agent in env.agents}
    time = 0
    max_time = 5000
    while len(env.active_agents) > 1 and time<max_time:
        actions = {agent: np.random.randint(0, 4) for agent in env.agents if not done[agent]}
        observations, rewards, dones, infos = env.step(actions)
        time += 1
        env.render()

        done.update(dones)

# Training

for agent in env.agents:
    env.reset()
    single_agent_env = PettingZooWrapper(env, agent)
    single_agent_env = Monitor(single_agent_env)

models = {}
num_games = 50  # Number of games to train
timesteps_per_game = 10000 # Number of timesteps per game

for agent in env.agents:
    single_agent_env = DummyVecEnv([lambda: PettingZooWrapper(env, agent)])
    model = DQN("MlpPolicy", single_agent_env, verbose=1, buffer_size=100000)

    for game in range(num_games):
        print(f"Training {agent} in game {game + 1}/{num_games}")
        model.learn(total_timesteps=timesteps_per_game)  

    model.save(f"dqn_{agent}")
    models[agent] = model

# Simulating Game

def simulate_game(env, models):
    observations, _ = env.reset()  
    dones = {agent: False for agent in env.agents}
    total_rewards = {agent: 0 for agent in env.agents}
    max_timesteps = 5000  

    timestep = 0
    while not all(dones.values()) and timestep < max_timesteps:
        actions = {}
        for agent in env.agents:
            if not dones[agent]:
                obs = observations[agent]
                action, _states = models[agent].predict(obs)
                actions[agent] = action

        observations, rewards, dones, infos = env.step(actions)
        for agent, reward in rewards.items():
            total_rewards[agent] += reward

        timestep += 1
        env.render()

    return total_rewards

models = {}
for agent in env.agents:
    models[agent] = DQN.load(f"dqn_{agent}")

env.reset()
env.render()
simulation_results = []
print(f"Starting game")
total_rewards = simulate_game(env, models)
simulation_results.append(total_rewards)
print(f"Game results: {total_rewards}")

env.close()
