import numpy as np
from matplotlib import pyplot as plt

from gridworld import GridWorldEnv
from utils import seed_everything, plot_trajectory

plt.ion()


class ValueIteration:
    def __init__(self, env: GridWorldEnv, gamma=0.95, theta=1e-5):
        self.max_iter = 1e4
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros((env.height, env.width))

    def get_action_values(self, state):
        action_values = {}
        for a in self.env.actions:
            value = 0
            next_states, probs = self.env.get_next_states(state, a)
            # if next_states is empty it means state is a terminal state
            if not next_states:
                value = self.env.get_reward(state)
            else:
                for i, next_state in enumerate(next_states):
                    value += probs[i] * (self.env.get_reward(next_state) + self.gamma * self.V[next_state])

            action_values[a] = value
        return list(action_values.values())

    def value_iteration(self):
        delta = float("inf")
        while delta > self.theta and self.max_iter > 0:
            delta = 0
            for width in range(self.env.width):
                for height in range(self.env.height):
                    current_v = self.V[height, width]
                    action_values = self.get_action_values((height, width))
                    self.V[height, width] = max(action_values)
                    delta = max(delta, abs(current_v - self.V[height, width]))
            self.max_iter -= 1

    def policy(self):
        policy = np.zeros((self.env.height, self.env.width), dtype=int)
        for width in range(self.env.width):
            for height in range(self.env.height):
                action_values = self.get_action_values((height, width))
                policy[height, width] = np.argmax(action_values)

        string_policy = np.vectorize(lambda x: self.env.actions[x])(policy)
        return string_policy


def run_episode(env, policy):
    done = False
    positions = []
    start_pos = env.reset()
    current_pos = start_pos
    positions.append(current_pos)
    env.render()
    while not done:
        action = policy[current_pos]
        next_state, reward, done = env.step(current_pos, action)
        env.render()
        positions.append(next_state)
        current_pos = next_state
    return positions


def main():
    seed_everything()
    env_path = 'gridworld4.png'
    if env_path == 'gridworld3.png':
        goal_pos = (1, 15)
        start_pos = (12, 15)
    else:
        goal_pos = (1, 59)
        start_pos = (49, 59)

    vi = ValueIteration(GridWorldEnv(env_path, goal_pos=goal_pos, start_pos=start_pos))
    vi.value_iteration()
    env = GridWorldEnv(env_path, goal_pos=goal_pos, start_pos=start_pos)
    plt.figure(figsize=(12, 10))
    positions = run_episode(env, vi.policy())
    plot_trajectory(positions, env, values=vi.V)


main()
