import numpy as np
from matplotlib import pyplot as plt

from gridworld import GridWorldEnv
plt.ion()


class ValueIteration:
    def __init__(self, env: GridWorldEnv, gamma=0.95, theta=1e-5):
        self.max_iter = 1e3
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros((env.width, env.height))

    def get_action_values(self, state):
        action_values = {}
        for a in self.env.actions:
            value = 0
            next_states, probs = self.env.get_next_states(state, a)
            # if next_states is empty it means state is a terminal state)
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
                    current_v = self.V[width, height]
                    action_values = self.get_action_values((width, height))
                    self.V[width, height] = max(action_values)
                    delta = max(delta, abs(current_v - self.V[width, height]))
            self.max_iter -= 1

    def policy(self):
        policy = np.zeros((self.env.width, self.env.height), dtype=int)
        for width in range(self.env.width):
            for height in range(self.env.height):
                action_values = self.get_action_values((width, height))
                policy[width, height] = np.argmax(action_values)

        string_policy = np.vectorize(lambda x: self.env.actions[x])(policy)
        return string_policy


def plot_trajectory(positions, env, values):
    plt.figure(figsize=(12, 10))
    plt.imshow(env.grid.T, cmap='binary', origin='lower')
    plt.imshow(values.T, cmap='jet', alpha=0.5, origin='lower')
    plt.plot(env.start_pos[0], env.start_pos[1], 'bs', markersize=10, label='Start')
    plt.plot(env.goal_pos[0], env.goal_pos[1], 'gs', markersize=10, label='Goal')
    x, y = zip(*positions)
    plt.plot(x, y, "o-", color="k", label="Trajectory")
    plt.legend()
    plt.colorbar()
    plt.title("Grid World Environment")
    plt.xlabel("Width")
    plt.ylabel("Height")

    # invert y axis to match the grid layout
    plt.gca().invert_yaxis()
    plt.savefig("trajectory.png", dpi=500, facecolor='white', edgecolor='none')


def main():
    vi = ValueIteration(GridWorldEnv())
    vi.value_iteration()
    env = GridWorldEnv()
    done = False
    positions = []
    start_pos = env.reset()
    policy = vi.policy()
    print(vi.V[1, 5])
    print(vi.V[env.goal_pos])
    current_pos = start_pos
    plt.figure(figsize=(12, 10))
    env.render()
    while not done:
        positions.append(current_pos)
        action = policy[current_pos]
        new_pos, reward, done = env.step(current_pos, action)
        env.render(action=action)
        print(f"Action: {action}, New position: {new_pos}, Reward: {reward}, Done: {done}")
        current_pos = new_pos
    positions.append(current_pos)
    plot_trajectory(positions, env, values=vi.V)


main()
