import random

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


class GridWorldEnv:
    """
    A class representing a grid world environment for reinforcement learning.

    This environment consists of a 2D grid where an agent can move up, down, left, or right.
    The grid contains obstacles, a start position, and a goal position. The agent receives
    a reward of -1 for each step taken. If the agent hits an obstacle, it receives a reward
    of -40 and the episode ends. The agent cannot move outside the boundaries of the map.

    Attributes:
        width (int): The width of the grid.
        height (int): The height of the grid.
        delta (float): The probability of a random action being taken instead of the chosen action.
        num_obstacles (int): The number of obstacles in the grid.
        grid (numpy.ndarray): The 2D array representing the grid world.
        start_pos (Tuple[int, int]): The starting position of the agent (width, height).
        goal_pos (Tuple[int, int]): The goal position (width, height).
        current_pos (Tuple[int, int]): The current position of the agent (width, height).
        actions (List[str]): The list of possible actions.
    """

    def __init__(self, width: int = 64, height: int = 53, delta: float = 0.1, num_obstacles: int = 80):
        """
        Initialize the GridWorldEnv.

        Args:
            width (int): The width of the grid. Defaults to 53.
            height (int): The height of the grid. Defaults to 64.
            delta (float): The probability of a random action. Defaults to 0.1.
            num_obstacles (int): The number of obstacles to place. Defaults to 80.
        """
        self.seed()
        self.width = width
        self.height = height
        self.delta = delta
        self.num_obstacles = num_obstacles

        self.grid = np.zeros((width, height))
        self.start_pos = (60, 50)  # (width, height)
        self.goal_pos = (60, 2)  # (width, height)
        self.current_pos = self.start_pos

        self._place_obstacles()
        self.grid[self.goal_pos] = 2  # Mark goal

        self.actions = ["up", "down", "left", "right"]

    def seed(self, seed=720):
        np.random.seed(seed)
        random.seed(seed)

    def _place_obstacles(self):
        """
        Place obstacles randomly in the grid.

        This method ensures that obstacles are not placed at the start or goal positions.
        """
        obstacles = 0
        while obstacles < self.num_obstacles:
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            if (x, y) != self.start_pos and (x, y) != self.goal_pos and self.grid[x, y] == 0:
                self.grid[x, y] = 1  # Mark obstacle
                obstacles += 1

    def reset(self) -> Tuple[int, int]:
        """
        Reset the environment to its initial state.

        Returns:
            Tuple[int, int]: The initial position of the agent (width, height).
        """
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, state, action: str) -> Tuple[Tuple[int, int], float, bool]:
        """
        Take a step in the environment based on the given action.

        Args:
            action (str): The action to take. Must be one of "up", "down", "left", or "right".

        Returns:
            Tuple[Tuple[int, int], float, bool]: A tuple containing:
                - The new position of the agent (width, height).
                - The reward received for this step (-1 for normal step, -40 for hitting an obstacle).
                - A boolean indicating whether the episode is done (True if goal is reached or obstacle is hit).
        """

        next_states, probs = self.get_next_states(state, action)

        next_state = random.choices(population=next_states, weights=probs)[0]
        reward = self.get_reward(next_state)
        self.current_pos = next_state

        done = next_state == self.goal_pos or self.grid[next_state[0], next_state[1]] == 1

        return self.current_pos, reward, done

    def get_next_states(self, state: Tuple[int, int], action: str) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Get the possible next states and their probabilities for a given state and action.

        Args:
            state (Tuple[int, int]): The current state (width, height).
            action (str): The action to take. Must be one of "up", "down", "left", or "right".

        Returns:
            Tuple[List[Tuple[int, int]], List[float]]: A tuple containing:
                - A list of possible next states (width, height).
                - A list of probabilities corresponding to each next state.
        """
        if self.grid[state[0], state[1]] == 1 or state == self.goal_pos:
            return [], None
        prob_action = 1 - self.delta
        prob_other_actions = self.delta / 3
        prob_index = self.actions.index(action)
        x, y = state
        probabilities = [prob_other_actions] * 4
        probabilities[prob_index] = prob_action

        pos_up = (x, min(self.height - 1, y + 1))
        pos_down = (x, max(0, y - 1))
        pos_left = (max(0, x - 1), y)
        pos_right = (min(self.width - 1, x + 1), y)
        positions = [pos_up, pos_down, pos_left, pos_right]

        return positions, probabilities

    def render(self, action: str = None):
        """
        Visualize the current state of the grid world.

        This method uses matplotlib to create a visual representation of the grid,
        showing obstacles, the start position, the goal position, and the current
        position of the agent.
        """
        plt.clf()
        plt.imshow(self.grid.T, cmap='binary', origin='lower')
        plt.plot(self.start_pos[0], self.start_pos[1], 'bs', markersize=10, label='Start')
        plt.plot(self.goal_pos[0], self.goal_pos[1], 'gs', markersize=10, label='Goal')
        plt.plot(self.current_pos[0], self.current_pos[1], 'rs', markersize=8, label='Agent')
        plt.legend()
        if action:
            plt.title(f"Grid World Environment - Action: {action}")
        else:
            plt.title("Grid World Environment")
        plt.xlabel("Width")
        plt.ylabel("Height")
        # invert y axis to match the grid layout
        plt.gca().invert_yaxis()
        plt.pause(0.5)

    def get_reward(self, next_state: Tuple[int, int]) -> float:
        x, y = next_state
        if self.grid[x, y] == 1:  # Hit an obstacle
            return -40

        done = next_state == self.goal_pos

        return 0 if done else -1

# # Example usage:
# env = GridWorldEnv()
# env.visualize()
#
# # Simulate a few steps
# for _ in range(5):
#     action = np.random.choice(env.actions)
#
#     if done:
#         print("Episode ended.")
#         break
#
# env.visualize()
