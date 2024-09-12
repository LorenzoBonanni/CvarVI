import random

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

from utils import rgb2gray


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
        grid (np.ndarray): A 2D numpy array representing the grid world.
        start_pos (Tuple[int, int]): The starting position of the agent (height, width).
        goal_pos (Tuple[int, int]): The goal position (height, width).
        current_pos (Tuple[int, int]): The current position of the agent (height, width).
        actions (List[str]): A list of possible actions the agent can take ("up", "down", "left", "right").
    """

    def __init__(self, path: str, delta: float = 0.05, goal_pos: Tuple[int, int] = (1, 15),
                 start_pos: Tuple[int, int] = (12, 15)):
        """
        Initialize the grid world environment.

        Args:
            path (str): The path to an image file representing the grid world.
            delta (float): The probability of a random action being taken instead of the chosen action.
            goal_pos (Tuple[int, int]): The goal position (height, width).
            start_pos (Tuple[int, int]): The starting position of the agent (height, width).
        """
        im = plt.imread(path)
        im = rgb2gray(im)
        height, width = im.shape

        self.width = width
        self.height = height
        self.delta = delta

        self.grid = np.zeros((height, width))
        self.grid[np.where(im == 0)] = 1  # Mark obstacles
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.current_pos = self.start_pos

        self.actions = ["up", "down", "left", "right"]


    def reset(self) -> Tuple[int, int]:
        """
        Reset the environment to its initial state.

        Returns:
            Tuple[int, int]: The initial position of the agent (width, height).
        """
        self.current_pos = self.start_pos
        return self.current_pos

    def is_terminal(self, state):
        return state == self.goal_pos or self.grid[state[0], state[1]] == 1

    def step(self, state, action: str) -> Tuple[Tuple[int, int], float, bool]:
        """
        Take a step in the environment based on the given action.

        Args:
            state (Tuple[int, int]): The current state (width, height).
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

        done = self.is_terminal(next_state)

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
        y, x = state

        pos_up = (min(self.height - 1, y + 1), x)
        pos_down = (max(0, y - 1), x)
        pos_left = (y, max(0, x - 1))
        pos_right = (y, min(self.width - 1, x + 1))

        if action == "up":
            positions = [pos_up, pos_left, pos_right]
        elif action == "down":
            positions = [pos_down, pos_left, pos_right]
        elif action == "left":
            positions = [pos_left, pos_up, pos_down]
        else:
            positions = [pos_right, pos_up, pos_down]

        probabilities = [prob_action, prob_other_actions, prob_other_actions]

        return positions, probabilities

    def render(self, action: str = None):
        """
        Visualize the current state of the grid world.

        This method uses matplotlib to create a visual representation of the grid,
        showing obstacles, the start position, the goal position, and the current
        position of the agent.
        """
        plt.clf()
        plt.imshow(self.grid, cmap='binary', origin='lower')
        plt.plot(self.start_pos[1], self.start_pos[0], 'bs', markersize=10, label='Start')
        plt.plot(self.goal_pos[1], self.goal_pos[0], 'gs', markersize=10, label='Goal')
        plt.plot(self.current_pos[1], self.current_pos[0], 'rs', markersize=8, label='Agent')
        plt.legend()
        if action:
            plt.title(f"Grid World Environment - Action: {action}")
        else:
            plt.title("Grid World Environment")
        plt.xlabel("Width")
        plt.ylabel("Height")
        # invert y-axis to match the grid layout
        plt.gca().invert_yaxis()
        plt.pause(0.5)

    def get_reward(self, next_state: Tuple[int, int]) -> float:
        y, x = next_state
        if self.grid[y, x] == 1:  # Hit an obstacle
            return -40

        done = next_state == self.goal_pos

        return 0 if done else -1
