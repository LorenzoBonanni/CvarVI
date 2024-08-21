import os
import random

import numpy as np
from matplotlib import pyplot as plt


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def seed_everything(seed=720):
    """
    Seeds for reproducibility of results

    Arguments:
        seed (int): Number of the seed
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def plot_trajectory(positions, env, values):
    plt.figure(figsize=(12, 10))
    plt.imshow(env.grid, cmap='binary', origin='lower')
    plt.imshow(values, cmap='jet', alpha=0.5, origin='lower')
    plt.plot(env.start_pos[1], env.start_pos[0], 'bs', markersize=10, label='Start')
    plt.plot(env.goal_pos[1], env.goal_pos[0], 'gs', markersize=10, label='Goal')
    y, x = zip(*positions)
    plt.plot(x, y, "o-", color="k", label="Trajectory")
    plt.legend()
    plt.colorbar()
    plt.title("Grid World Environment")
    plt.xlabel("Width")
    plt.ylabel("Height")

    # invert y axis to match the grid layout
    plt.gca().invert_yaxis()
    plt.savefig("trajectory.png", dpi=500, facecolor='white', edgecolor='none')
