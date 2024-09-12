from copy import deepcopy

import numpy as np
from scipy import stats
from scipy.optimize import linprog
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt


def plot_distribution(z, p):
    plt.figure()
    plt.plot(z, p, label='Distribution')
    # plt.axvline(cvar, color='r', linestyle='--', label='CVaR')
    plt.xlabel('z')
    plt.ylabel('p')
    plt.legend()
    plt.show()


def calculate_cvar_lp(alpha, z, num_points):
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # Variables
    var = solver.NumVar(-solver.infinity(), solver.infinity(), 'var')
    y = [solver.NumVar(0, solver.infinity(), f'y_{i}') for i in range(num_points)]

    # Objective: minimize var + (1 / (1-alpha)) * (1/n) * sum(y)
    obj = var + (1 / (1 - alpha)) * (1 / num_points) * solver.Sum(y)
    solver.Minimize(obj)

    # Constraints
    for i in range(num_points):
        solver.Add(y[i] >= -z[i] - var)  # Note the negation of z[i]

    # Solve
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        cvar = -solver.Objective().Value()  # Negate the result
        print(f'Solved in {solver.wall_time():.2f} milliseconds in {solver.iterations()} iterations')
        return cvar
    else:
        print('The solver could not find an optimal solution.')
        return None

def gt_cvar(z, alpha):
    copy_z = deepcopy(z)
    copy_z = np.sort(copy_z)
    num_samples = len(copy_z)
    cvar_index = int((1 - alpha) * num_samples)
    CVaR = np.mean(copy_z[:cvar_index])
    return CVaR


np.random.seed(42)
# Example usage
alpha = 0.95  # Confidence level
mu = 0  # Mean of the Gaussian distribution
sigma = 1  # Standard deviation of the Gaussian distribution
num_points = 10_000  # Number of points to discretize the distribution
z = np.random.normal(mu, sigma, num_points)

cvar = calculate_cvar_lp(alpha, z, num_points)
# cvar_xi = calculate_cvar_xi(z, alpha, mu, sigma)
gt = gt_cvar(z, alpha)
print(f"CVaR at {alpha * 100}% confidence level: {cvar}")
print(f"Ground truth CVaR at {alpha * 100}% confidence level: {gt}")
