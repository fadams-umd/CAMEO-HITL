import numpy as np
from scipy.stats import mannwhitneyu, norm
import matplotlib.pyplot as plt

folder = './quantitative_results/'

boundary_control_scores = np.loadtxt(folder + 'boundary_control_scores.txt')
boundary_hitl_scores = np.loadtxt(folder + 'boundary_hitl_scores.txt')
region_control_scores = np.loadtxt(folder + 'region_control_scores.txt')
region_hitl_scores = np.loadtxt(folder + 'region_hitl_scores.txt')

def mu(n):
    return(n**2 / 2)

def sigma(n):
    return(np.sqrt(n**2 * (2 * n + 1) / 12))

def power(alt, n):
    H0 = norm(loc=mu(n), scale=sigma(n))
    critical_value = H0.ppf(0.95)
    return(1 - norm(loc=alt, scale=sigma(n)).cdf(critical_value))

n_values = np.concatenate((np.linspace(10, 100, 10),
                           np.linspace(100, 1000, 10)))

d_values = np.linspace(5, 50, 10) / 100

n_grid, d_grid = np.meshgrid(n_values, d_values)

I, J = n_grid.shape

power_grid = np.zeros_like(n_grid)

for i in range(I):
    for j in range(J):
        power_grid[i, j] = power((1 + d_grid[i, j]) * mu(n_grid[i, j]),
                                 n_grid[i, j])

fig, ax = plt.subplots()

cs = ax.contour(n_grid, d_grid, power_grid)

plt.xscale('log')

plt.clabel(cs)

ax.set_xlabel('number of samples')
ax.set_ylabel('fraction difference in U statistic')
ax.set_title('statistical power, p < 0.05')

plt.show()