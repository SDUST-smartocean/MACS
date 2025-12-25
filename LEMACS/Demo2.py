import numpy as np
import matplotlib.pyplot as plt
# 目标概率图初始化函数

def plot_probability_map(ranges_and_multipliers):
    total_range = (-5, 5)
    grid_size = 100
    x_edges = np.linspace(total_range[0], total_range[1], grid_size + 1)
    y_edges = np.linspace(total_range[0], total_range[1], grid_size + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    z = np.ones_like(x_grid) / (grid_size * grid_size)
    for params in ranges_and_multipliers:
        x_range = params['x_range']
        y_range = params['y_range']
        multiplier = params['multiplier']
        high_prob_area_mask = (x_grid > x_range[0]) & (x_grid < x_range[1])\
                              & (y_grid > y_range[0]) & (
                    y_grid < y_range[1])
        z[high_prob_area_mask] *= multiplier
    z /= np.sum(z)