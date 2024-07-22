# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:03:52 2024

@author: vaishu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 20:49:27 2024

@author: vaishu
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 2 * np.pi  # Domain size
N = 100  # Number of grid points
dx = L / N  # Grid spacing

# Spatial grid
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

# Velocity field
def u(x, y):
    return np.sin(x) * np.cos(y)

def v(x, y):
    return -np.cos(x) * np.sin(y)

# Diffusivity
D_values = [0.1, 0.01, 0.001]

# Initial concentration
def initial_concentration(x, y, x0, y0):
    return np.exp(-1*((x - x0)**2 + (y - y0)**2) / (2 * 0.2*2))

# Advection-diffusion equation solver
def solve_advection_diffusion(u, v, D, dt, T, X, Y):
    nt = int(T / dt)  # Number of time steps
    C = initial_concentration(X, Y, np.pi/2, np.pi/2)  # Initial concentration
    for t in range(nt):
        # Advection
        C_x = u(X, Y) * np.gradient(C, axis=1)  # Partial derivative in x
        C_y = v(X, Y) * np.gradient(C, axis=0)  # Partial derivative in y
        adv = -(C_x + C_y)

        # Diffusion
        laplacian = (np.roll(C, -1, axis=1) + np.roll(C, 1, axis=1) +
                     np.roll(C, -1, axis=0) + np.roll(C, 1, axis=0) - 4*C) / dx**2
        diff = D * laplacian

        # Update concentration
        C += dt * (diff + adv)

    return C

# Time parameters
dt = 0.001
times = [1, 10]

# Plotting
for D in D_values:
    for t in times:
        concentration = solve_advection_diffusion(u, v, D, dt, t, X, Y)
        plt.figure()
        plt.imshow(concentration, extent=(0, L, 0, L), origin='lower', cmap='viridis')
        plt.colorbar(label='Concentration')
        plt.title(f'Concentration Field at t = {t}, D = {D}')
        plt.xlabel('x')
        plt.ylabel('y')

plt.show()  # Display all figures together