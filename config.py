# config.py

import numpy as np
from scipy.stats import cauchy

# Number of oscillators
N = 100

# Coupling strength
<<<<<<< HEAD
K = 7.0
=======
K = 10.0
>>>>>>> d02ad67c7b9cd6f4071d667646e175fde812e6d7

# Baseline natural frequency (the center of the distribution)
x0 = 0.5

# Set a fixed random seed for reproducibility
# seed = 42
# np.random.seed(seed)

# Draw natural frequencies from a Cauchy distribution.
# Even though the frequencies are drawn from a distribution,
# using the fixed seed ensures consistent (non-random) parameters over trials.
# Then make sure that the frequencies are not negative
omega = cauchy.rvs(loc=x0, scale=1, size=N)
omega = np.max(omega, 0)

# Fixed initial phases – here we use equally spaced phases in [0, 2π)
theta0 = np.linspace(0, 2 * np.pi, N, endpoint=False)

# Time parameters for the simulation
t0 = 0            # start time
t_final = 100     # end time
dt = 0.01         # time step for our iterative simulation

# Create an array of time values to evaluate the simulation
t_eval = np.arange(t0, t_final + dt, dt)
