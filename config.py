# config.py
import numpy as np
from scipy.stats import cauchy
import network
from time import time_ns

# ---------- physical parameters ----------
N = 10  # number of oscillators
K = 3  # default coupling
x0 = 50  # centre of the Cauchy distribution (mean frequency)
scale = 1.0  # Cauchy scale parameter gamma

# ---------- reproducibility ----------
rng = np.random.default_rng(time_ns())

# Natural frequencies drawn from Cauchy distribution
# FIXED: Removed the line that overrode these with constant frequencies
omega = cauchy.rvs(loc=x0, scale=scale, size=N, random_state=rng)

# Initial phases - slightly perturbed from uniform distribution
theta0 = np.linspace(0, 2 * np.pi, N, endpoint=False)
theta0[1] += 0.1

# ---------- integration settings ----------
t0 = 0.0
t_final = 10.0
dt = 0.001
t_eval = np.arange(t0, t_final + dt, dt)
t_delay = 0

# ---------- network choice ----------------
# choices: "full", "ring", "star", "random_er"
network_type = "full"
network_params = {"p": 0.3}  # only used if network_type=="random_er"


def get_adjacency(vary=False, Nloc=N):
    """
    Returns an N×N adjacency matrix based on network_type.
    If vary=True and the network is random_er, it will re-draw edges each call.
    """
    kind = network_type
    if kind == "full":
        return network.full(Nloc)
    if kind == "ring":
        return network.ring(Nloc)
    if kind == "star":
        return network.star(Nloc)
    if kind == "random_er":
        return network.random_er(
            Nloc, network_params.get("p", 0.5), rng if not vary else None
        )
    raise ValueError(f"Unknown network_type: {kind}")


def draw_omega(rng=None, size=N):
    """
    Return a fresh set of natural frequencies from the same Cauchy law.
    """
    if rng is None:
        rng = np.random.default_rng()
    return cauchy.rvs(loc=x0, scale=scale, size=size, random_state=rng)
