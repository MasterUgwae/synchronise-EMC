# config.py
import numpy as np
from scipy.stats import cauchy
import network

# ---------- physical parameters ----------
N          = 14          # number of oscillators
K          = 0.2         # default coupling
x0         = 5          # centre of the Cauchy distribution (mean frequency)
scale      = 1.0         # Cauchy scale parameter gamma

# ---------- reproducibility ----------
seed       = 43
rng        = np.random.default_rng(seed)   # NumPy 1.17+ generator

# deterministic "random" frequencies and phases
omega      = cauchy.rvs(loc=x0, scale=scale, size=N, random_state=rng)
omega      = np.ndarray([1]*N)
theta0     = np.linspace(0, 2*np.pi, N, endpoint=False)  # evenly spaced
theta0[1] += 0.1  # perturb the second oscillator
# ---------- integration settings ----------
t0         = 0.0
t_final    = 100.0
dt         = 0.01
t_eval     = np.arange(t0, t_final + dt, dt)             # uniform grid

# ---------- network choice ----------------
# choices: "full", "ring", "star", "random_er"
network_type   = "star"
network_params = {"p": 0.3}   # only used if network_type=="random_er"

def get_adjacency(vary=False,Nloc = N):
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
        return network.random_er(Nloc, network_params.get("p", 0.5),
                                 rng if not vary else None)
    raise ValueError(f"Unknown network_type: {kind}")

def draw_omega(rng=None):
    """
    Return a fresh set of natural frequencies from the same Cauchy law.
    """
    rng = np.random.default_rng() if rng is None else rng
    return cauchy.rvs(loc=x0, scale=scale, size=N, random_state=rng)
