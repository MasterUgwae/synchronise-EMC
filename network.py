"""
Define adjacency-matrix factories for different network topologies.
Each returns an N×N array A where A[i,j]=1 if nodes i and j are coupled.
"""

import numpy as np

def full(N: int) -> np.ndarray:
    """All-to-all (no self-loops)."""
    A = np.ones((N, N), dtype=float)
    np.fill_diagonal(A, 0.0)
    return A

def ring(N: int) -> np.ndarray:
    """Each node i connected to i±1 mod N."""
    A = np.zeros((N, N), dtype=float)
    for i in range(N):
        A[i, (i - 1) % N] = 1.0
        A[i, (i + 1) % N] = 1.0
    return A

def star(N: int) -> np.ndarray:
    """Node 0 is the hub, connected to all others."""
    A = np.zeros((N, N), dtype=float)
    for i in range(1, N):
        A[0, i] = 1.0
        A[i, 0] = 1.0
    return A

def random_er(N: int, p: float, rng=None) -> np.ndarray:
    """
    Erdős–Rényi: each edge included with prob p, then symmetrized.
    Excludes self-loops.
    """
    if rng is None:
        rng = np.random.default_rng()
    # upper triangle random draw
    U = rng.random((N, N))
    A = np.triu((U < p).astype(float), k=1)
    # make symmetric
    A = A + A.T
    return A
