import numpy as np
from dataclasses import dataclass
from typing import Literal

@dataclass
class SimResult:
    t:        np.ndarray    # time grid, shape (T,)
    theta:    np.ndarray    # phases, shape (T, N)
    r:        np.ndarray    # order parameter time‐series, shape (T,)
    r_final:  float         # final order parameter, r[-1]

def _kuramoto_rhs(theta: np.ndarray,
                  omega: np.ndarray,
                  K: float,
                  A: np.ndarray) -> np.ndarray:
    """
    Compute dθ/dt for each oscillator on graph A:
      dθ_i/dt = ω_i + (K / degree_i) * Σ_j A[i,j] * sin(θ_j − θ_i)
    """
    # pairwise differences θ_j − θ_i
    diff = theta[None, :] - theta[:, None]    # shape (N, N)
    sin_diff = np.sin(diff)

    # weighted sum of interactions
    interaction = np.sum(A * sin_diff, axis=1)

    # degree of each node (number of neighbors)
    degree = A.sum(axis=1)
    # avoid division by zero if an isolated node appears
    with np.errstate(divide='ignore', invalid='ignore'):
        coupling = np.where(degree > 0, K * interaction / degree, 0.0)

    return omega + coupling

def _euler(theta0: np.ndarray,
           omega: np.ndarray,
           K: float,
           A: np.ndarray,
           t_eval: np.ndarray) -> np.ndarray:
    """
    Explicit Euler integration on uniform time grid t_eval.
    Returns theta of shape (T, N).
    """
    dt = t_eval[1] - t_eval[0]
    T, N = t_eval.size, theta0.size
    theta = np.zeros((T, N), dtype=float)
    theta[0] = theta0.copy()

    for k in range(1, T):
        dθ = _kuramoto_rhs(theta[k-1], omega, K, A)
        theta[k] = (theta[k-1] + dt * dθ) % (2 * np.pi)

    return theta

def _solve_ivp(theta0: np.ndarray,
               omega: np.ndarray,
               K: float,
               A: np.ndarray,
               t_eval: np.ndarray,
               method: str = "RK45") -> np.ndarray:
    """
    SciPy solve_ivp (adaptive Runge–Kutta) on [t0, t_final].
    Returns theta of shape (T, N).
    """
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        # y has shape (N,)
        return _kuramoto_rhs(y, omega, K, A)

    sol = solve_ivp(
        rhs,
        t_span=(t_eval[0], t_eval[-1]),
        y0=theta0,
        t_eval=t_eval,
        method=method,
    )
    # sol.y has shape (N, T) → transpose to (T, N), wrap into [0, 2π)
    return sol.y.T % (2 * np.pi)

def simulate(theta0: np.ndarray,
             omega: np.ndarray,
             K: float,
             A: np.ndarray,
             t_eval: np.ndarray,
             backend: Literal["euler", "solve_ivp"] = "euler",
             **ivp_kwargs) -> SimResult:
    """
    Run one Kuramoto-model simulation on adjacency A.
    
    Parameters
    ----------
    theta0 : (N,) initial phases
    omega  : (N,) natural frequencies
    K      : coupling strength
    A      : (N, N) adjacency matrix
    t_eval : (T,) array of time points
    backend: "euler" or "solve_ivp"
    ivp_kwargs
           : extra args passed to solve_ivp (e.g. method="DOP853")
    
    Returns
    -------
    SimResult with theta(t), r(t), and r_final.
    """
    if backend == "euler":
        theta = _euler(theta0, omega, K, A, t_eval)
    elif backend == "solve_ivp":
        theta = _solve_ivp(theta0, omega, K, A, t_eval, **ivp_kwargs)
    else:
        raise ValueError("backend must be 'euler' or 'solve_ivp'")

    # compute Kuramoto order parameter at each time
    r = np.abs(np.mean(np.exp(1j * theta), axis=1))
    return SimResult(t=t_eval, theta=theta, r=r, r_final=r[-1])

def demo():
    """
    Quick demo: run one simulation using settings in config.py
    and plot both θ_i(t) and r(t).
    """
    import config
    import matplotlib.pyplot as plt

    A = config.get_adjacency(vary=False)
    res = simulate(
        config.theta0,
        config.omega,
        config.K,
        A,
        config.t_eval,
        backend="solve_ivp"
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    for i in range(config.N):
        ax1.plot(res.t, res.theta[:, i], lw=0.8)
    ax1.set_ylabel("θ (rad)")
    ax1.set_title("Phase Evolution")

    ax2.plot(res.t, res.r, color="k")
    ax2.axhline(0.9, color="r", ls="--", label="r = 0.9")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Order parameter r")
    ax2.set_title(f"Synchronisation (r_final = {res.r_final:.3f})")
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo()
