import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional
from config import t_eval


@dataclass
class SimResult:
    t: np.ndarray  # time grid, shape (T,)
    theta: np.ndarray  # phases, shape (T, N)
    r: np.ndarray  # order parameter time-series, shape (T,)
    r_final: float  # final order parameter, r[-1]

    # New metrics
    t_sync: Optional[float] = None  # time to reach synchronization
    phase_variance: Optional[float] = None  # variance of final phases
    convergence_rate: Optional[float] = None  # exponential convergence rate


def _kuramoto_rhs(
    theta: np.ndarray, omega: np.ndarray, K: float, A: np.ndarray
) -> np.ndarray:
    diff = theta[None, :] - theta[:, None]  # shape (N,N)
    sin_diff = np.sin(diff)
    interaction = np.sum(A * sin_diff, axis=1)
    degree = A.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        coupling = np.where(degree > 0, K * interaction / degree, 0.0)
    return omega + coupling


def _kuramoto_rhs_delay(
    theta_now: np.ndarray,
    theta_delay: np.ndarray,
    omega: np.ndarray,
    K: float,
    A: np.ndarray,
) -> np.ndarray:
    """
    Delay-coupled Kuramoto RHS:
      dθ_i/dt = ω_i + (K/deg_i) * sum_j A_ij * sin(θ_j(t-τ) - θ_i(t))
    """
    diff = theta_delay[None, :] - theta_now[:, None]
    sin_diff = np.sin(diff)
    interaction = np.sum(A * sin_diff, axis=1)
    degree = A.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        coupling = np.where(degree > 0, K * interaction / degree, 0.0)
    return omega + coupling


def _euler_delay(
    theta0: np.ndarray,
    omega: np.ndarray,
    K: float,
    A: np.ndarray,
    t_eval: np.ndarray,
    t_delay: float,
    prehistory: Literal["hold_initial", "no_delay"] = "hold_initial",
) -> np.ndarray:
    dt = t_eval[1] - t_eval[0]
    T, N = t_eval.size, theta0.size
    theta = np.zeros((T, N))
    theta[0] = theta0.copy()
    L = int(round(t_delay / dt))

    for k in range(1, T):
        if L <= 0:
            theta_delay = theta[k - 1]
        else:
            if k - L >= 0:
                theta_delay = theta[k - L]
            else:
                if prehistory == "hold_initial":
                    theta_delay = theta0
                elif prehistory == "no_delay":
                    theta_delay = theta[k - 1]
                else:
                    raise ValueError("prehistory must be 'hold_initial' or 'no_delay'")

        dθ = _kuramoto_rhs_delay(theta[k - 1], theta_delay, omega, K, A)
        theta[k] = (theta[k - 1] + dt * dθ) % (2 * np.pi)

    return theta


def _euler(
    theta0: np.ndarray, omega: np.ndarray, K: float, A: np.ndarray, t_eval: np.ndarray
) -> np.ndarray:
    dt = t_eval[1] - t_eval[0]
    T, N = t_eval.size, theta0.size
    theta = np.zeros((T, N))
    theta[0] = theta0.copy()
    for k in range(1, T):
        dθ = _kuramoto_rhs(theta[k - 1], omega, K, A)
        theta[k] = (theta[k - 1] + dt * dθ) % (2 * np.pi)
    return theta


def _solve_ivp(
    theta0: np.ndarray,
    omega: np.ndarray,
    K: float,
    A: np.ndarray,
    t_eval: np.ndarray,
    method: str = "RK45",
) -> np.ndarray:
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        return _kuramoto_rhs(y, omega, K, A)

    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), theta0, t_eval=t_eval, method=method)
    return sol.y.T % (2 * np.pi)


def compute_metrics(
    t: np.ndarray, theta: np.ndarray, r: np.ndarray, threshold: float = 0.9
) -> dict:
    """
    Compute additional metrics for synchronization analysis.

    Returns:
        dict with keys:
            - t_sync: time to reach r >= threshold (None if never reached)
            - phase_variance: variance of final phases
            - convergence_rate: exponential fit to r(t) growth
    """
    metrics = {}

    # Time to synchronization
    sync_indices = np.where(r >= threshold)[0]
    if len(sync_indices) > 0:
        metrics["t_sync"] = t[sync_indices[0]]
    else:
        metrics["t_sync"] = None

    # Phase variance at final time
    final_phases = theta[-1, :]
    # Use circular variance: 1 - |mean(e^{i*theta})|
    metrics["phase_variance"] = 1 - np.abs(np.mean(np.exp(1j * final_phases)))

    # Convergence rate (exponential fit to second half of simulation)
    # Fit r(t) ≈ 1 - exp(-λt) or r(t) ≈ r0 * exp(λt)
    try:
        midpoint = len(r) // 2
        if r[midpoint] < 0.5:  # Growing from low r
            valid = r[midpoint:] > 0.01
            if np.any(valid):
                t_fit = t[midpoint:][valid]
                r_fit = r[midpoint:][valid]
                log_r = np.log(r_fit)
                # Linear fit to log(r) vs t
                coeffs = np.polyfit(t_fit, log_r, 1)
                metrics["convergence_rate"] = coeffs[0]
            else:
                metrics["convergence_rate"] = None
        else:  # Already high, fit to 1-r
            remainder = 1 - r[midpoint:]
            valid = remainder > 0.001
            if np.any(valid):
                t_fit = t[midpoint:][valid]
                remainder_fit = remainder[valid]
                log_remainder = np.log(remainder_fit)
                coeffs = np.polyfit(t_fit, log_remainder, 1)
                metrics["convergence_rate"] = -coeffs[0]  # Negative decay rate
            else:
                metrics["convergence_rate"] = None
    except:
        metrics["convergence_rate"] = None

    return metrics


def simulate(
    theta0: np.ndarray,
    omega: np.ndarray,
    K: float,
    A: np.ndarray,
    t_eval: np.ndarray,
    backend: Literal["euler", "solve_ivp", "euler_delay"] = "euler",
    t_delay: float = 0.0,
    prehistory: Literal["hold_initial", "no_delay"] = "hold_initial",
    compute_extra_metrics: bool = True,
    sync_threshold: float = 0.9,
    **ivp_kwargs,
) -> SimResult:
    """
    Simulate the Kuramoto model.

    Args:
        theta0: Initial phases, shape (N,)
        omega: Natural frequencies, shape (N,)
        K: Coupling strength
        A: Adjacency matrix, shape (N, N)
        t_eval: Time points to evaluate
        backend: Integration method
        t_delay: Time delay (only for euler_delay)
        prehistory: How to handle pre-t0 history for delay
        compute_extra_metrics: If True, compute additional metrics
        sync_threshold: Threshold for computing t_sync
        **ivp_kwargs: Additional arguments for solve_ivp

    Returns:
        SimResult object with simulation data and metrics
    """
    if backend == "euler":
        theta = _euler(theta0, omega, K, A, t_eval)
    elif backend == "solve_ivp":
        theta = _solve_ivp(theta0, omega, K, A, t_eval, **ivp_kwargs)
    elif backend == "euler_delay":
        theta = _euler_delay(theta0, omega, K, A, t_eval, t_delay, prehistory)
    else:
        raise ValueError("backend must be 'euler', 'solve_ivp', or 'euler_delay'")

    r = np.abs(np.mean(np.exp(1j * theta), axis=1))

    result = SimResult(t=t_eval, theta=theta, r=r, r_final=r[-1])

    if compute_extra_metrics:
        metrics = compute_metrics(t_eval, theta, r, sync_threshold)
        result.t_sync = metrics["t_sync"]
        result.phase_variance = metrics["phase_variance"]
        result.convergence_rate = metrics["convergence_rate"]

    return result


def demo():
    import config
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    A = config.get_adjacency(vary=False)
    res = simulate(
        config.theta0,
        config.omega,
        config.K,
        A,
        config.t_eval,
        backend="euler_delay",
        t_delay=config.t_delay,
        prehistory="hold_initial",
    )

    # Print metrics
    print(f"Final order parameter: {res.r_final:.3f}")
    print(
        f"Time to sync (r>0.9): {res.t_sync:.3f}s"
        if res.t_sync
        else "Did not synchronize"
    )
    print(f"Phase variance: {res.phase_variance:.4f}")
    print(
        f"Convergence rate: {res.convergence_rate:.4f}"
        if res.convergence_rate
        else "N/A"
    )

    N = config.N
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(N)}

    deg = A.sum(axis=1)
    weights = {}
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j]:
                w = config.K * 0.5 * (1 / deg[i] + 1 / deg[j])
                weights[(i, j)] = w

    max_w = max(weights.values()) if weights else 1.0
    scale = 5.0 / max_w

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.axis("off")
    title = ax.text(-1.0, 1.0, "t = 0.0s")

    for (i, j), w in weights.items():
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        ax.plot([x0, x1], [y0, y1], lw=w * scale, color="gray", zorder=1)

    xs = [pos[i][0] for i in range(N)]
    ys = [pos[i][1] for i in range(N)]
    scat = ax.scatter(
        xs,
        ys,
        c=res.theta[0],
        cmap="hsv",
        vmin=0,
        vmax=2 * np.pi,
        s=200,
        edgecolors="k",
        zorder=3,
    )

    frames = np.arange(0, len(res.t), max(1, len(res.t) // 100))
    framerate = 50

    def update(frame):
        scat.set_array(res.theta[frame])
        title.set_text(f"t = {res.t[frame]:.2f}s, r = {res.r[frame]:.3f}")
        return (
            scat,
            title,
        )

    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=framerate, blit=True
    )
    plt.show()


if __name__ == "__main__":
    demo()
