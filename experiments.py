"""
Sweep the coupling strength K and plot the final synchronisation r∞ vs K.
"""

import numpy as np
import matplotlib.pyplot as plt
import config
from simulation import simulate

def sweep_K(K_values,
            theta0,
            omega,
            A,
            t_eval,
            backend="solve_ivp",
            **ivp_kwargs):
    """
    For each K in K_values, run one simulation and record r_final.
    Returns array of r_final of same length as K_values.
    """
    r_final = []
    for K in K_values:
        res = simulate(theta0, omega, K, A, t_eval,
                       backend=backend, **ivp_kwargs)
        print(f"K = {K:.2f} → r_final = {res.r_final:.3f}")
        r_final.append(res.r_final)
    return np.array(r_final)

def main():
    # 1) build a single static network adjacency
    A = config.get_adjacency(vary=False)

    # 2) grab the fixed ICs
    theta0 = config.theta0
    omega  = config.omega
    t_eval = config.t_eval

    # 3) define sweep range for K
    K_values = np.linspace(0.0, 1.0, 101)

    # 4) run the sweep
    r_inf = sweep_K(K_values, theta0, omega, A, t_eval,
                    backend="solve_ivp")

    # 5) plot r∞ vs K
    plt.figure(figsize=(6, 4))
    plt.plot(K_values, r_inf, marker='o', lw=1.2)
    plt.axhline(0.9, color='r', ls='--', label="r = 0.9")
    plt.xlabel("Coupling strength K")
    plt.ylabel("Final order parameter r(t_final)")
    plt.title("Kuramoto Synchronisation Threshold")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
