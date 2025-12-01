"""
Sweep coupling K on your chosen network and plot r_final vs K.
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
    r_final = []
    for K in K_values:
        res = simulate(theta0, omega, K, A, t_eval,
                       backend=backend, **ivp_kwargs)
        print(f"K = {K:.2f} → r_final = {res.r_final:.3f}")
        r_final.append(res.r_final)
    return np.array(r_final)

def main():
    # single adjacency from config
    A      = config.get_adjacency(vary=False)
    theta0 = config.theta0
    omega  = config.omega
    t_eval = config.t_eval

    K_values = np.linspace(0.0, 8.0, 41)
    r_inf    = sweep_K(K_values, theta0, omega, A, t_eval,
                       backend="solve_ivp")

    plt.figure(figsize=(6,4))
    plt.plot(K_values, r_inf, marker='o', lw=1.2)
    plt.axhline(0.9, color='r', ls='--', label="r = 0.9")
    plt.xlabel("Coupling strength K")
    plt.ylabel("Final order parameter r")
    plt.title(f"Synchronisation on {config.network_type} network")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
