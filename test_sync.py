# test_sync.py

import config
from simulation import simulate_kuramoto, compute_order_parameter

def test_synchronisation(n_trials=10, threshold=0.9):
    """
    Runs the simulation repeatedly and checks if the final synchronisation order
    parameter r exceeds the specified threshold.
    
    Parameters:
      n_trials: Number of simulation runs.
      threshold: Synchronisation threshold (r value).
    """
    success_count = 0
    for i in range(n_trials):
        omega = config.cauchy.rvs(loc=config.x0, scale=1, size=config.N)
        t, theta_sol = simulate_kuramoto(config.theta0, omega, config.K, config.t_eval)
        final_r = compute_order_parameter(theta_sol[-1])
        print(f"Trial {i + 1}: Final order parameter, r = {final_r:.3f}")
        if final_r >= threshold:
            success_count += 1
    print(f"\nSummary: {success_count} out of {n_trials} trials reached synchronisation (r ≥ {threshold}).")

if __name__ == '__main__':
    test_synchronisation(n_trials=100, threshold=0.9)
