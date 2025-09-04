import argparse
import config
from simulation import simulate
import numpy as np

def test_synchronisation(n_trials=10,
                         threshold=0.9,
                         vary="omega",
                         backend="solve_ivp"):
    """
    Same initial condition every trial except for one quantity:
      vary="omega"   → redraw natural frequencies each run
      vary=None      → reuse the exact same ω and graph
    """
    ok = 0
    # static graph adjacency
    A = config.get_adjacency(vary=False)

    for i in range(n_trials):
        if vary == "omega":
            ω = config.draw_omega()
        else:
            ω = config.omega
        if vary == "num":
            if n_trials!=0:
                A = config.get_adjacency(Nloc=i)
                res = simulate(config.theta0[:i], ω, config.K,
                               A, config.t_eval, backend=backend)
                print(f"trial {i+1:02d}: r_final = {res.r_final:.3f}")
                ok += (res.r_final >= threshold)
        else:
            res = simulate(config.theta0, ω, config.K,
                       A, config.t_eval, backend=backend)
            print(f"trial {i+1:02d}: r_final = {res.r_final:.3f}")
            ok += (res.r_final >= threshold)

    print(f"\n{ok}/{n_trials} trials reached r ≥ {threshold}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--n_trials", type=int, default=10)
    parser.add_argument("-t","--threshold", type=float, default=0.9)
    parser.add_argument("-v","--vary", choices=["omega","num","none"], default="none",
                        help="what to reshuffle each trial")
    parser.add_argument("-b","--backend", choices=["euler","solve_ivp"],
                        default="solve_ivp")
    args = parser.parse_args()

    vary = None if args.vary == "none" else args.vary
    test_synchronisation(args.n_trials, args.threshold, vary, args.backend)
