"""
experiments.py - Comprehensive experiment suite for Kuramoto model analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import config
from simulation import simulate
import pandas as pd
from tqdm import tqdm


def experiment_1_coupling_vs_time(K_values=None, n_trials=10, save_results=True):
    """
    Experiment 1: Effect of coupling strength on synchronization time

    Varies K and measures time to reach r >= 0.9
    """
    if K_values is None:
        K_values = np.linspace(0, 10, 21)

    results = []
    A = config.get_adjacency(vary=False)

    print("Experiment 1: Coupling strength vs. synchronization time")
    for K in tqdm(K_values):
        for trial in range(n_trials):
            omega = config.draw_omega()
            theta0 = np.random.uniform(0, 2 * np.pi, config.N)

            res = simulate(theta0, omega, K, A, config.t_eval)

            results.append(
                {
                    "K": K,
                    "trial": trial,
                    "t_sync": res.t_sync,
                    "r_final": res.r_final,
                    "phase_variance": res.phase_variance,
                    "convergence_rate": res.convergence_rate,
                }
            )

    df = pd.DataFrame(results)

    if save_results:
        df.to_csv("results_exp1_coupling.csv", index=False)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Time to sync vs K
    grouped = df.groupby("K")["t_sync"].agg(["mean", "std"])
    axes[0, 0].errorbar(
        grouped.index, grouped["mean"], yerr=grouped["std"], fmt="o-", capsize=5
    )
    axes[0, 0].set_xlabel("Coupling Strength K")
    axes[0, 0].set_ylabel("Time to Synchronization (s)")
    axes[0, 0].set_title("Synchronization Time vs. Coupling")
    axes[0, 0].grid(True)

    # Final r vs K
    grouped = df.groupby("K")["r_final"].agg(["mean", "std"])
    axes[0, 1].errorbar(
        grouped.index, grouped["mean"], yerr=grouped["std"], fmt="o-", capsize=5
    )
    axes[0, 1].set_xlabel("Coupling Strength K")
    axes[0, 1].set_ylabel("Final Order Parameter r")
    axes[0, 1].set_title("Final Synchronization vs. Coupling")
    axes[0, 1].axhline(y=0.9, color="r", linestyle="--", label="Threshold")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Phase variance vs K
    grouped = df.groupby("K")["phase_variance"].agg(["mean", "std"])
    axes[1, 0].errorbar(
        grouped.index, grouped["mean"], yerr=grouped["std"], fmt="o-", capsize=5
    )
    axes[1, 0].set_xlabel("Coupling Strength K")
    axes[1, 0].set_ylabel("Phase Variance")
    axes[1, 0].set_title("Phase Variance vs. Coupling")
    axes[1, 0].grid(True)

    # Convergence rate vs K
    grouped = (
        df[df["convergence_rate"].notna()]
        .groupby("K")["convergence_rate"]
        .agg(["mean", "std"])
    )
    axes[1, 1].errorbar(
        grouped.index, grouped["mean"], yerr=grouped["std"], fmt="o-", capsize=5
    )
    axes[1, 1].set_xlabel("Coupling Strength K")
    axes[1, 1].set_ylabel("Convergence Rate")
    axes[1, 1].set_title("Convergence Rate vs. Coupling")
    axes[1, 1].grid(True)

    plt.tight_layout()
    if save_results:
        plt.savefig("exp1_coupling_vs_time.png", dpi=300)
    plt.show()

    return df


def experiment_2_network_size(N_values=None, K=5, n_trials=10, save_results=True):
    """
    Experiment 2: Effect of network size on synchronization
    """
    if N_values is None:
        N_values = [5, 10, 20, 30, 50]

    results = []

    print("Experiment 2: Network size scaling")
    for N in tqdm(N_values):
        for trial in range(n_trials):
            omega = config.draw_omega(size=N)
            theta0 = np.random.uniform(0, 2 * np.pi, N)
            A = config.get_adjacency(vary=False, Nloc=N)

            res = simulate(theta0, omega, K, A, config.t_eval)

            results.append(
                {
                    "N": N,
                    "trial": trial,
                    "t_sync": res.t_sync,
                    "r_final": res.r_final,
                    "phase_variance": res.phase_variance,
                }
            )

    df = pd.DataFrame(results)

    if save_results:
        df.to_csv("results_exp2_network_size.csv", index=False)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    grouped = df.groupby("N")["t_sync"].agg(["mean", "std"])
    axes[0].errorbar(
        grouped.index, grouped["mean"], yerr=grouped["std"], fmt="o-", capsize=5
    )
    axes[0].set_xlabel("Number of Oscillators N")
    axes[0].set_ylabel("Time to Synchronization (s)")
    axes[0].set_title("Sync Time vs. Network Size")
    axes[0].grid(True)

    grouped = df.groupby("N")["r_final"].agg(["mean", "std"])
    axes[1].errorbar(
        grouped.index, grouped["mean"], yerr=grouped["std"], fmt="o-", capsize=5
    )
    axes[1].set_xlabel("Number of Oscillators N")
    axes[1].set_ylabel("Final Order Parameter r")
    axes[1].set_title("Final Sync vs. Network Size")
    axes[1].axhline(y=0.9, color="r", linestyle="--")
    axes[1].grid(True)

    grouped = df.groupby("N")["phase_variance"].agg(["mean", "std"])
    axes[2].errorbar(
        grouped.index, grouped["mean"], yerr=grouped["std"], fmt="o-", capsize=5
    )
    axes[2].set_xlabel("Number of Oscillators N")
    axes[2].set_ylabel("Phase Variance")
    axes[2].set_title("Phase Variance vs. Network Size")
    axes[2].grid(True)

    plt.tight_layout()
    if save_results:
        plt.savefig("exp2_network_size.png", dpi=300)
    plt.show()

    return df


def experiment_3_network_topology(K=5, n_trials=10, save_results=True):
    """
    Experiment 3: Comparison of different network topologies
    """
    topologies = ["full", "ring", "star", "random_er"]
    results = []

    print("Experiment 3: Network topology comparison")
    for topology in tqdm(topologies):
        for trial in range(n_trials):
            omega = config.draw_omega()
            theta0 = np.random.uniform(0, 2 * np.pi, config.N)

            # Create adjacency matrix for this topology
            old_type = config.network_type
            config.network_type = topology
            A = config.get_adjacency(vary=(topology == "random_er"))
            config.network_type = old_type

            res = simulate(theta0, omega, K, A, config.t_eval)

            results.append(
                {
                    "topology": topology,
                    "trial": trial,
                    "t_sync": res.t_sync,
                    "r_final": res.r_final,
                    "phase_variance": res.phase_variance,
                    "convergence_rate": res.convergence_rate,
                    "avg_degree": A.sum(axis=1).mean(),
                }
            )

    df = pd.DataFrame(results)

    if save_results:
        df.to_csv("results_exp3_topology.csv", index=False)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Time to sync by topology
    df_plot = df[df["t_sync"].notna()]
    df_plot.boxplot(column="t_sync", by="topology", ax=axes[0, 0])
    axes[0, 0].set_xlabel("Network Topology")
    axes[0, 0].set_ylabel("Time to Synchronization (s)")
    axes[0, 0].set_title("Sync Time by Topology")
    plt.sca(axes[0, 0])
    plt.xticks(rotation=45)

    # Final r by topology
    df.boxplot(column="r_final", by="topology", ax=axes[0, 1])
    axes[0, 1].axhline(y=0.9, color="r", linestyle="--")
    axes[0, 1].set_xlabel("Network Topology")
    axes[0, 1].set_ylabel("Final Order Parameter r")
    axes[0, 1].set_title("Final Sync by Topology")
    plt.sca(axes[0, 1])
    plt.xticks(rotation=45)

    # Phase variance by topology
    df.boxplot(column="phase_variance", by="topology", ax=axes[1, 0])
    axes[1, 0].set_xlabel("Network Topology")
    axes[1, 0].set_ylabel("Phase Variance")
    axes[1, 0].set_title("Phase Variance by Topology")
    plt.sca(axes[1, 0])
    plt.xticks(rotation=45)

    # Average degree by topology
    grouped = df.groupby("topology")["avg_degree"].mean()
    axes[1, 1].bar(range(len(grouped)), grouped.values)
    axes[1, 1].set_xticks(range(len(grouped)))
    axes[1, 1].set_xticklabels(grouped.index, rotation=45)
    axes[1, 1].set_xlabel("Network Topology")
    axes[1, 1].set_ylabel("Average Degree")
    axes[1, 1].set_title("Average Connectivity by Topology")

    plt.tight_layout()
    if save_results:
        plt.savefig("exp3_topology_comparison.png", dpi=300)
    plt.show()

    return df


def experiment_4_time_delay(delay_values=None, K=5, n_trials=10, save_results=True):
    """
    Experiment 4: Effect of time delay on synchronization
    """
    if delay_values is None:
        delay_values = np.linspace(0, 2.0, 21)

    results = []
    A = config.get_adjacency(vary=False)

    print("Experiment 4: Time delay effects")
    for delay in tqdm(delay_values):
        for trial in range(n_trials):
            omega = config.draw_omega()
            theta0 = np.random.uniform(0, 2 * np.pi, config.N)

            res = simulate(
                theta0, omega, K, A, config.t_eval, backend="euler_delay", t_delay=delay
            )

            results.append(
                {
                    "delay": delay,
                    "trial": trial,
                    "t_sync": res.t_sync,
                    "r_final": res.r_final,
                    "phase_variance": res.phase_variance,
                }
            )

    df = pd.DataFrame(results)

    if save_results:
        df.to_csv("results_exp4_time_delay.csv", index=False)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    grouped = df[df["t_sync"].notna()].groupby("delay")["t_sync"].agg(["mean", "std"])
    axes[0].errorbar(
        grouped.index, grouped["mean"], yerr=grouped["std"], fmt="o-", capsize=5
    )
    axes[0].set_xlabel("Time Delay τ (s)")
    axes[0].set_ylabel("Time to Synchronization (s)")
    axes[0].set_title("Sync Time vs. Delay")
    axes[0].grid(True)

    grouped = df.groupby("delay")["r_final"].agg(["mean", "std"])
    axes[1].errorbar(
        grouped.index, grouped["mean"], yerr=grouped["std"], fmt="o-", capsize=5
    )
    axes[1].set_xlabel("Time Delay τ (s)")
    axes[1].set_ylabel("Final Order Parameter r")
    axes[1].set_title("Final Sync vs. Delay")
    axes[1].axhline(y=0.9, color="r", linestyle="--")
    axes[1].grid(True)

    grouped = df.groupby("delay")["phase_variance"].agg(["mean", "std"])
    axes[2].errorbar(
        grouped.index, grouped["mean"], yerr=grouped["std"], fmt="o-", capsize=5
    )
    axes[2].set_xlabel("Time Delay τ (s)")
    axes[2].set_ylabel("Phase Variance")
    axes[2].set_title("Phase Variance vs. Delay")
    axes[2].grid(True)

    plt.tight_layout()
    if save_results:
        plt.savefig("exp4_time_delay.png", dpi=300)
    plt.show()

    return df


def experiment_5_critical_coupling(
    topologies=None, K_range=(0, 10), K_steps=50, n_trials=20, save_results=True
):
    """
    Experiment 5: Finding critical coupling for different topologies
    """
    if topologies is None:
        topologies = ["full", "ring", "star", "random_er"]

    K_values = np.linspace(K_range[0], K_range[1], K_steps)
    results = []

    print("Experiment 5: Critical coupling threshold")
    for topology in tqdm(topologies):
        for K in K_values:
            sync_count = 0
            for trial in range(n_trials):
                omega = config.draw_omega()
                theta0 = np.random.uniform(0, 2 * np.pi, config.N)

                old_type = config.network_type
                config.network_type = topology
                A = config.get_adjacency(vary=(topology == "random_er"))
                config.network_type = old_type

                res = simulate(theta0, omega, K, A, config.t_eval)

                if res.r_final >= 0.9:
                    sync_count += 1

            sync_prob = sync_count / n_trials
            results.append(
                {"topology": topology, "K": K, "sync_probability": sync_prob}
            )

    df = pd.DataFrame(results)

    if save_results:
        df.to_csv("results_exp5_critical_coupling.csv", index=False)

    # Plot phase diagram
    plt.figure(figsize=(10, 6))
    for topology in topologies:
        subset = df[df["topology"] == topology]
        plt.plot(
            subset["K"],
            subset["sync_probability"],
            marker="o",
            label=topology,
            linewidth=2,
        )

    plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Coupling Strength K", fontsize=12)
    plt.ylabel("Synchronization Probability", fontsize=12)
    plt.title("Phase Transition to Synchronization", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_results:
        plt.savefig("exp5_critical_coupling.png", dpi=300)
    plt.show()

    return df


def run_all_experiments():
    """Run all experiments sequentially"""
    print("=" * 60)
    print("Running all experiments for Kuramoto model analysis")
    print("=" * 60)

    exp1_df = experiment_1_coupling_vs_time()
    exp2_df = experiment_2_network_size()
    exp3_df = experiment_3_network_topology()
    exp4_df = experiment_4_time_delay()
    exp5_df = experiment_5_critical_coupling()

    print("\nAll experiments completed!")
    print("Results saved to CSV files and plots saved as PNG files.")

    return {
        "exp1": exp1_df,
        "exp2": exp2_df,
        "exp3": exp3_df,
        "exp4": exp4_df,
        "exp5": exp5_df,
    }


if __name__ == "__main__":
    # Run all experiments
    results = run_all_experiments()
