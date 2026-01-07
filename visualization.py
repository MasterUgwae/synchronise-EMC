"""
visualization.py - Publication-quality visualization tools
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.animation as animation
from simulation import simulate
import config

# Set publication style
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 150


def plot_order_parameter_evolution(
    results_list, labels, colors=None, save_path="order_param_evolution.png"
):
    """
    Plot r(t) for multiple simulations on the same axes.

    Args:
        results_list: List of SimResult objects
        labels: List of labels for each simulation
        colors: Optional list of colors
        save_path: Path to save figure
    """
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))

    fig, ax = plt.subplots(figsize=(8, 5))

    for res, label, color in zip(results_list, labels, colors):
        ax.plot(res.t, res.r, label=label, color=color, linewidth=2)

    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="Sync threshold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Order Parameter r(t)")
    ax.set_title("Order Parameter Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_phase_space(res, times_to_plot=None, save_path="phase_space.png"):
    """
    Plot phase trajectories θ_i(t) for all oscillators.

    Args:
        res: SimResult object
        times_to_plot: Specific time indices to highlight (optional)
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    N = res.theta.shape[1]
    colors = plt.cm.hsv(np.linspace(0, 1, N, endpoint=False))

    for i in range(N):
        ax.plot(
            res.t,
            res.theta[:, i],
            color=colors[i],
            alpha=0.6,
            linewidth=1,
            label=f"Oscillator {i+1}" if N <= 10 else None,
        )

    if times_to_plot is not None:
        for t_idx in times_to_plot:
            ax.axvline(
                x=res.t[t_idx], color="gray", linestyle="--", alpha=0.3, linewidth=1
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phase θ (radians)")
    ax.set_title("Phase Trajectories")
    ax.set_ylim([0, 2 * np.pi])
    ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"])
    if N <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_network_with_weights(
    A, K, pos=None, title="Network Structure", save_path="network_structure.png"
):
    """
    Visualize network topology with edge weights.

    Args:
        A: Adjacency matrix
        K: Coupling strength
        pos: Dictionary of node positions {i: (x, y)}
        title: Plot title
        save_path: Path to save figure
    """
    N = A.shape[0]

    if pos is None:
        # Arrange nodes in a circle
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(N)}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)

    # Calculate edge weights
    deg = A.sum(axis=1)
    weights = {}
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] > 0:
                w = (
                    K * 0.5 * (1 / deg[i] + 1 / deg[j])
                    if deg[i] > 0 and deg[j] > 0
                    else 0
                )
                weights[(i, j)] = w

    if len(weights) > 0:
        max_w = max(weights.values())
        scale = 5.0 / max_w if max_w > 0 else 1.0

        # Draw edges
        for (i, j), w in weights.items():
            x0, y0 = pos[i]
            x1, y1 = pos[j]
            ax.plot([x0, x1], [y0, y1], lw=w * scale, color="gray", alpha=0.6, zorder=1)

    # Draw nodes
    xs = [pos[i][0] for i in range(N)]
    ys = [pos[i][1] for i in range(N)]
    ax.scatter(xs, ys, s=300, c="lightblue", edgecolors="black", linewidths=2, zorder=3)

    # Add node labels
    for i in range(N):
        ax.text(
            pos[i][0],
            pos[i][1],
            str(i + 1),
            ha="center",
            va="center",
            fontsize=10,
            zorder=4,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_circular_phases(
    theta, t_indices=None, labels=None, save_path="circular_phases.png"
):
    """
    Plot phases on unit circle at specific time points.

    Args:
        theta: Phase array, shape (T, N)
        t_indices: List of time indices to plot
        labels: Labels for each time point
        save_path: Path to save figure
    """
    if t_indices is None:
        # Plot initial, middle, and final
        T = theta.shape[0]
        t_indices = [0, T // 2, T - 1]

    if labels is None:
        labels = [f"t = {i}" for i in t_indices]

    n_plots = len(t_indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, t_idx, label in zip(axes, t_indices, labels):
        ax.set_aspect("equal")
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_title(label)

        # Draw unit circle
        circle = Circle((0, 0), 1, fill=False, color="gray", linewidth=2)
        ax.add_patch(circle)

        # Plot phases
        phases = theta[t_idx]
        x = np.cos(phases)
        y = np.sin(phases)
        ax.scatter(x, y, s=100, c="red", alpha=0.7, edgecolors="black")

        # Draw mean resultant vector
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        r = np.sqrt(mean_x**2 + mean_y**2)
        if r > 0.01:
            arrow = FancyArrowPatch(
                (0, 0),
                (mean_x, mean_y),
                arrowstyle="->",
                mutation_scale=20,
                linewidth=2,
                color="blue",
            )
            ax.add_patch(arrow)
            ax.text(0.05, 1.3, f"r = {r:.3f}", fontsize=10)

        ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color="gray", linewidth=0.5, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_heatmap_sync_region(
    df, x_col, y_col, z_col="r_final", threshold=0.9, save_path="sync_heatmap.png"
):
    """
    Create a heatmap showing synchronization regions.

    Args:
        df: DataFrame with experimental results
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        z_col: Column name for color values
        threshold: Threshold to mark synchronized region
        save_path: Path to save figure
    """
    # Pivot data for heatmap
    pivot = df.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc="mean")

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        pivot.values, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0, vmax=1
    )

    # Add contour line at threshold
    if threshold is not None:
        contour = ax.contour(
            pivot.values, levels=[threshold], colors="blue", linewidths=2, alpha=0.7
        )
        ax.clabel(contour, inline=True, fontsize=10)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], rotation=45)
    ax.set_yticklabels([f"{y:.2f}" for y in pivot.index])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{z_col} Heatmap (Synchronized region r ≥ {threshold})")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(z_col)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def create_animation_with_plot(res, A, save_path="animation.mp4", fps=30):
    """
    Create animation showing network and order parameter.

    Args:
        res: SimResult object
        A: Adjacency matrix
        save_path: Path to save video
        fps: Frames per second
    """
    N = res.theta.shape[1]
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(N)}

    fig = plt.figure(figsize=(14, 6))

    # Left: network visualization
    ax1 = plt.subplot(121)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])

    # Draw edges
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j]:
                x0, y0 = pos[i]
                x1, y1 = pos[j]
                ax1.plot([x0, x1], [y0, y1], "gray", alpha=0.3, zorder=1)

    # Node scatter
    xs = [pos[i][0] for i in range(N)]
    ys = [pos[i][1] for i in range(N)]
    scat = ax1.scatter(
        xs,
        ys,
        c=res.theta[0],
        cmap="hsv",
        vmin=0,
        vmax=2 * np.pi,
        s=300,
        edgecolors="black",
        linewidths=2,
        zorder=3,
    )

    title1 = ax1.set_title(f"t = 0.00s, r = {res.r[0]:.3f}")

    # Right: order parameter plot
    ax2 = plt.subplot(122)
    (line,) = ax2.plot([], [], "b-", linewidth=2)
    ax2.axhline(y=0.9, color="r", linestyle="--", alpha=0.5)
    ax2.set_xlim([res.t[0], res.t[-1]])
    ax2.set_ylim([0, 1.05])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Order Parameter r(t)")
    ax2.set_title("Synchronization Progress")
    ax2.grid(True, alpha=0.3)

    # Animation update function
    frames = range(0, len(res.t), max(1, len(res.t) // (fps * 10)))

    def update(frame):
        # Update node colors
        scat.set_array(res.theta[frame])
        title1.set_text(f"t = {res.t[frame]:.2f}s, r = {res.r[frame]:.3f}")

        # Update order parameter line
        line.set_data(res.t[: frame + 1], res.r[: frame + 1])

        return scat, title1, line

    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=1000 / fps, blit=True
    )

    # Save animation
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=fps, bitrate=1800)
    ani.save(save_path, writer=writer)
    print(f"Animation saved to {save_path}")

    plt.close()


def demo_visualizations():
    """Generate example visualizations"""
    # Run simulation
    A = config.get_adjacency()
    res = simulate(config.theta0, config.omega, config.K, A, config.t_eval)

    print("Generating visualizations...")

    # Order parameter evolution
    plot_order_parameter_evolution(
        [res], ["Simulation"], save_path="demo_order_param.png"
    )

    # Phase space
    plot_phase_space(res, save_path="demo_phase_space.png")

    # Network structure
    plot_network_with_weights(A, config.K, save_path="demo_network.png")

    # Circular phases
    T = len(res.t)
    plot_circular_phases(
        res.theta,
        t_indices=[0, T // 2, T - 1],
        labels=["Initial", "Middle", "Final"],
        save_path="demo_circular_phases.png",
    )

    print("Visualizations complete!")


if __name__ == "__main__":
    demo_visualizations()
