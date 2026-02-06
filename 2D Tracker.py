

import numpy as np
import hashlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ===================== CONFIGURATION =====================
@dataclass
class SylvachConfig:
    """Core simulation parameters for Sylvach 2D Kinetic Tracker."""
    n_nodes: int = 1000
    rounds: int = 150
    alpha: float = 1.12       # Over-relaxation factor
    k_constant: float = 0.88  # Contraction mapping constant
    seed: Optional[int] = 2026

    # Visualization
    node_size: float = 3
    cmap: str = 'viridis'
    fig_size: tuple = (16, 9)
    interval: int = 40  # ms per frame
    hud_window: int = 100  # rolling window for HUD metrics

# ===================== POTENTIAL FIELD =====================
def get_potential_field(x, y) -> np.ndarray:
    """Contraction Field: Represents the metric space topography."""
    r_sq = x**2 + y**2
    return -(0.5 * r_sq - 2.0 * np.cos(1.2*x) * np.cos(1.2*y))

# ===================== SIMULATION CLASS =====================
class SylvachSimulation2D:
    """2D kinetic simulation with node swarm and metrics HUD."""

    def __init__(self, config: SylvachConfig):
        self.config = config
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        # Node states
        self.states = np.random.uniform(-5, 5, (self.config.n_nodes, 2))
        self.prev_states = self.states.copy()

        # Metrics history
        self.metrics_names = [
            "CONTRACTION VELOCITY", 
            "PHASE COHERENCE", 
            "STABILITY TENSION", 
            "SHA-256 STATE DELTA"
        ]
        self.metrics_colors = ["#00ffcc", "#ffaa00", "#ff0055", "#aa00ff"]
        self.history: Dict[str, List[float]] = {k: [] for k in self.metrics_names}

        # ===================== FIGURE & AXES =====================
        self.fig = plt.figure(figsize=self.config.fig_size, facecolor='#050505')
        gs = self.fig.add_gridspec(2, 4)

        # Contraction map panel
        self.ax_map = self.fig.add_subplot(gs[:, 0:2], facecolor='#050505')
        extent = 6
        x_v = np.linspace(-extent, extent, 100)
        y_v = np.linspace(-extent, extent, 100)
        X, Y = np.meshgrid(x_v, y_v)
        Z = get_potential_field(X, Y)
        self.ax_map.contourf(X, Y, Z, levels=35, cmap=self.config.cmap, alpha=0.3)
        self.ax_map.contour(X, Y, Z, levels=15, colors='white', alpha=0.05, linewidths=0.5)
        self.node_scat = self.ax_map.scatter([], [], s=self.config.node_size, alpha=0.6, edgecolors='none')
        self.fixed_point = self.ax_map.plot([0], [0], 'r+', markersize=15, alpha=0.8, label="Fixed Point")[0]
        self.ax_map.set_axis_off()

        # HUD panels
        self.axes = [self.fig.add_subplot(gs[i, j], facecolor='#111') for i in [0,1] for j in [2,3]]
        self.lines = [ax.plot([], [], color=c, lw=1.5)[0] for ax, c in zip(self.axes, self.metrics_colors)]
        for ax, title in zip(self.axes, self.metrics_names):
            ax.set_title(title, color='white', fontsize=9, weight='bold')
            ax.tick_params(colors='gray', labelsize=8)
            ax.grid(True, color='#222', linestyle=':')

    def update(self, frame: int):
        """Update simulation frame."""
        current_mean = np.mean(self.states, axis=0)
        self.prev_states = self.states.copy()
        self.states = self.states + self.config.alpha * (current_mean - self.states)

        # Metrics
        velocity = np.mean(np.linalg.norm(self.states - self.prev_states, axis=1))
        to_center = -self.prev_states
        move_dir = self.states - self.prev_states
        cos_sim = np.sum(to_center * move_dir, axis=1) / (
            np.linalg.norm(to_center, axis=1) * np.linalg.norm(move_dir, axis=1) + 1e-9)
        coherence = np.mean(cos_sim)
        tension = np.std(np.linalg.norm(self.states, axis=1))
        h_delta = int(hashlib.sha256(self.states.tobytes()).hexdigest()[:6], 16) / 16777215.0

        vals = [velocity, coherence, tension, h_delta]
        for i, (line, val) in enumerate(zip(self.lines, vals)):
            self.history[self.metrics_names[i]].append(val)
            hist = self.history[self.metrics_names[i]]
            line.set_data(range(len(hist)), hist)
            recent = hist[-self.config.hud_window:]
            self.axes[i].set_xlim(max(0, frame - self.config.hud_window), frame + 5)
            self.axes[i].set_ylim(min(recent)*0.95, max(recent)*1.05 + 1e-6)

        # Update nodes
        dist_to_fixed = np.linalg.norm(self.states, axis=1)
        self.node_scat.set_offsets(self.states)
        self.node_scat.set_color(plt.cm.viridis(1 - (dist_to_fixed / 7)))

        # Title
        self.ax_map.set_title(
            fr"SYLVACH 2D KINETIC TRACKER | ROUND {frame} | α: {self.config.alpha}",
            color='white', fontsize=11
        )

        return [self.node_scat] + self.lines

    def run(self):
        """Run the animated simulation."""
        ani = animation.FuncAnimation(
            self.fig, self.update, frames=self.config.rounds, interval=self.config.interval, blit=False
        )
        plt.tight_layout()
        print(f"[*] SYLVACH KINETIC TRACKER ACTIVE: MONITORING {self.config.n_nodes} NODES IN 2D PHASE-SPACE.")
        plt.show()

# ===================== MAIN =====================
if __name__ == "__main__":
    config = SylvachConfig()
    sim = SylvachSimulation2D(config)
    sim.run()
