

import hashlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import time
from dataclasses import dataclass, field

# ===================== CONFIGURATION =====================
@dataclass
class SimulationConfig:
    """
    Core simulation parameters.
    """
    n_nodes: int = 100_000
    alpha: float = 0.05
    noise_std: float = 0.4
    interval: int = 50           # milliseconds per frame
    max_display: int = 35        # number of hashes to show in log
    node_size: float = 0.8
    seed: int = None              # Optional reproducible seed

    # Visualization
    bg_color: str = '#000000'
    frame_color: str = '#333333'
    accent: str = '#00FFAD'
    text_main: str = '#FFFFFF'
    collision_color: str = '#FF0055'

# ===================== SIMULATION CLASS =====================
class HashSimulation2D:
    def __init__(self, config: SimulationConfig):
        self.config = config

        # Seed initialization
        if self.config.seed is None:
            self.config.seed = int(time.time() * 1000) % 2**32
        np.random.seed(self.config.seed)

        # Node state initialization
        self.states = np.random.uniform(-9, 9, (self.config.n_nodes, 2))
        self.sha_history = []

        # ===================== FIGURE SETUP =====================
        self.fig = plt.figure(figsize=(22, 11), facecolor=self.config.bg_color)

        # --- LEFT PANEL: SHA-256 Log ---
        self.ax_list = self.fig.add_axes([0.03, 0.05, 0.46, 0.9], facecolor=self.config.bg_color)
        for spine in self.ax_list.spines.values():
            spine.set_visible(True)
            spine.set_color(self.config.frame_color)
            spine.set_linewidth(1.5)
        self.ax_list.set_xticks([]); self.ax_list.set_yticks([])
        self.hash_display = self.ax_list.text(
            0.02, 0.97, "", color=self.config.text_main, fontsize=9,
            family='monospace', va='top', linespacing=1.6
        )
        self.ax_list.set_title("CRYPTOGRAPHIC STATE LOG (SHA-256)", 
                              color=self.config.accent, loc='left', fontsize=14, pad=15)

        # --- RIGHT PANEL: NODE DYNAMICS ---
        self.ax_nodes = self.fig.add_axes([0.51, 0.05, 0.46, 0.9], facecolor=self.config.bg_color)
        for spine in self.ax_nodes.spines.values():
            spine.set_visible(True)
            spine.set_color(self.config.frame_color)
            spine.set_linewidth(1.5)
        self.ax_nodes.set_xlim(-12, 12); self.ax_nodes.set_ylim(-12, 12)
        self.ax_nodes.set_xticks([]); self.ax_nodes.set_yticks([])
        self.ax_nodes.set_title("STOCHASTIC TOPOLOGY DYNAMICS", 
                               color=self.config.accent, loc='right', fontsize=14, pad=15)

        # Iteration HUD
        self.count_text = self.ax_nodes.text(
            0, 11.2, "SYSTEM: INITIALIZING", color=self.config.text_main, 
            fontsize=13, family='monospace', ha='center', weight='bold'
        )

        # Initialize scatter plot
        norms = np.linalg.norm(self.states, axis=1)
        self.scatter = self.ax_nodes.scatter(
            self.states[:, 0], self.states[:, 1],
            c=norms, cmap='magma', s=self.config.node_size, alpha=0.6, edgecolors='none'
        )

        self.running = True

    # ===================== UPDATE LOOP =====================
    def update(self, frame):
        if not self.running:
            return self.scatter, self.hash_display, self.count_text

        # --- NODE DYNAMICS ---
        centroid = np.mean(self.states, axis=0)
        self.states += self.config.alpha * (centroid - self.states)
        self.states += np.random.normal(0, self.config.noise_std, self.states.shape)

        # --- HASHING & COLLISION CHECK ---
        current_hash = hashlib.sha256(self.states.tobytes()).hexdigest()
        if current_hash in self.sha_history:
            self.running = False
            self.count_text.set_text("!! STATE COLLISION DETECTED !!")
            self.count_text.set_color(self.config.collision_color)
            return self.scatter, self.hash_display, self.count_text

        self.sha_history.append(current_hash)
        count = len(self.sha_history)

        # --- VISUAL UPDATE ---
        self.scatter.set_offsets(self.states)
        self.scatter.set_array(np.linalg.norm(self.states, axis=1))
        self.count_text.set_text(f"ITERATION: {count:06d}")

        # --- HASH LOG DISPLAY ---
        display_list = self.sha_history[-self.config.max_display:]
        start_idx = count - len(display_list) + 1
        lines = []
        for i, h in enumerate(display_list):
            idx = start_idx + i
            if i == len(display_list) - 1:
                lines.append(f"[{idx:06d}] > {h}")
            else:
                lines.append(f" {idx:06d} | {h}")
        self.hash_display.set_text("\n".join(lines))

        return self.scatter, self.hash_display, self.count_text

    # ===================== RUN SIMULATION =====================
    def run(self):
        self.ani = animation.FuncAnimation(
            self.fig, self.update, frames=None,
            interval=self.config.interval, blit=True, cache_frame_data=False
        )
        plt.show()

# ===================== MAIN =====================
def main():
    parser = argparse.ArgumentParser(description="Run 2D stochastic hash simulation")
    parser.add_argument("--nodes", type=int, default=100_000, help="Number of nodes")
    parser.add_argument("--alpha", type=float, default=0.05, help="Convergence factor")
    parser.add_argument("--noise", type=float, default=0.4, help="Noise standard deviation")
    parser.add_argument("--max_hashes", type=int, default=35, help="Max hashes to display")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    config = SimulationConfig(
        n_nodes=args.nodes,
        alpha=args.alpha,
        noise_std=args.noise,
        max_display=args.max_hashes,
        seed=args.seed
    )

    sim = HashSimulation2D(config)
    sim.run()

if __name__ == "__main__":
    main()
