import hashlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# ===================== CONFIGURATION =====================
class SimulationConfig:
    """
    Simulation Parameters:
    N_NODES: Adjust based on CPU/GPU power. 100k is heavy but visually dense.
    ALPHA: Convergence strength toward the center.
    NOISE_STD: The 'Entropy' factor.
    """
    N_NODES = 100000
    ALPHA = 0.05          
    INTERVAL = 50          
    NOISE_STD = 0.4        
    MAX_DISPLAY = 35       
    NODE_SIZE = 0.8        # Reduced size for high-density 100k nodes
    
    # High-Contrast UI Theme
    BG_COLOR = '#000000'   
    FRAME_COLOR = '#333333'
    ACCENT = '#00FFAD'     
    TEXT_MAIN = '#FFFFFF'  

class HashSimulation2D:
    def __init__(self):
        self.config = SimulationConfig()
        np.random.seed(int(time.time() * 1000) % 2**32)

        # Initial node spread
        self.states = np.random.uniform(-9, 9, (self.config.N_NODES, 2))
        self.sha_history = []

        # Setup Figure - 22x11 Aspect Ratio for dual-pane view
        self.fig = plt.figure(figsize=(22, 11), facecolor=self.config.BG_COLOR)
        
        # --- LEFT PANEL: DATA FRAME ---
        self.ax_list = self.fig.add_axes([0.03, 0.05, 0.46, 0.9], facecolor=self.config.BG_COLOR)
        for spine in self.ax_list.spines.values():
            spine.set_visible(True)
            spine.set_color(self.config.FRAME_COLOR)
            spine.set_linewidth(1.5)
        self.ax_list.set_xticks([]); self.ax_list.set_yticks([])
        
        # Text object for SHA-256 strings
        self.hash_display = self.ax_list.text(
            0.02, 0.97, "", color=self.config.TEXT_MAIN, fontsize=9,
            family='monospace', va='top', linespacing=1.6
        )
        self.ax_list.set_title("CRYPTOGRAPHIC STATE LOG (SHA-256)", 
                              color=self.config.ACCENT, loc='left', fontsize=14, pad=15)

        # --- RIGHT PANEL: NODE DYNAMICS ---
        self.ax_nodes = self.fig.add_axes([0.51, 0.05, 0.46, 0.9], facecolor=self.config.BG_COLOR)
        for spine in self.ax_nodes.spines.values():
            spine.set_visible(True)
            spine.set_color(self.config.FRAME_COLOR)
            spine.set_linewidth(1.5)
        self.ax_nodes.set_xlim(-12, 12); self.ax_nodes.set_ylim(-12, 12)
        self.ax_nodes.set_xticks([]); self.ax_nodes.set_yticks([])
        
        self.ax_nodes.set_title("STOCHASTIC TOPOLOGY DYNAMICS", 
                               color=self.config.ACCENT, loc='right', fontsize=14, pad=15)

        # Iteration HUD
        self.count_text = self.ax_nodes.text(
            0, 11.2, "SYSTEM: INITIALIZING", color='white', 
            fontsize=13, family='monospace', ha='center', weight='bold'
        )

        # Initialize Scatter Plot
        self.scatter = self.ax_nodes.scatter(
            self.states[:, 0], self.states[:, 1], 
            c=np.linalg.norm(self.states, axis=1), 
            cmap='magma', s=self.config.NODE_SIZE, alpha=0.6, edgecolors='none'
        )

        self.running = True

    def update(self, frame):
        if not self.running:
            return self.scatter, self.hash_display, self.count_text

        # 1. Physical State Evolution
        # Formula: P = P + alpha * (mean - P) + noise
        self.states += self.config.ALPHA * (np.mean(self.states, axis=0) - self.states)
        self.states += np.random.normal(0, self.config.NOISE_STD, self.states.shape)

        # 2. Synchronized Hashing
        # We hash the raw byte-stream of the node coordinates
        current_hash = hashlib.sha256(self.states.tobytes()).hexdigest()
        
        # Detect Collisions (Impossible with SHA-256 in this context, but good for logic)
        if current_hash in self.sha_history:
            self.running = False
            self.count_text.set_text("!! STATE COLLISION DETECTED !!")
            self.count_text.set_color('#FF0055')
            return self.scatter, self.hash_display, self.count_text

        self.sha_history.append(current_hash)
        count = len(self.sha_history)

        # 3. Visual Synchronization
        self.scatter.set_offsets(self.states)
        # Update colors based on new distance from center
        self.scatter.set_array(np.linalg.norm(self.states, axis=1))
        self.count_text.set_text(f"ITERATION: {count:06d}")

        # 4. Hash Ledger Construction
        display_list = self.sha_history[-self.config.MAX_DISPLAY:]
        start_idx = count - len(display_list) + 1
        
        lines = []
        for i, h in enumerate(display_list):
            idx = start_idx + i
            if i == len(display_list) - 1: # Highlight head of the chain
                lines.append(f"[{idx:06d}] > {h}")
            else: # Dim history
                lines.append(f" {idx:06d} | {h}")

        self.hash_display.set_text("\n".join(lines))

        return self.scatter, self.hash_display, self.count_text

    def run(self):
        # cache_frame_data=False prevents memory leaks during long-running simulations
        self.ani = animation.FuncAnimation(
            self.fig, self.update, frames=None,
            interval=self.config.INTERVAL, blit=True, cache_frame_data=False
        )
        plt.show()

if __name__ == "__main__":
    sim = HashSimulation2D()
    sim.run()