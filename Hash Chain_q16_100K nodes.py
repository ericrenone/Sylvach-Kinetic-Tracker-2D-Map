import hashlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# ===================== CONFIGURATION =====================
N_NODES = 100_000         # 100k nodes
ALPHA_FP = 0.05           # Convergence strength
NOISE_STD_FP = 0.35       # Stochastic noise
INTERVAL = 30             # Animation interval (ms)
RADIUS_FP = 12
MAX_LOG_LINES = 40
GROUPS = 10
CENTER_AURA_MAX = 200
CENTER_AURA_MIN = 100
CENTER_AURA_SPEED = 0.05

# Q16.16 helpers
Q_SHIFT = 16
def to_fixed(val):
    return np.int32(val * (1 << Q_SHIFT))

def from_fixed(val):
    return val.astype(np.float32) / (1 << Q_SHIFT)

def fixed_add(a, b):
    return a + b

def fixed_mul(a, b):
    return np.int32((np.int64(a) * b) >> Q_SHIFT)

class LavaHashQ16_100k:
    def __init__(self):
        # --- True random seed ---
        seed = int.from_bytes(os.urandom(8), 'big') % (2**32)
        np.random.seed(seed)
        print(f"[INFO] True random seed: {seed}")

        # --- Initialize nodes in circle ---
        angles = np.random.uniform(0, 2*np.pi, N_NODES)
        radii = RADIUS_FP * np.sqrt(np.random.uniform(0, 1, N_NODES))
        x = to_fixed(radii * np.cos(angles))
        y = to_fixed(radii * np.sin(angles))
        self.states = np.column_stack((x, y)).astype(np.int32)

        self.prev_hash = b"GENESIS"
        self.history = []
        self.global_index = 0

        # --- Figure setup ---
        self.fig, (self.ax_sim, self.ax_log) = plt.subplots(
            1, 2, figsize=(18, 9),
            facecolor='#0a0a0a',
            gridspec_kw={'width_ratios': [1.5, 1]}
        )
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)

        # --- Simulation axes ---
        self.ax_sim.set_facecolor('#000000')
        self.ax_sim.set_xlim(-RADIUS_FP-5, RADIUS_FP+5)
        self.ax_sim.set_ylim(-RADIUS_FP-5, RADIUS_FP+5)
        self.ax_sim.axis('off')
        self.ax_sim.set_title("VIBRANT Q16.16 ENTROPY (100K NODES)", color='#00FFAD', fontsize=16)

        # --- Ledger axes ---
        self.ax_log.set_facecolor('#050505')
        self.ax_log.axis('off')
        self.hash_text = self.ax_log.text(
            0.05, 0.95, "", color='#00FFAD',
            family='monospace', fontsize=7, va='top', ha='left'
        )
        self.ax_log.set_title("DIGITAL LEDGER (SHA-256 CHAIN)", color='#00FFAD', fontsize=14)

        # --- Vibrant contrast colors per group ---
        colors = np.zeros(N_NODES)
        nodes_per_group = 10  # 10 nodes per group
        for g in range(GROUPS):
            for k in range(0, 1000, nodes_per_group):
                start = g*1000 + k
                end = start + nodes_per_group
                if end > N_NODES: end = N_NODES
                colors[start:end] = g / GROUPS + np.random.uniform(0, 0.05, end-start)

        # --- Initial scatter ---
        distances = np.hypot(from_fixed(self.states[:,0]), from_fixed(self.states[:,1]))
        self.scatter = self.ax_sim.scatter(
            from_fixed(self.states[:,0]),
            from_fixed(self.states[:,1]),
            s=1.5,
            c=colors + distances/np.max(distances)*0.3,
            cmap='plasma',
            edgecolors='black',
            linewidths=0.03,
            alpha=0.9,
            animated=True
        )

        # --- Center holographic invariant ---
        self.center_aura_size = CENTER_AURA_MIN
        self.center_aura_growing = True
        self.center_holo = self.ax_sim.scatter(
            0, 0,
            s=self.center_aura_size,
            c='black',
            edgecolors='cyan',
            linewidths=1.0,
            alpha=0.85,
            zorder=5
        )

        # --- Convert constants to fixed-point ---
        self.alpha = to_fixed(ALPHA_FP)
        self.noise_std = to_fixed(NOISE_STD_FP)

    def update(self, frame):
        # --- Physics in Q16.16 ---
        self.states = fixed_add(self.states, fixed_mul(self.alpha, -self.states))
        noise = np.random.normal(0, NOISE_STD_FP, self.states.shape)
        self.states = fixed_add(self.states, to_fixed(noise))

        # --- Update scatter ---
        distances = np.hypot(from_fixed(self.states[:,0]), from_fixed(self.states[:,1]))
        self.scatter.set_offsets(np.column_stack((from_fixed(self.states[:,0]), from_fixed(self.states[:,1]))))
        colors = self.scatter.get_array() + np.random.uniform(0, 0.003, N_NODES)
        self.scatter.set_array(colors)

        # --- Hash chain ---
        raw_data = self.states.tobytes() + self.prev_hash
        current_hash = hashlib.sha256(raw_data).hexdigest()
        self.prev_hash = current_hash.encode()

        # --- Update ledger ---
        self.history.append((self.global_index, current_hash))
        self.global_index += 1
        if len(self.history) > MAX_LOG_LINES:
            self.history = self.history[-MAX_LOG_LINES:]
        log_display = "\n".join([f"[{idx:06d}] {h}" for idx, h in self.history])
        self.hash_text.set_text(log_display)

        # --- Animate holographic center aura ---
        if self.center_aura_growing:
            self.center_aura_size += CENTER_AURA_SPEED * (CENTER_AURA_MAX - CENTER_AURA_MIN)
            if self.center_aura_size >= CENTER_AURA_MAX:
                self.center_aura_growing = False
        else:
            self.center_aura_size -= CENTER_AURA_SPEED * (CENTER_AURA_MAX - CENTER_AURA_MIN)
            if self.center_aura_size <= CENTER_AURA_MIN:
                self.center_aura_growing = True
        self.center_holo.set_sizes([self.center_aura_size])

        return self.scatter, self.hash_text, self.center_holo

    def run(self):
        self.ani = animation.FuncAnimation(
            self.fig, self.update, interval=INTERVAL,
            blit=True, cache_frame_data=False
        )
        plt.show()

if __name__ == "__main__":
    sim = LavaHashQ16_100k()
    sim.run()
