import numpy as np
import hashlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime

# --- KINETIC ENGINE: FIXED-POINT CONTRACTION ---
SEED = 2026
np.random.seed(SEED)

class SylvachConfig:
    n_nodes = 1000
    rounds = 150
    alpha = 1.12  # Over-relaxation factor
    # Contraction Mapping Constant (Banach)
    k_constant = 0.88 

def get_potential_field(x, y):
    """Contraction Field: Represents the metric space topography."""
    r_sq = x**2 + y**2
    return -(0.5 * r_sq - 2.0 * np.cos(1.2*x) * np.cos(1.2*y))

# --- UI ARCHITECTURE (2D PHASE-SPACE) ---
fig = plt.figure(figsize=(16, 9), facecolor='#050505')
gs = fig.add_gridspec(2, 4)

# 1. PRIMARY TRACKING FIELD (Contraction Map)
ax_map = fig.add_subplot(gs[:, 0:2], facecolor='#050505')
extent = 6
x_v = np.linspace(-extent, extent, 100)
y_v = np.linspace(-extent, extent, 100)
X, Y = np.meshgrid(x_v, y_v)
Z = get_potential_field(X, Y)

# Visualizing the Fixed-Point Attractor
ax_map.contourf(X, Y, Z, levels=35, cmap='viridis', alpha=0.3)
ax_map.contour(X, Y, Z, levels=15, colors='white', alpha=0.05, linewidths=0.5)

# Node Swarm and Fixed-Point Marker
node_scat = ax_map.scatter([], [], s=3, alpha=0.6, edgecolors='none')
fixed_point = ax_map.plot([0], [0], 'r+', markersize=15, alpha=0.8, label="Fixed Point")[0]

# 2. HIGH-FLUX KINETIC HUD
# These metrics track the most critical changes in operator behavior
metrics = ["CONTRACTION VELOCITY", "PHASE COHERENCE", "STABILITY TENSION", "SHA-256 STATE DELTA"]
colors = ["#00ffcc", "#ffaa00", "#ff0055", "#aa00ff"]
axes = [fig.add_subplot(gs[i, j], facecolor='#111') for i in [0, 1] for j in [2, 3]]
lines = [ax.plot([], [], color=c, lw=1.5)[0] for ax, c in zip(axes, colors)]

for ax, title in zip(axes, metrics):
    ax.set_title(title, color='white', fontsize=9, weight='bold')
    ax.tick_params(colors='gray', labelsize=8)
    ax.grid(True, color='#222', linestyle=':')

# --- DATA LOGIC ---
states = np.random.uniform(-5, 5, (SylvachConfig.n_nodes, 2))
prev_states = states.copy()
history = {k: [] for k in metrics}

def update(frame):
    global states, prev_states
    
    # 1. BANACH CONTRACTION STEP (Sylvach-Style)
    current_mean = np.mean(states, axis=0)
    
    # Track metrics before update
    dist_to_fixed = np.linalg.norm(states, axis=1)
    prev_residual = np.mean(dist_to_fixed)
    
    # Perform the iterative mapping: X_{n+1} = (1-alpha)X_n + alpha(Mean)
    prev_states = states.copy()
    states = states + SylvachConfig.alpha * (current_mean - states)
    
    # 2. DYNAMIC METRIC EXTRACTION
    # Velocity: Magnitude of state shift
    velocity = np.mean(np.linalg.norm(states - prev_states, axis=1))
    
    # Phase Coherence: Alignment of nodes toward the Fixed Point
    # Shows if nodes are spiraling or collapsing directly
    to_center = -prev_states 
    move_dir = states - prev_states
    # Cosine similarity calculation
    cos_sim = np.sum(to_center * move_dir, axis=1) / (
        np.linalg.norm(to_center, axis=1) * np.linalg.norm(move_dir, axis=1) + 1e-9)
    coherence = np.mean(cos_sim)
    
    # Stability Tension: Standard deviation of node distribution
    tension = np.std(np.linalg.norm(states, axis=1))
    
    # SHA-256 Delta: Cryptographic fingerprint of the current operator state
    state_hash = hashlib.sha256(states.tobytes()).hexdigest()
    h_delta = int(state_hash[:6], 16) / 16777215.0

    # 3. UPDATE VISUALS
    node_scat.set_offsets(states)
    # Heatmap coloring based on proximity to attractor
    node_scat.set_color(plt.cm.viridis(1 - (dist_to_fixed / 7)))
    
    # Update HUD
    vals = [velocity, coherence, tension, h_delta]
    for i, (line, val) in enumerate(zip(lines, vals)):
        history[metrics[i]].append(val)
        line.set_data(range(len(history[metrics[i]])), history[metrics[i]])
        axes[i].set_xlim(max(0, frame - 100), frame + 5)
        recent = history[metrics[i]][-100:]
        axes[i].set_ylim(min(recent)*0.95, max(recent)*1.05 + 1e-6)

    ax_map.set_title(fr"SYLVACH 2D KINETIC TRACKER | ROUND {frame} | $\alpha$: {SylvachConfig.alpha}", 
                     color='white', fontsize=11)
    
    return [node_scat] + lines

ax_map.set_axis_off()
ani = animation.FuncAnimation(fig, update, frames=SylvachConfig.rounds, interval=40, blit=False)
plt.tight_layout()
print("[*] SYLVACH KINETIC TRACKER ACTIVE: MONITORING 1,000 NODES IN 2D PHASE-SPACE.")
plt.show()