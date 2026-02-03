# Sylvach: Large-Scale Distributed Consensus Benchmark

Single-file benchmark comparing relaxed over-relaxation consensus against classic random pairwise gossip.

Sylvach leverages **first-principles linear algebra and contraction mappings** to accelerate convergence. It demonstrates iteration counts largely **independent of network size** for suitable α, while pairwise gossip scales as ~O(n log n) iterations.


## 🔹 Key Features

- **Dual Consensus Methods**
  - **Random Pairwise Gossip** — classic sparse, fully decentralized baseline (pairwise averaging)
  - **Sylvach (Relaxed Over-Relaxation)** — vectorized global-mean style updates with tunable relaxation parameter α  
    → supports both over-relaxation (α > 1) and under-relaxation (α < 1)

- **Multi-State Support**
  - Scalar, vector, and symmetric matrix node states
  - Same core algorithm works for high-dimensional states (no code changes needed)

- **Large-Scale Capable**
  - Efficient NumPy vectorization
  - Comfortably handles **10k – 500k+ nodes** in scalar mode
  - Still feasible memory & runtime even for large N

- **α Parameter Sweep**
  - Test multiple relaxation factors in one benchmark run
  - Explore trade-off between convergence speed and stability

- **Residual Tracking & Convergence Monitoring**
  - Per-iteration residual:  
    ```math
    \max_i \| \mathbf{x}_i - \mathbf{x}_\text{avg} \|


🏷️ Why "Sylvach"?
The name Sylvach is a portmanteau of two pillars of mathematical analysis:

Sylvester (James Joseph Sylvester): Representing the Symmetric Matrix support and the stability criteria of the consensus states.

Banach (Stefan Banach): Representing the Fixed-Point Convergence and the contractive nature of the relaxed update rule.


Theoretical Concept,Code Implementation,Mathematical Impact
Symmetry,(A + A.T) / 2,"Ensures states remain within the ""Sylvester"" domain of valid symmetric matrices."
Contraction,X += alpha * (Xbar - X),"The ""Banach"" step that pulls every node toward the fixed-point mean."
Stability,alpha parameter,"Controls the ""aggressiveness"" of the contraction (Optimal at 1<α<2)."
