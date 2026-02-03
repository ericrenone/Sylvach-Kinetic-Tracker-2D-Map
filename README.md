# sylvach
Single-file benchmark comparing relaxed over-relaxation consensus against classic random pairwise gossip.

# Sylvach vs Pairwise Gossip – Large-Scale Distributed Consensus Benchmark


Sylvach leverages **first-principles linear algebra and contraction mappings** to accelerate convergence. It demonstrates iteration counts largely **independent of network size** for suitable α, while pairwise gossip scales as ~O(n log n) iterations.

---

## 🔹 Mathematical Overview

Let \(X_i \in \mathbb{R}^d\) denote the state of node \(i\), with global average:

\[
X_\text{avg} = \frac{1}{N} \sum_{i=1}^N X_i
\]

The goal of distributed consensus is for all nodes to converge to \(X_\text{avg}\), i.e.,  

\[
\lim_{t \to \infty} X_i^{(t)} = X_\text{avg}, \quad \forall i
\]

### 1️⃣ Random Pairwise Gossip

Classic gossip selects disjoint random pairs \((i,j)\) each iteration and averages them:

\[
X_i, X_j \gets \frac{X_i + X_j}{2}
\]

- Sparse, local communication  
- Iteration complexity: ~O(n log n)  
- Residual per iteration:

\[
r^{(t)} = \max_i \| X_i^{(t)} - X_\text{avg} \|
\]

```python
for i in range(0, n-1, 2):
    a, b = idx[i], idx[i+1]
    avg = (X[a] + X[b]) / 2
    X[a] = avg
    X[b] = avg
