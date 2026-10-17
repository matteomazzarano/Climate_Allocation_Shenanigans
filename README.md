# Replication Package
## Unpredictable Behavior in Climate Allocation Shenanigans
**Author:** Matteo Mazzarano  

---

### 1. Overview

This replication package contains all the simulation codes, mathematical models, and figure generation routines required to reproduce the analytical derivations, bifurcation diagrams, Lyapunov exponent calculations, chaos diagnostics, two-dimensional parameter basins, and social welfare analyses presented in the manuscript.

All simulations utilize the normalized baseline parameter calibration established in Table 1 of the manuscript:
- Competitive output price: $p = 2.0$
- Marginal production cost parameter: $c_q = 1.0$
- Marginal abatement cost elasticity: $\theta = 1.0$
- Emissions cap: $Q = 50.0$
- Number of industrial installations: $N = 100.0$ (Cap per installation: $Q/N = 0.50$)
- Operational fixed costs: $C_f = 1.25$
- Allocation fraction: $k \in [0, 1]$
- Intensity of choice (switching speed): $\beta \in [4.0, 20.0]$

---

### 2. File Manifest

#### Python Scripts
- **`generate_all_revised_figures.py`**:  
  Master replication script that executes the complete numerical pipeline and reproduces all seven figures of the manuscript:
  - **Figure 1**: Bifurcation diagram and Lyapunov exponent for low responsiveness ($\beta = 4.0$) (`Low_beta.jpg` / `.png`).
  - **Figure 2**: Bifurcation diagram and Lyapunov exponent for moderate responsiveness ($\beta = 8.5$) (`medium_beta.jpg` / `.png`).
  - **Figure 3**: Bifurcation diagram and Lyapunov exponent for high responsiveness ($\beta = 12.0$) (`high_beta.jpg` / `.png`).
  - **Figure 4**: Bifurcation diagram and Lyapunov exponent for very high responsiveness ($\beta = 20.0$) (`very_high_beta.jpg` / `.png`).
  - **Figure 5**: Comprehensive chaos diagnostics for $k = 0.5$: return map $F(x_t)$, empirical invariant density, exponential trajectory divergence, and Gottwald–Melbourne 0–1 test for chaos ($K = 0.998$) (`diagnostic_chaos.png`).
  - **Figure 6**: High-resolution two-dimensional Lyapunov exponent parameter basins in $(k, \beta)$ and $(C_f, c_q)$ space (`map2d_recomputed.png`).
  - **Figure 7**: Social welfare $\langle W \rangle$ and permit price $\langle \tau^* \rangle$ time-averages and variance volatility metrics across stable, periodic, and chaotic regimes (`welfare_analysis.png`).

- **`simulate_model.py`**:  
  Self-contained, lightweight Python script for fast simulation and verification of the replicator dynamic map, bifurcation diagram, and Lyapunov exponent for $\beta = 12.0$.

#### MATLAB Scripts and Live Scripts
- **`simulate_model.m`**:  
  Standalone MATLAB script simulating the discrete-time replicator map across all four choice intensity values ($\beta \in \{4.0, 8.5, 12.0, 20.0\}$) and calculating Lyapunov exponents.
- **`bifurcation_ETS_4_with_LY.mlx`**:  
  MATLAB Live Script containing analytical derivations, symbolic derivatives of the permit price and profit differential, bifurcation diagrams, and Lyapunov spectrum calculation.
- **`bifurcation_ETS_4_with_LY_extracted.m`**:  
  Plain-text MATLAB equivalent of `bifurcation_ETS_4_with_LY.mlx`.
- **`general_code_with_derivation.mlx`**:  
  MATLAB Live Script containing general model derivations, second-order optimality conditions, and total comparative statics.
- **`general_code_with_derivation_extracted.m`**:  
  Plain-text MATLAB equivalent of `general_code_with_derivation.mlx`.
- **`LE.m`**:  
  MATLAB function implementing spectrum-averaged Lyapunov exponent estimation.

---

### 3. Software Requirements and Execution Instructions

#### Running the Python Replication Pipeline
Requirements: Python 3.8+ with `numpy` and `matplotlib`.

To execute the master replication pipeline from the command line:
```bash
python generate_all_revised_figures.py
```
This automatically computes all trajectories and exports high-resolution (300 DPI) images to the current directory and to the `images/` folder.

To run the quick verification script:
```bash
python simulate_model.py
```

#### Running the MATLAB Replication Scripts
Requirements: MATLAB R2020b or later.
From the MATLAB command window, execute:
```matlab
simulate_model
```
or open and run the Live Scripts `bifurcation_ETS_4_with_LY.mlx` and `general_code_with_derivation.mlx` interactively.

---

### 4. Image Directory
All generated figure files are archived in high-resolution JPG and PNG formats in:
- `images/`
- `source/images/`
