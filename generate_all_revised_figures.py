"""
Master Replication Script:
"Unpredictable Behavior in Climate Allocation Shenanigans"
Author: Matteo Mazzarano

This script generates all revised publication figures:
- Figure 1: Bifurcation diagram and Lyapunov exponent for beta = 4.0 (Low_beta.jpg/.png)
- Figure 2: Bifurcation diagram and Lyapunov exponent for beta = 8.5 (medium_beta.jpg/.png)
- Figure 3: Bifurcation diagram and Lyapunov exponent for beta = 12.0 (high_beta.jpg/.png)
- Figure 4: Bifurcation diagram and Lyapunov exponent for beta = 20.0 (very_high_beta.jpg/.png)
- Figure 5: Comprehensive chaos diagnostics for k = 0.5 (diagnostic_chaos.png)
- Figure 6: Two-dimensional Lyapunov exponent parameter basins (map2d_recomputed.png)
- Figure 7: Social welfare and permit price volatility dynamics (welfare_analysis.png)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Determine output directory
# -------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
# Priority: if an 'images' subfolder exists or is requested, save there as well
images_dir = os.path.join(script_dir, "images")
source_dir = os.path.join(script_dir, "source")

target_dirs = [script_dir]
if os.path.isdir(images_dir):
    target_dirs.append(images_dir)
if os.path.isdir(source_dir):
    target_dirs.append(source_dir)

def save_to_all(fig, filename, dpi=300):
    for d in target_dirs:
        p = os.path.join(d, filename)
        fig.savefig(p, dpi=dpi)
    print(f"Saved: {filename} to {len(target_dirs)} locations.")

# -------------------------------------------------------------------------
# Publication Style Configuration
# -------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13
})

# -------------------------------------------------------------------------
# Baseline Parameter Vector (Normalized Units)
# -------------------------------------------------------------------------
p = 2.0      # Competitive output price
cq = 1.0     # Production cost parameter
theta = 1.0  # Abatement cost parameter
Q = 50.0     # Cap
N = 100.0    # Number of firms (Q/N = 0.5)
Cf = 1.25    # Operational fixed cost (balanced: gross operating profit = 1.25 at x=0.5)

# -------------------------------------------------------------------------
# Core Model Functions
# -------------------------------------------------------------------------
def tau_func(x, k):
    """Explicit market-clearing permit price tau*(x, k)."""
    num = 1.0 - (1.0 - k) * x
    den = (1.0 - x)**2
    H = num / den
    val = (cq * theta / (cq + theta)) * (p / cq - (Q / N) * H)
    return max(0.0, val)

def dtau_dx(x, k):
    """Derivative of permit price with respect to exit share x."""
    num = 1.0 - (1.0 - k) * x
    den = (1.0 - x)**2
    H = num / den
    val = (cq * theta / (cq + theta)) * (p / cq - (Q / N) * H)
    if val <= 0:
        return 0.0
    dH = (1.0 + k - (1.0 - k) * x) / (1.0 - x)**3
    return - (cq * theta / (cq + theta)) * (Q / N) * dH

def dDelta_dx(x, k):
    """Derivative of profit differential with respect to exit share x."""
    tau = tau_func(x, k)
    if tau <= 0:
        return 0.0
    dDelta_dtau = (p - tau) / cq - tau / theta
    return dDelta_dtau * dtau_dx(x, k)

def F_and_deriv(x, k, beta):
    """Replicator dynamic map and its exact analytical derivative."""
    tau = tau_func(x, k)
    dpi = Cf - (p - tau)**2 / (2 * cq) - tau**2 / (2 * theta)
    arg = np.clip(-beta * dpi, -50, 50)
    term = np.exp(arg)
    x_next = x / (x + (1 - x) * term)
    x_next = np.clip(x_next, 1e-6, 1 - 1e-6)
    dD = dDelta_dx(x, k)
    deriv = x_next * (1 - x_next) * (1.0 / (x * (1 - x)) + beta * dD)
    return x_next, deriv

# =========================================================================
# 1. Generate 1D Bifurcation Diagrams & Lyapunov Exponents (Figures 1-4)
# =========================================================================
def generate_bifurcations():
    beta_configs = [
        (4.0, "Low_beta.jpg", "Low Adjustment Speed ($\\beta = 4.0$)"),
        (8.5, "medium_beta.jpg", "Moderate Adjustment Speed ($\\beta = 8.5$)"),
        (12.0, "high_beta.jpg", "High Adjustment Speed ($\\beta = 12.0$)"),
        (20.0, "very_high_beta.jpg", "Very High Adjustment Speed ($\\beta = 20.0$)")
    ]

    k_grid = np.linspace(0.01, 0.99, 400)
    T_trans = 400
    T_collect = 150

    for beta, filename, title_text in beta_configs:
        print(f"Simulating bifurcation for beta = {beta}...")
        fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.2]})
        
        k_pts = []
        x_pts = []
        lyap_vals = []
        
        for k in k_grid:
            x = 0.501
            for _ in range(T_trans):
                x, _ = F_and_deriv(x, k, beta)
                
            lyap_sum = 0.0
            for _ in range(T_collect):
                x, d = F_and_deriv(x, k, beta)
                k_pts.append(k)
                x_pts.append(x)
                lyap_sum += np.log(max(1e-12, abs(d)))
                
            lyap_vals.append(lyap_sum / T_collect)
            
        # Top panel: Bifurcation diagram
        axes[0].plot(k_pts, x_pts, ',k', alpha=0.6)
        axes[0].set_ylabel("Exit Share ($x$)")
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].set_title(title_text, fontweight='bold')
        axes[0].grid(True, linestyle=':', alpha=0.5)
        
        # Bottom panel: Lyapunov exponent
        lyap_vals = np.array(lyap_vals)
        axes[1].plot(k_grid, lyap_vals, 'r-', linewidth=1.2, label='$\\lambda(k)$')
        axes[1].axhline(0.0, color='black', linestyle='--', linewidth=0.8, alpha=0.8)
        axes[1].fill_between(k_grid, 0, lyap_vals, where=(lyap_vals > 0), color='gray', alpha=0.35, label='Chaos ($\\lambda > 0$)')
        axes[1].set_xlabel("Free Allocation Share ($k$)")
        axes[1].set_ylabel("Lyapunov Exp. ($\\lambda$)")
        axes[1].set_ylim(-2.5, 1.0)
        axes[1].set_xlim(0, 1)
        axes[1].grid(True, linestyle=':', alpha=0.5)
        axes[1].legend(loc='lower left', framealpha=0.8)
        
        plt.tight_layout()
        save_to_all(fig, filename)
        save_to_all(fig, filename.replace('.jpg', '.png'))
        plt.close()

# =========================================================================
# 2. Generate Chaos Diagnostics Figure (Figure 5)
# =========================================================================
def generate_diagnostics():
    print("Generating Figure 5 (Chaos Diagnostics)...")
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7))

    # Panel A: Return map x_{t+1} vs x_t for k=0.5
    k_diag = 0.5
    x_domain = np.linspace(0.01, 0.99, 300)
    for beta, col, lab in [(4.0, 'blue', '$\\beta=4.0$ (Stable)'), (8.5, 'green', '$\\beta=8.5$ (Period-2)'), (12.0, 'red', '$\\beta=12.0$ (Chaos)')]:
        fx = [F_and_deriv(xx, k_diag, beta)[0] for xx in x_domain]
        axes[0, 0].plot(x_domain, fx, color=col, linewidth=1.5, label=lab)
    axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='$45^\\circ$ line')
    axes[0, 0].set_xlabel("Exit Share $x_t$")
    axes[0, 0].set_ylabel("Exit Share $x_{t+1}$")
    axes[0, 0].set_title("(a) Return Map $F(x_t)$ ($k=0.5$)", fontweight='bold')
    axes[0, 0].legend(loc='upper left', framealpha=0.8)
    axes[0, 0].grid(True, linestyle=':', alpha=0.5)

    # Panel B: Invariant Density for beta=12.0 (Chaos)
    x = 0.501
    beta_ch = 12.0
    for _ in range(1000):
        x, _ = F_and_deriv(x, k_diag, beta_ch)
    traj_long = []
    for _ in range(10000):
        x, _ = F_and_deriv(x, k_diag, beta_ch)
        traj_long.append(x)
    axes[0, 1].hist(traj_long, bins=60, density=True, color='crimson', alpha=0.7, edgecolor='black', linewidth=0.3)
    axes[0, 1].set_xlabel("Exit Share ($x$)")
    axes[0, 1].set_ylabel("Empirical Density")
    axes[0, 1].set_title("(b) Invariant Density ($\\beta=12.0, k=0.5$)", fontweight='bold')
    axes[0, 1].grid(True, linestyle=':', alpha=0.5)

    # Panel C: Sensitive Dependence on Initial Conditions (Trajectory Divergence)
    x1 = 0.501
    x2 = 0.501 + 1e-6
    div = []
    for _ in range(100):
        div.append(abs(x1 - x2))
        x1, _ = F_and_deriv(x1, k_diag, beta_ch)
        x2, _ = F_and_deriv(x2, k_diag, beta_ch)
    axes[1, 0].semilogy(range(len(div)), div, 'purple', linewidth=1.5)
    axes[1, 0].set_xlabel("Time Step ($t$)")
    axes[1, 0].set_ylabel("Separation $|x_t - x'_t|$ (log scale)")
    axes[1, 0].set_title("(c) Sensitive Trajectory Divergence", fontweight='bold')
    axes[1, 0].grid(True, linestyle=':', alpha=0.5)

    # Panel D: 0-1 Test for Chaos (p-q translation plot)
    c_const = 1.7
    ts_sample = np.array(traj_long[:2000])
    j_idx = np.arange(1, len(ts_sample) + 1)
    p_trans = np.cumsum(ts_sample * np.cos(j_idx * c_const))
    q_trans = np.cumsum(ts_sample * np.sin(j_idx * c_const))
    axes[1, 1].plot(p_trans, q_trans, color='navy', linewidth=0.6, alpha=0.8)
    axes[1, 1].set_xlabel("Translation $p(n)$")
    axes[1, 1].set_ylabel("Translation $q(n)$")
    axes[1, 1].set_title("(d) 0-1 Test: Diffusive $p-q$ Dynamics", fontweight='bold')
    axes[1, 1].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    save_to_all(fig, "diagnostic_chaos.png")
    plt.close()

# =========================================================================
# 3. Generate Welfare & Price Volatility Analysis (Figure 7)
# =========================================================================
def generate_welfare():
    print("Generating Figure 7 (Welfare & Volatility)...")
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7))

    k_vals_w = np.linspace(0.05, 0.95, 50)
    damage_factor = 0.5

    for beta, col, lab in [(4.0, 'blue', '$\\beta=4.0$ (Stable)'), (8.5, 'green', '$\\beta=8.5$ (Period-2)'), (12.0, 'crimson', '$\\beta=12.0$ (Chaos)')]:
        mean_w = []
        var_w = []
        mean_tau = []
        var_tau = []
        
        for k in k_vals_w:
            x = 0.501
            for _ in range(400):
                x, _ = F_and_deriv(x, k, beta)
                
            w_t = []
            tau_t = []
            for _ in range(500):
                x, _ = F_and_deriv(x, k, beta)
                tau = tau_func(x, k)
                q = (p - tau) / cq
                z = tau / theta
                E = (1 - x) * N * (q - z)
                pi_i = (p - tau)**2 / (2 * cq) + tau**2 / (2 * theta) - Cf
                total_pi = (1 - x) * N * pi_i
                rev = (1 - k) * Q * tau
                cs = 0.5 * (1 - x) * N * q**2
                W = cs + total_pi + rev - damage_factor * E
                w_t.append(W)
                tau_t.append(tau)
                
            mean_w.append(np.mean(w_t))
            var_w.append(np.var(w_t))
            mean_tau.append(np.mean(tau_t))
            var_tau.append(np.var(tau_t))
            
        axes[0, 0].plot(k_vals_w, mean_w, color=col, linewidth=1.5, label=lab)
        axes[0, 1].plot(k_vals_w, var_w, color=col, linewidth=1.5, label=lab)
        axes[1, 0].plot(k_vals_w, mean_tau, color=col, linewidth=1.5, label=lab)
        axes[1, 1].plot(k_vals_w, var_tau, color=col, linewidth=1.5, label=lab)

    axes[0, 0].set_ylabel("Mean Social Welfare $\\langle W \\rangle$")
    axes[0, 0].set_xlabel("Free Allocation Share ($k$)")
    axes[0, 0].set_title("(a) Average Social Welfare", fontweight='bold')
    axes[0, 0].legend(loc='lower right', framealpha=0.8)
    axes[0, 0].grid(True, linestyle=':', alpha=0.5)

    axes[0, 1].set_ylabel("Welfare Variance $\\mathrm{Var}(W)$")
    axes[0, 1].set_xlabel("Free Allocation Share ($k$)")
    axes[0, 1].set_title("(b) Social Welfare Volatility", fontweight='bold')
    axes[0, 1].grid(True, linestyle=':', alpha=0.5)

    axes[1, 0].set_ylabel("Mean Permit Price $\\langle \\tau^* \\rangle$")
    axes[1, 0].set_xlabel("Free Allocation Share ($k$)")
    axes[1, 0].set_title("(c) Average Permit Price", fontweight='bold')
    axes[1, 0].grid(True, linestyle=':', alpha=0.5)

    axes[1, 1].set_ylabel("Permit Price Variance $\\mathrm{Var}(\\tau^*)$")
    axes[1, 1].set_xlabel("Free Allocation Share ($k$)")
    axes[1, 1].set_title("(d) Permit Price Volatility", fontweight='bold')
    axes[1, 1].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    save_to_all(fig, "welfare_analysis.png")
    plt.close()

# =========================================================================
# 4. Generate 2D Parameter Bifurcation Basins (Figure 6)
# =========================================================================
def generate_2d_maps():
    print("Generating Figure 6 (2D Lyapunov Parameter Basins)...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Map 1: (k, beta)
    k_mesh = np.linspace(0.01, 0.99, 100)
    beta_mesh = np.linspace(1.0, 20.0, 80)
    K_grid, B_grid = np.meshgrid(k_mesh, beta_mesh)
    L_map = np.zeros_like(K_grid)

    for i in range(len(beta_mesh)):
        b_val = beta_mesh[i]
        for j in range(len(k_mesh)):
            k_val = k_mesh[j]
            x = 0.501
            for _ in range(250):
                x, _ = F_and_deriv(x, k_val, b_val)
            l_sum = 0.0
            for _ in range(150):
                x, d = F_and_deriv(x, k_val, b_val)
                l_sum += np.log(max(1e-12, abs(d)))
            L_map[i, j] = l_sum / 150.0

    cp1 = axes[0].contourf(K_grid, B_grid, L_map, levels=40, cmap='coolwarm', vmin=-1.5, vmax=0.8)
    cbar1 = plt.colorbar(cp1, ax=axes[0])
    cbar1.set_label("Lyapunov Exp. ($\\lambda$)")
    axes[0].contour(K_grid, B_grid, L_map, levels=[0.0], colors='black', linewidths=1.2)
    axes[0].set_xlabel("Allocation Share ($k$)")
    axes[0].set_ylabel("Adjustment Speed ($\\beta$)")
    axes[0].set_title("(a) Parameter Basin in $(k, \\beta)$ Space", fontweight='bold')

    # Map 2: (Cf, cq) with fixed beta=12.0, k=0.5
    cf_mesh = np.linspace(0.5, 2.0, 80)
    cq_mesh = np.linspace(0.4, 2.0, 80)
    CF_grid, CQ_grid = np.meshgrid(cf_mesh, cq_mesh)
    L_map2 = np.zeros_like(CF_grid)

    b_fixed = 12.0
    k_fixed = 0.5

    for i in range(len(cq_mesh)):
        cq_val = cq_mesh[i]
        for j in range(len(cf_mesh)):
            cf_val = cf_mesh[j]
            
            def step_local(xx):
                num = 1.0 - (1.0 - k_fixed) * xx
                den = (1.0 - xx)**2
                H = num / den
                tau_v = max(0.0, (cq_val * theta / (cq_val + theta)) * (p / cq_val - (Q / N) * H))
                dpi_v = cf_val - (p - tau_v)**2 / (2 * cq_val) - tau_v**2 / (2 * theta)
                arg = np.clip(-b_fixed * dpi_v, -50, 50)
                xx_next = xx / (xx + (1 - xx) * np.exp(arg))
                xx_next = np.clip(xx_next, 1e-6, 1 - 1e-6)
                
                if tau_v <= 0:
                    dD_v = 0.0
                else:
                    dH_v = (1.0 + k_fixed - (1.0 - k_fixed) * xx) / (1.0 - xx)**3
                    dtau_v = - (cq_val * theta / (cq_val + theta)) * (Q / N) * dH_v
                    dDelta_dtau_v = (p - tau_v) / cq_val - tau_v / theta
                    dD_v = dDelta_dtau_v * dtau_v
                    
                deriv_v = xx_next * (1 - xx_next) * (1.0 / (xx * (1 - xx)) + b_fixed * dD_v)
                return xx_next, deriv_v

            xx = 0.501
            for _ in range(250):
                xx, _ = step_local(xx)
            l_sum2 = 0.0
            for _ in range(150):
                xx, dv = step_local(xx)
                l_sum2 += np.log(max(1e-12, abs(dv)))
            L_map2[i, j] = l_sum2 / 150.0

    cp2 = axes[1].contourf(CF_grid, CQ_grid, L_map2, levels=40, cmap='coolwarm', vmin=-1.5, vmax=0.8)
    cbar2 = plt.colorbar(cp2, ax=axes[1])
    cbar2.set_label("Lyapunov Exp. ($\\lambda$)")
    axes[1].contour(CF_grid, CQ_grid, L_map2, levels=[0.0], colors='black', linewidths=1.2)
    axes[1].set_xlabel("Fixed Operational Cost ($C_f$)")
    axes[1].set_ylabel("Production Cost Parameter ($c_q$)")
    axes[1].set_title("(b) Parameter Basin in $(C_f, c_q)$ Space", fontweight='bold')

    plt.tight_layout()
    save_to_all(fig, "map2d_recomputed.png")
    plt.close()

if __name__ == '__main__':
    print("Starting full manuscript replication...")
    generate_bifurcations()
    generate_diagnostics()
    generate_welfare()
    generate_2d_maps()
    print("Manuscript replication finished successfully.")
