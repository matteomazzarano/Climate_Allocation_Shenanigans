"""
Replication Script:
"Unpredictable Behavior in Climate Allocation Shenanigans"
Author: Matteo Mazzarano
"""

This script simulates the discrete-time replicator dynamic, computes the bifurcation
diagram and spectrum-averaged Lyapunov exponent for beta = 12.0, and saves a verification plot.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Baseline Model Calibration (Normalized Units matching Table 1)
p = 2.0      # Competitive output price
cq = 1.0     # Marginal production cost parameter
theta = 1.0  # Marginal abatement cost parameter
Q = 50.0     # Emissions cap
N = 100.0    # Total number of firms (Q/N = 0.5)
Cf = 1.25    # Fixed operational costs (balanced around gross operating margins)

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

def simulate_bifurcation(beta=12.0, k_steps=400):
    k_grid = np.linspace(0.01, 0.99, k_steps)
    T_trans = 400
    T_collect = 150
    k_pts = []
    x_pts = []
    lyap_vals = []
    
    for k in k_grid:
        x = 0.501
        for _ in range(T_trans):
            x, _ = F_and_deriv(x, k, beta)
        l_sum = 0.0
        for _ in range(T_collect):
            x, d = F_and_deriv(x, k, beta)
            k_pts.append(k)
            x_pts.append(x)
            l_sum += np.log(max(1e-12, abs(d)))
        lyap_vals.append(l_sum / T_collect)
    return k_grid, k_pts, x_pts, np.array(lyap_vals)

if __name__ == '__main__':
    print("Testing replication script for beta = 12.0...")
    k_grid, k_pts, x_pts, lyap = simulate_bifurcation(beta=12.0)
    print(f"Max Lyapunov Exponent: {np.max(lyap):.4f}")
    print(f"Mean Lyapunov Exponent: {np.mean(lyap):.4f}")
    
    # Plot and save replication figure
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.2]})
    axes[0].plot(k_pts, x_pts, ',k', alpha=0.6)
    axes[0].set_ylabel("Exit Share ($x$)")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("Replication Verification: Bifurcation Diagram ($\\beta = 12.0$)", fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.5)

    axes[1].plot(k_grid, lyap, 'r-', linewidth=1.2, label='$\\lambda(k)$')
    axes[1].axhline(0.0, color='black', linestyle='--', linewidth=0.8)
    axes[1].fill_between(k_grid, 0, lyap, where=(lyap > 0), color='gray', alpha=0.35, label='Chaos ($\\lambda > 0$)')
    axes[1].set_xlabel("Free Allocation Share ($k$)")
    axes[1].set_ylabel("Lyapunov Exp. ($\\lambda$)")
    axes[1].set_ylim(-2.5, 1.0)
    axes[1].set_xlim(0, 1)
    axes[1].grid(True, linestyle=':', alpha=0.5)
    axes[1].legend(loc='lower left', framealpha=0.8)

    plt.tight_layout()
    output_path = "replication_beta_12.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved to '{output_path}'. Replication successful.")
