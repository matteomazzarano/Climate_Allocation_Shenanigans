%% Replication Code: Strategic Exit and Deterministic Chaos under Mixed Allocation
%% Matteo Mazzarano
% This script simulates the discrete-time replicator dynamic and computes
% bifurcation diagrams, Lyapunov exponents, and 2D parameter basins.

clear; close all; clc;

%% 1. Baseline Model Calibration (Normalized Units)
p = 2.0;      % Competitive output price
cq = 1.0;     % Marginal production cost parameter
theta = 1.0;  % Marginal abatement cost parameter
Q = 50.0;     % Emissions cap
N = 100.0;    % Total number of firms (Q/N = 0.5)
Cf = 1.25;    % Fixed operating cost

%% 2. Function Definitions
% Explicit permit price tau*(x, k):
tau_func = @(x, k) max(0.0, (cq * theta / (cq + theta)) * (p / cq - (Q / N) * (1.0 - (1.0 - k) * x) ./ ((1.0 - x).^2)));

% Derivative of permit price wrt exit share x:
dtau_func = @(x, k) (tau_func(x, k) > 0) .* (- (cq * theta / (cq + theta)) * (Q / N) .* (1.0 + k - (1.0 - k) * x) ./ ((1.0 - x).^3));

% Profit differential DeltaPi = Pi_j* - Pi_i*:
dpi_func = @(x, k) Cf - (p - tau_func(x, k)).^2 / (2 * cq) - (tau_func(x, k)).^2 / (2 * theta);

% Derivative of profit differential wrt x:
ddpi_func = @(x, k) ((p - tau_func(x, k)) / cq - tau_func(x, k) / theta) .* dtau_func(x, k);

%% 3. Bifurcation Diagram Simulation
beta_vals = [4.0, 8.5, 12.0, 20.0];
k_grid = linspace(0.01, 0.99, 400);
T_trans = 400;
T_collect = 150;

figure('Name', 'Bifurcation Diagrams Across Intensity of Choice', 'Position', [100, 100, 1000, 800]);

for b_idx = 1:length(beta_vals)
    beta = beta_vals(b_idx);
    
    M_x = zeros(T_collect, length(k_grid));
    lyap = zeros(1, length(k_grid));
    
    for j = 1:length(k_grid)
        k = k_grid(j);
        x = 0.501; % Initial condition
        
        % Discard transient steps
        for t = 1:T_trans
            dpi = dpi_func(x, k);
            arg = max(min(-beta * dpi, 50), -50);
            x = x / (x + (1 - x) * exp(arg));
            x = max(min(x, 1 - 1e-6), 1e-6);
        end
        
        % Collect attractor points and calculate Lyapunov exponent
        l_sum = 0;
        for t = 1:T_collect
            dpi = dpi_func(x, k);
            arg = max(min(-beta * dpi, 50), -50);
            x_next = x / (x + (1 - x) * exp(arg));
            x_next = max(min(x_next, 1 - 1e-6), 1e-6);
            
            % Exact derivative of replicator map:
            dD = ddpi_func(x, k);
            deriv = x_next * (1 - x_next) * (1.0 / (x * (1 - x)) + beta * dD);
            l_sum = l_sum + log(max(1e-12, abs(deriv)));
            
            M_x(t, j) = x_next;
            x = x_next;
        end
        lyap(j) = l_sum / T_collect;
    end
    
    % Subplot: Bifurcation and Lyapunov
    subplot(2, 2, b_idx);
    plot(k_grid, M_x, '.k', 'MarkerSize', 1);
    hold on;
    title(sprintf('\\beta = %.1f', beta), 'FontWeight', 'bold');
    xlabel('Allocation share (k)');
    ylabel('Exit share (x)');
    ylim([0, 1]);
    xlim([0, 1]);
    grid on;
end

sgtitle('Bifurcation Diagrams under Mixed Allocation Mechanisms');
