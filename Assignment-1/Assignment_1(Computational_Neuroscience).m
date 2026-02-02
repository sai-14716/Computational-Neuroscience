% ==== Assignment-1 ====
% ==== SRIKANTAM SAI SRINIVAS ====
clear; clc; close all;
mu_values = [0.1, 1, 100, 1000];
t_span = [0 200];
x_0 = [1; 0];

% ==== Numerical Solution ====
for i = 1:length(mu_values)
    mu = mu_values(i);

    % Define system depending on mu
    f = @(t, x)[mu * x(2); mu * (1 - x(1).^2) .* x(2) - (x(1)/mu)];

    % ===== ode15s =====
    tic
    [t_s, x_s] = ode15s(f, t_span, x_0);
    time_s = toc;
    fprintf("mu = %d | ode15s time = %.5f seconds\n", mu, time_s);

    figure
    plot(t_s, x_s(:,1), 'r--');
    title(['ode15s: \mu = ', num2str(mu)]);
    xlabel('Time')
    ylabel('y(t)')
    
    % ===== ode45 =====
    tic
    [t_45, x_45] = ode45(f, t_span, x_0);
    time_45 = toc;
    fprintf("mu = %d | ode45 time = %.5f seconds\n", mu, time_45);

    figure
    plot(t_45, x_45(:,1), 'r');
    title(['ode45: \mu = ', num2str(mu)])
    xlabel('Time')
    ylabel('y(t)')
end

% ==== Phase Plane ====
for i = 1:length(mu_values)
    mu = mu_values(i);
    f = @(t, x)[mu * x(2); mu * (1 - x(1).^2) .* x(2) - (x(1)/mu)];
    [x, y] = meshgrid(-10:0.5:10, -10:0.5:10);
    
    u = mu * y;
    v = mu * (1 - x.^2).*y - (x/mu);
    
    figure
    quiver(x, y, u, v, 'r');
    
    hold on;
    
    % ==== Overlaying Trajectories ====
    
    Initial_condition = [1 0; 1 2; -4 -2];
    color = ['b', 'g', 'y'];
    for i = 1:length(Initial_condition)
        ic = Initial_condition(i, :);
        [t, x] = ode45(f, t_span, ic);
        plot(x(:, 1), x(:, 2), color(i), 'LineWidth', 1.5);
    end
    
    axis tight;
    xlabel('y');
    ylabel('dy/dt * 1/\mu');
    title(['Phase plane analysis for \mu:', num2str(mu)]);
    legend({'Phase plane', 'IC:(1 0)', 'IC:(1 2)', 'IC:(-4 -2)'})
end
