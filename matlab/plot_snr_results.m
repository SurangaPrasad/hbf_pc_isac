% Plot SNR-sweep figures from Python cached MAT file.
% Uses same curves, colors and markers as main_SNR.py.

clear; clc; close all;

% --------------------------- User settings ---------------------------
Nt = 64;
OMEGA = 0.25;
system_config = sprintf('%dTX_4UE_4RF', Nt);
result_dir = fullfile('..', 'sim_results', system_config);
cache_file = fullfile(result_dir, sprintf('snr_plot_cache_%d_%g.mat', Nt, OMEGA));

if ~isfile(cache_file)
    error('SNR cache file not found: %s\nRun main_SNR.py once with run_program=1 first.', cache_file);
end
S = load(cache_file);

snr_dB_list = as_row(S.snr_dB_list);

% Colors (same semantic mapping as Python)
black  = [0.00 0.00 0.00];
blue   = [0.00 0.00 1.00];
red    = [1.00 0.00 0.00];
green  = [0.00 0.50 0.00];
orange = [1.00 0.55 0.00];

% ========================== Rate vs SNR ==========================
figure('Color', 'w', 'Units', 'inches', 'Position', [1.5 1.5 8 4.2]);
hold on; grid on;

if isfield(S, 'rate_conv_PGA')
    plot(snr_dB_list, as_row(S.rate_conv_PGA), '--', 'Color', black, 'LineWidth', 3, 'MarkerSize', 8, 'DisplayName', 'Conv. PGA J1');
end
if isfield(S, 'rate_conv_PGA_J5')
    plot(snr_dB_list, as_row(S.rate_conv_PGA_J5), '--', 'Color', blue, 'LineWidth', 3, 'MarkerSize', 8, 'DisplayName', 'Conv. PGA J5');
end
if isfield(S, 'rate_conv_PGA_J10')
    plot(snr_dB_list, as_row(S.rate_conv_PGA_J10), '-*', 'Color', blue, 'LineWidth', 3, 'MarkerSize', 8, 'DisplayName', 'Conv. PGA J10');
end

if isfield(S, 'rate_UPGA_J4')
    plot(snr_dB_list, as_row(S.rate_UPGA_J4), '--', 'Color', orange, 'LineWidth', 3, 'MarkerSize', 8, 'DisplayName', 'UPGA J4');
end
if isfield(S, 'rate_UPGA_J5')
    plot(snr_dB_list, as_row(S.rate_UPGA_J5), '--', 'Color', red, 'LineWidth', 3, 'MarkerSize', 8, 'DisplayName', 'UPGA J5');
end
if isfield(S, 'rate_UPGA_J6')
    plot(snr_dB_list, as_row(S.rate_UPGA_J6), '-d', 'Color', orange, 'LineWidth', 3, 'MarkerSize', 8, 'DisplayName', 'UPGA J6');
end
if isfield(S, 'rate_UPGA_J10')
    plot(snr_dB_list, as_row(S.rate_UPGA_J10), '-*', 'Color', red, 'LineWidth', 3, 'MarkerSize', 8, 'DisplayName', 'UPGA J10');
end
if isfield(S, 'rate_UPGA_J20')
    plot(snr_dB_list, as_row(S.rate_UPGA_J20), '-*', 'Color', red, 'LineWidth', 3, 'MarkerSize', 8, 'DisplayName', 'UPGA J20');
end

if isfield(S, 'rate_UPGA_J5_decay')
    plot(snr_dB_list, as_row(S.rate_UPGA_J5_decay), '--', 'Color', green, 'LineWidth', 3, 'MarkerSize', 8, 'DisplayName', 'UPGA J5 decay');
end
if isfield(S, 'rate_UPGA_J10_decay')
    plot(snr_dB_list, as_row(S.rate_UPGA_J10_decay), '-*', 'Color', green, 'LineWidth', 3, 'MarkerSize', 8, 'DisplayName', 'UPGA J10 decay');
end
if isfield(S, 'rate_UPGA_J20_decay')
    plot(snr_dB_list, as_row(S.rate_UPGA_J20_decay), ':p', 'Color', green, 'LineWidth', 3, 'MarkerSize', 8, 'DisplayName', 'UPGA J20 decay');
end

xlabel('SNR [dB]', 'FontSize', 14);
ylabel('R [bits/s/Hz]', 'FontSize', 14);
set(gca, 'FontSize', 12);
legend('Location', 'best');

rate_png = fullfile(result_dir, sprintf('rate_vs_SNR_%d_%g_matlab.png', Nt, OMEGA));
rate_eps = fullfile(result_dir, sprintf('rate_vs_SNR_%d_%g_matlab.eps', Nt, OMEGA));
exportgraphics(gcf, rate_png, 'Resolution', 300);
exportgraphics(gcf, rate_eps, 'ContentType', 'vector');

% ========================== CRLB vs SNR ==========================
figure('Color', 'w', 'Units', 'inches', 'Position', [1.5 1.5 8 4.2]);
ax = gca;
hold(ax, 'on');
grid(ax, 'on');

curves = {};
if isfield(S, 'CRB_conv_PGA')
    curves{end+1} = {snr_dB_list, crlb_from_log_inv(as_row(S.CRB_conv_PGA)), '--', black, 'Conv. PGA J1'}; %#ok<AGROW>
end
if isfield(S, 'CRB_conv_PGA_J5')
    curves{end+1} = {snr_dB_list, crlb_from_log_inv(as_row(S.CRB_conv_PGA_J5)), '--', blue, 'Conv. PGA J5'}; %#ok<AGROW>
end
if isfield(S, 'CRB_conv_PGA_J10')
    curves{end+1} = {snr_dB_list, crlb_from_log_inv(as_row(S.CRB_conv_PGA_J10)), '-*', blue, 'Conv. PGA J10'}; %#ok<AGROW>
end

if isfield(S, 'CRB_UPGA_J4')
    curves{end+1} = {snr_dB_list, crlb_from_log_inv(as_row(S.CRB_UPGA_J4)), '--', orange, 'UPGA J4'}; %#ok<AGROW>
end
if isfield(S, 'CRB_UPGA_J5')
    curves{end+1} = {snr_dB_list, crlb_from_log_inv(as_row(S.CRB_UPGA_J5)), '--', red, 'UPGA J5'}; %#ok<AGROW>
end
if isfield(S, 'CRB_UPGA_J6')
    curves{end+1} = {snr_dB_list, crlb_from_log_inv(as_row(S.CRB_UPGA_J6)), '-d', orange, 'UPGA J6'}; %#ok<AGROW>
end
if isfield(S, 'CRB_UPGA_J10')
    curves{end+1} = {snr_dB_list, crlb_from_log_inv(as_row(S.CRB_UPGA_J10)), '-*', red, 'UPGA J10'}; %#ok<AGROW>
end
if isfield(S, 'CRB_UPGA_J20')
    curves{end+1} = {snr_dB_list, crlb_from_log_inv(as_row(S.CRB_UPGA_J20)), ':s', red, 'UPGA J20'}; %#ok<AGROW>
end

if isfield(S, 'CRB_UPGA_J5_decay')
    curves{end+1} = {snr_dB_list, crlb_from_log_inv(as_row(S.CRB_UPGA_J5_decay)), '--', green, 'UPGA J5 decay'}; %#ok<AGROW>
end
if isfield(S, 'CRB_UPGA_J10_decay')
    curves{end+1} = {snr_dB_list, crlb_from_log_inv(as_row(S.CRB_UPGA_J10_decay)), '-*', green, 'UPGA J10 decay'}; %#ok<AGROW>
end
if isfield(S, 'CRB_UPGA_J20_decay')
    curves{end+1} = {snr_dB_list, crlb_from_log_inv(as_row(S.CRB_UPGA_J20_decay)), ':p', green, 'UPGA J20 decay'}; %#ok<AGROW>
end

for i = 1:numel(curves)
    c = curves{i};
    plot(ax, c{1}, c{2}, c{3}, 'Color', c{4}, 'LineWidth', 3, 'MarkerSize', 7, 'DisplayName', c{5});
end

xlabel(ax, 'SNR [dB]', 'FontSize', 14);
ylabel(ax, 'CRLB', 'FontSize', 14);
set(ax, 'FontSize', 12);

% Inset zoom (10 to 12 dB)
axins = axes('Position', [0.58 0.52 0.30 0.32]);
hold(axins, 'on');
grid(axins, 'on');
for i = 1:numel(curves)
    c = curves{i};
    plot(axins, c{1}, c{2}, c{3}, 'Color', c{4}, 'LineWidth', 2, 'MarkerSize', 5);
end
xlim(axins, [10 12]);

zoom_vals = [];
for i = 1:numel(curves)
    c = curves{i};
    mask = (c{1} >= 10) & (c{1} <= 12);
    if any(mask)
        zoom_vals = [zoom_vals, c{2}(mask)]; %#ok<AGROW>
    end
end
if ~isempty(zoom_vals)
    ymin = min(zoom_vals);
    ymax = max(zoom_vals);
    ypad = 0.15 * (ymax - ymin);
    ylim(axins, [ymin - ypad, ymax + ypad]);
end

legend(ax, 'Location', 'southoutside', 'NumColumns', 2, 'Box', 'off');

crb_png = fullfile(result_dir, sprintf('CRB_vs_SNR_%d_%g_matlab.png', Nt, OMEGA));
crb_eps = fullfile(result_dir, sprintf('CRB_vs_SNR_%d_%g_matlab.eps', Nt, OMEGA));
exportgraphics(gcf, crb_png, 'Resolution', 300);
exportgraphics(gcf, crb_eps, 'ContentType', 'vector');

fprintf('Saved MATLAB SNR plots to:\n  %s\n  %s\n  %s\n  %s\n', rate_png, rate_eps, crb_png, crb_eps);

% ============================ Helpers ============================
function x = as_row(v)
x = v(:).';
end

function y = crlb_from_log_inv(x)
y = 1 ./ exp(x);
end
