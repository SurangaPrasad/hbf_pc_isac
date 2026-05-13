%% plot_snr_results.m
%  Loads SNR-curve results saved by main_SNR.py and produces a
%  publication-quality two-panel figure:
%    (a) Rate vs SNR
%    (b) I(theta) vs SNR
%
%  Usage: run from the matlab/ folder, or adjust MAT_FILE accordingly.

clearvars; close all; clc;

% -------------------------------------------------------------------------
%  Configuration
% -------------------------------------------------------------------------
MAT_FILE = '../sim_results/64TX_4UE_4RF/snr_results_64_0.05.mat';

% -------------------------------------------------------------------------
%  Figure aesthetics
% -------------------------------------------------------------------------
FONT_SIZE   = 10;
LEGEND_SIZE = 9;
LINE_WIDTH  = 1.7;
MARKER_SIZE = 5;
FIG_WIDTH   = 14;   % cm
FIG_HEIGHT  = 5.2;  % cm

COL = struct( ...
    'blue',   [0.00 0.45 0.70], ...
    'orange', [0.90 0.62 0.00], ...
    'green',  [0.00 0.62 0.45], ...
    'red',    [0.84 0.15 0.16], ...
    'purple', [0.49 0.18 0.56], ...
    'cyan',   [0.34 0.71 0.91], ...
    'teal',   [0.00 0.62 0.62], ...
    'brown',  [0.55 0.34 0.29], ...
    'black',  [0.00 0.00 0.00]  ...
);

% -------------------------------------------------------------------------
%  Load data
% -------------------------------------------------------------------------
if ~isfile(MAT_FILE)
    error('File not found: %s\nRun main_SNR.py first.', MAT_FILE);
end
d = load(MAT_FILE);

if ~isfield(d, 'snr_dB_list')
    error('snr_dB_list is missing in %s.', MAT_FILE);
end
x = d.snr_dB_list(:);

get = @(name) getfield_safe(d, name);

% -------------------------------------------------------------------------
%  Build series table: {label, rate, itheta, color, linestyle, marker}
%  Here I(theta) is read from crb_* fields saved in main_SNR.py.
% -------------------------------------------------------------------------
series = {};

if isfield(d,'rate_conv_PGA_J1')
    series{end+1} = {'PGA ($J\!=\!1$)',         get('rate_conv_PGA_J1'),      get('crb_conv_PGA_J1'),      COL.blue,   '--', 'none'};
end
if isfield(d,'rate_conv_PGA_J10')
    series{end+1} = {'PGA ($J\!=\!10$)',        get('rate_conv_PGA_J10'),     get('crb_conv_PGA_J10'),     COL.black,  '--', '*'};
end
if isfield(d,'rate_UPGA_J1')
    series{end+1} = {'UPGA ($J\!=\!1$)',        get('rate_UPGA_J1'),          get('crb_UPGA_J1'),          COL.cyan,   '-',  'o'};
end
if isfield(d,'rate_UPGA_J10')
    series{end+1} = {'UPGA ($J\!=\!10$)',       get('rate_UPGA_J10'),         get('crb_UPGA_J10'),         COL.red,    ':',  '*'};
end
if isfield(d,'rate_UPGA_J20')
    series{end+1} = {'UPGA ($J\!=\!20$)',       get('rate_UPGA_J20'),         get('crb_UPGA_J20'),         COL.red,    '-',  'none'};
end
if isfield(d,'rate_UPGA_J10_decay')
    series{end+1} = {'UPGA ($J\!=\!10$, decay)',get('rate_UPGA_J10_decay'),   get('crb_UPGA_J10_decay'),   COL.purple, ':',  'd'};
end
if isfield(d,'rate_UPGA_J20_decay')
    series{end+1} = {'UPGA ($J\!=\!20$, decay)',get('rate_UPGA_J20_decay'),   get('crb_UPGA_J20_decay'),   COL.brown,  '-.', 'p'};
end
if isfield(d,'rate_UPGA_J_GradReuse')
    series{end+1} = {'UPGA GradReuse ($J\!=\!10$)', get('rate_UPGA_J_GradReuse'), get('crb_UPGA_J_GradReuse'), COL.teal, ':', '^'};
end

if isempty(series)
    error('No recognised rate/crb series found in %s.', MAT_FILE);
end

% -------------------------------------------------------------------------
%  Create figure
% -------------------------------------------------------------------------
fig = figure('Units','centimeters','Position',[2 2 FIG_WIDTH FIG_HEIGHT]);
set(fig,'defaultAxesFontName','Times New Roman', ...
        'defaultAxesFontSize', FONT_SIZE, ...
        'defaultTextInterpreter','latex', ...
        'defaultAxesTickLabelInterpreter','latex', ...
        'defaultLegendInterpreter','latex');

ax1 = subplot(1,2,1);
hold(ax1,'on'); box(ax1,'on'); grid(ax1,'on');
set(ax1, 'GridLineStyle', ':', 'GridAlpha', 0.4, 'TickDir', 'in', 'LineWidth', 0.8);

ax2 = subplot(1,2,2);
hold(ax2,'on'); box(ax2,'on'); grid(ax2,'on');
set(ax2, 'GridLineStyle', ':', 'GridAlpha', 0.4, 'TickDir', 'in', 'LineWidth', 0.8);

for k = 1:numel(series)
    s   = series{k};
    lbl = s{1};
    yr  = s{2};
    yi  = s{3};
    col = s{4};
    ls  = s{5};
    mk  = s{6};

    if strcmp(mk,'none')
        plot(ax1, x, yr, ls, 'Color', col, 'LineWidth', LINE_WIDTH, 'DisplayName', lbl);
        plot(ax2, x, yi, ls, 'Color', col, 'LineWidth', LINE_WIDTH, 'DisplayName', lbl);
    else
        plot(ax1, x, yr, [ls mk], 'Color', col, 'LineWidth', LINE_WIDTH, ...
            'MarkerSize', MARKER_SIZE, 'MarkerFaceColor', 'none', 'DisplayName', lbl);
        plot(ax2, x, yi, [ls mk], 'Color', col, 'LineWidth', LINE_WIDTH, ...
            'MarkerSize', MARKER_SIZE, 'MarkerFaceColor', 'none', 'DisplayName', lbl);
    end
end

xlabel(ax1, 'SNR [dB]');
ylabel(ax1, '$R$ [bits/s/Hz]');
title(ax1, '(a) Rate vs SNR', 'FontWeight', 'normal');

xlabel(ax2, 'SNR [dB]');
ylabel(ax2, '$I(\theta)$');
title(ax2, '(b) $I(\theta)$ vs SNR', 'FontWeight', 'normal');

% Shared legend under both panels
lgd = legend(ax1, 'Location','southoutside', 'Orientation','horizontal', ...
    'NumColumns', min(numel(series), 4), 'FontSize', LEGEND_SIZE, 'Box', 'on');
lgd.Position(1) = (1 - lgd.Position(3)) / 2;
lgd.Position(2) = 0.01;

% Leave room for legend
set(fig, 'Units', 'normalized');
ax1.Position(2) = 0.28; ax1.Position(4) = 0.62;
ax2.Position(2) = 0.28; ax2.Position(4) = 0.62;

% -------------------------------------------------------------------------
%  Export
% -------------------------------------------------------------------------
out_dir = fileparts(MAT_FILE);
base = fullfile(out_dir, 'snr_results_panel');

print(fig, base, '-dpdf',  '-r300', '-bestfit');
print(fig, base, '-depsc', '-r300');
fprintf('Saved:\n  %s.pdf\n  %s.eps\n', base, base);

% -------------------------------------------------------------------------
%  Helper
% -------------------------------------------------------------------------
function v = getfield_safe(s, name)
    if isfield(s, name)
        v = s.(name)(:);
    else
        v = [];
    end
end
