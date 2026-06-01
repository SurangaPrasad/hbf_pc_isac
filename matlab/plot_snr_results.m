%% plot_snr_results.m
%  Loads SNR-curve results saved by main_SNR.py and produces a
%  publication-quality two-panel figure:
%    (a) Rate vs SNR
%    (b) CRLB vs SNR
%
%  Legend is placed below the plots. No additional legend column is used.

clearvars; close all; clc;

%% ------------------------------------------------------------------------
%  Configuration
% -------------------------------------------------------------------------
MAT_FILE = '../sim_results/64TX_4UE_4RF/snr_results_64_0.05.mat';

SAVE_FIG = true;
OUT_NAME = 'snr_results_panel';

%% ------------------------------------------------------------------------
%  Figure aesthetics
% -------------------------------------------------------------------------
FONT_NAME = 'Times New Roman';

% Larger fonts for paper readability
FONT_SIZE   = 20;     % axis tick labels
LABEL_SIZE  = 21;     % x/y labels
TITLE_SIZE  = 19;     % subplot titles
LEGEND_SIZE = 16;     % legend text

LINE_WIDTH  = 2.3;
MARKER_SIZE = 8.0;

% Larger physical figure helps preserve readable fonts in the paper
FIG_WIDTH   = 18.0;   % cm
FIG_HEIGHT  = 9.2;    % cm

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

%% ------------------------------------------------------------------------
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

%% ------------------------------------------------------------------------
%  Build series table
%
%  Format:
%  {label, rate, itheta_log, color, linestyle, marker, is_decay}
%
%  The crb_* fields are assumed to store log(I(theta)).
%  The function log_info_to_crlb converts them to CRLB = 1/I(theta).
% -------------------------------------------------------------------------
series = {};

if isfield(d,'rate_conv_PGA_J1')
    series{end+1} = {'PGA, $J=1$', ...
        get('rate_conv_PGA_J1'), get('crb_conv_PGA_J1'), ...
        COL.blue, '--', 'none', false};
end

if isfield(d,'rate_conv_PGA_J5')
    series{end+1} = {'PGA, $J=5$', ...
        get('rate_conv_PGA_J5'), get('crb_conv_PGA_J5'), ...
        COL.orange, '--', 'none', false};
end

if isfield(d,'rate_conv_PGA_J10')
    series{end+1} = {'PGA, $J=10$', ...
        get('rate_conv_PGA_J10'), get('crb_conv_PGA_J10'), ...
        COL.black, '--', 'none', false};
end

if isfield(d,'rate_UPGA_J5')
    series{end+1} = {'UPGA, $J=5$', ...
        get('rate_UPGA_J5'), get('crb_UPGA_J5'), ...
        COL.orange, '-', '*', false};
end

if isfield(d,'rate_UPGA_J10')
    series{end+1} = {'UPGA, $J=10$', ...
        get('rate_UPGA_J10'), get('crb_UPGA_J10'), ...
        COL.red, '-', '*', false};
end

if isfield(d,'rate_UPGA_J20')
    series{end+1} = {'UPGA, $J=20$', ...
        get('rate_UPGA_J20'), get('crb_UPGA_J20'), ...
        COL.red, '-', 'none', false};
end

if isfield(d,'rate_UPGA_J5_decay')
    series{end+1} = {'UPGA-d, $J_{\max}=5$', ...
        get('rate_UPGA_J5_decay'), get('crb_UPGA_J5_decay'), ...
        COL.purple, '-', '^', true};
end

if isfield(d,'rate_UPGA_J10_decay')
    series{end+1} = {'UPGA-d, $J_{\max}=10$', ...
        get('rate_UPGA_J10_decay'), get('crb_UPGA_J10_decay'), ...
        COL.purple, '-', 'd', true};
end

if isfield(d,'rate_UPGA_J20_decay')
    series{end+1} = {'UPGA-d, $J_{\max}=20$', ...
        get('rate_UPGA_J20_decay'), get('crb_UPGA_J20_decay'), ...
        COL.purple, '-', 'p', true};
end

if isempty(series)
    error('No recognised rate/crb series found in %s.', MAT_FILE);
end

%% ------------------------------------------------------------------------
%  Create figure
% -------------------------------------------------------------------------
fig = figure( ...
    'Units', 'centimeters', ...
    'Position', [2 2 FIG_WIDTH FIG_HEIGHT], ...
    'Color', 'w');

set(fig, ...
    'DefaultAxesFontName', FONT_NAME, ...
    'DefaultTextFontName', FONT_NAME, ...
    'DefaultAxesFontSize', FONT_SIZE, ...
    'DefaultTextInterpreter', 'latex', ...
    'DefaultAxesTickLabelInterpreter', 'latex', ...
    'DefaultLegendInterpreter', 'latex');

tl = tiledlayout(fig, 1, 2, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

%% ------------------------------------------------------------------------
%  Panel (a): Rate vs SNR
% -------------------------------------------------------------------------
ax1 = nexttile(tl, 1);
hold(ax1, 'on');
box(ax1, 'on');
grid(ax1, 'on');

set(ax1, ...
    'FontName', FONT_NAME, ...
    'FontSize', FONT_SIZE, ...
    'GridLineStyle', ':', ...
    'GridAlpha', 0.35, ...
    'TickDir', 'in', ...
    'LineWidth', 1.2, ...
    'Layer', 'top');

%% ------------------------------------------------------------------------
%  Panel (b): CRLB vs SNR
% -------------------------------------------------------------------------
ax2 = nexttile(tl, 2);
hold(ax2, 'on');
box(ax2, 'on');
grid(ax2, 'on');

set(ax2, ...
    'FontName', FONT_NAME, ...
    'FontSize', FONT_SIZE, ...
    'GridLineStyle', ':', ...
    'GridAlpha', 0.35, ...
    'TickDir', 'in', ...
    'LineWidth', 1.2, ...
    'Layer', 'top');

%% ------------------------------------------------------------------------
%  Plot all curves
% -------------------------------------------------------------------------
for k = 1:numel(series)

    s   = series{k};
    lbl = s{1};
    yr  = s{2};
    yc  = log_info_to_crlb(s{3});
    col = s{4};
    ls  = s{5};
    mk  = s{6};

    is_decay = logical(s{7});

    line_width_k = LINE_WIDTH;
    marker_size_k = MARKER_SIZE;
    marker_face_color_k = 'none';

    if is_decay
        line_width_k = LINE_WIDTH + 0.8;
        marker_size_k = MARKER_SIZE + 1.0;
        marker_face_color_k = col;
    end

    if strcmpi(mk, 'none')

        plot(ax1, x, yr, ...
            'LineStyle', ls, ...
            'Color', col, ...
            'LineWidth', line_width_k, ...
            'DisplayName', lbl);

        plot(ax2, x, yc, ...
            'LineStyle', ls, ...
            'Color', col, ...
            'LineWidth', line_width_k, ...
            'DisplayName', lbl);

    else

        plot(ax1, x, yr, ...
            'LineStyle', ls, ...
            'Marker', mk, ...
            'Color', col, ...
            'LineWidth', line_width_k, ...
            'MarkerSize', marker_size_k, ...
            'MarkerFaceColor', marker_face_color_k, ...
            'MarkerEdgeColor', col, ...
            'DisplayName', lbl);

        plot(ax2, x, yc, ...
            'LineStyle', ls, ...
            'Marker', mk, ...
            'Color', col, ...
            'LineWidth', line_width_k, ...
            'MarkerSize', marker_size_k, ...
            'MarkerFaceColor', marker_face_color_k, ...
            'MarkerEdgeColor', col, ...
            'DisplayName', lbl);

    end
end

%% ------------------------------------------------------------------------
%  Axis labels and titles
% -------------------------------------------------------------------------
xlabel(ax1, 'SNR [dB]', ...
    'FontSize', LABEL_SIZE, ...
    'Interpreter', 'latex');

ylabel(ax1, '$R$ [bits/s/Hz]', ...
    'FontSize', LABEL_SIZE, ...
    'Interpreter', 'latex');

title(ax1, '(a) Rate vs SNR', ...
    'FontSize', TITLE_SIZE, ...
    'FontWeight', 'normal', ...
    'Interpreter', 'latex');

xlabel(ax2, 'SNR [dB]', ...
    'FontSize', LABEL_SIZE, ...
    'Interpreter', 'latex');

ylabel(ax2, '$\mathrm{CRLB}=1/I(\theta)$', ...
    'FontSize', LABEL_SIZE, ...
    'Interpreter', 'latex');

title(ax2, '(b) CRLB vs SNR', ...
    'FontSize', TITLE_SIZE, ...
    'FontWeight', 'normal', ...
    'Interpreter', 'latex');

xlim(ax1, [min(x), max(x)]);
xlim(ax2, [min(x), max(x)]);

xticks(ax1, x);
xticks(ax2, x);

%% ------------------------------------------------------------------------
%  Optional axis limits
% -------------------------------------------------------------------------
% ylim(ax1, [10 30]);
% ylim(ax2, [0 12e-7]);

%% ------------------------------------------------------------------------
%  Shared legend below both panels
% -------------------------------------------------------------------------
h = findobj(ax1, 'Type', 'Line');
h = flipud(h);

lgd = legend(ax1, h, ...
    'Location', 'southoutside', ...
    'Orientation', 'horizontal', ...
    'NumColumns', 3, ...
    'FontSize', LEGEND_SIZE, ...
    'Interpreter', 'latex', ...
    'Box', 'off');

lgd.Layout.Tile = 'south';
lgd.ItemTokenSize = [24, 10];

lgd.Units = 'normalized';
lgd.Position(1) = 0.08;
lgd.Position(3) = 0.84;

%% ------------------------------------------------------------------------
%  Export
% -------------------------------------------------------------------------
if SAVE_FIG

    out_dir = fileparts(MAT_FILE);

    if isempty(out_dir)
        out_dir = pwd;
    end

    base = fullfile(out_dir, OUT_NAME);

    set(fig, 'PaperUnits', 'centimeters');
    set(fig, 'PaperSize', [FIG_WIDTH FIG_HEIGHT]);
    set(fig, 'PaperPosition', [0 0 FIG_WIDTH FIG_HEIGHT]);

    exportgraphics(fig, [base '.pdf'], ...
        'ContentType', 'vector');

    exportgraphics(fig, [base '.eps'], ...
        'ContentType', 'vector');

    exportgraphics(fig, [base '.png'], ...
        'Resolution', 600);

    fprintf('Saved:\n  %s.pdf\n  %s.eps\n  %s.png\n', base, base, base);
end

%% ------------------------------------------------------------------------
%  Helper functions
% -------------------------------------------------------------------------
function v = getfield_safe(s, name)

    if isfield(s, name)
        v = s.(name)(:);
    else
        v = [];
    end

end

function y = log_info_to_crlb(log_i_theta)

    % get_crb_fe stores log(I(theta)).
    % Therefore:
    %
    %     CRLB = 1 / I(theta) = exp(-log(I(theta))).

    y = nan(size(log_i_theta));
    mask = isfinite(log_i_theta);
    y(mask) = exp(-log_i_theta(mask));

end