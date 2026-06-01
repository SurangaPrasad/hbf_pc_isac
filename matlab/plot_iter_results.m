%% plot_iter_results_publication.m
% Publication-quality three-panel plot:
% (a) Objective Function, (b) Sum Rate, (c) CRLB^{-1}
%
% This script loads the MAT file produced by your simulation and plots
% the convergence curves in a paper-ready format.

clearvars; close all; clc;

%% ------------------------------------------------------------------------
%  User configuration
% -------------------------------------------------------------------------
MAT_FILE = '../sim_results/64TX_4UE_4RF/iter_results_64_0.05.mat';

SAVE_FIG = true;
OUT_NAME = 'iter_results_publication';

%% ------------------------------------------------------------------------
%  Figure style for research paper
% -------------------------------------------------------------------------
FONT_NAME = 'Times New Roman';

% Larger font sizes for paper readability
FONT_SIZE   = 14;     % tick labels
LABEL_SIZE  = 15;     % x/y labels
TITLE_SIZE  = 14;     % subplot titles
LEGEND_SIZE = 13;     % legend text

LINE_WIDTH  = 1.8;
MARKER_SIZE = 6.5;
MARK_EVERY  = 6;

% Double-column paper figure size
FIG_WIDTH_CM  = 18.0;
FIG_HEIGHT_CM = 8.2;

% Colorblind-friendly, print-safe palette
COL.blue   = [0.000, 0.447, 0.741];
COL.orange = [0.850, 0.325, 0.098];
COL.yellow = [0.929, 0.694, 0.125];
COL.purple = [0.494, 0.184, 0.556];
COL.green  = [0.466, 0.674, 0.188];
COL.cyan   = [0.301, 0.745, 0.933];
COL.red    = [0.635, 0.078, 0.184];
COL.black  = [0.100, 0.100, 0.100];

%% ------------------------------------------------------------------------
%  Load data
% -------------------------------------------------------------------------
if ~isfile(MAT_FILE)
    error('MAT file not found: %s', MAT_FILE);
end

d = load(MAT_FILE);

if ~isfield(d, 'iter_outer_x')
    error('The MAT file does not contain iter_outer_x.');
end

x = d.iter_outer_x(:);

get = @(name) getfield_safe(d, name);

%% ------------------------------------------------------------------------
%  Define plotted series
%
%  Format:
%  {label, x, rate, crb, obj, color, linestyle, marker, is_highlight}
% -------------------------------------------------------------------------
series = {};

if isfield(d,'rate_conv_PGA_J1')
    series{end+1} = {'PGA, $J=1$', ...
        x, get('rate_conv_PGA_J1'), get('crb_conv_PGA_J1'), get('obj_conv_PGA_J1'), ...
        COL.blue, '--', 'none', false};
end

if isfield(d,'rate_conv_PGA_J5')
    series{end+1} = {'PGA, $J=5$', ...
        x, get('rate_conv_PGA_J5'), get('crb_conv_PGA_J5'), get('obj_conv_PGA_J5'), ...
        COL.orange, '-.', 'none', false};
end

if isfield(d,'rate_conv_PGA_J10')
    series{end+1} = {'PGA, $J=10$', ...
        x, get('rate_conv_PGA_J10'), get('crb_conv_PGA_J10'), get('obj_conv_PGA_J10'), ...
        COL.black, '--', '*', false};
end

if isfield(d,'rate_UPGA_J1')
    series{end+1} = {'UPGA, $J=1$', ...
        x, get('rate_UPGA_J1'), get('crb_UPGA_J1'), get('obj_UPGA_J1'), ...
        COL.cyan, '-', 'o', false};
end

if isfield(d,'rate_UPGA_J5')
    series{end+1} = {'UPGA, $J=5$', ...
        x, get('rate_UPGA_J5'), get('crb_UPGA_J5'), get('obj_UPGA_J5'), ...
        COL.yellow, '-', 'd', false};
end

if isfield(d,'rate_UPGA_J10')
    series{end+1} = {'UPGA, $J=10$', ...
        x, get('rate_UPGA_J10'), get('crb_UPGA_J10'), get('obj_UPGA_J10'), ...
        COL.red, ':', '*', false};
end

if isfield(d,'rate_UPGA_J5_decay')
    x5d = d.iter_outer_x_J5_decay(:);
    series{end+1} = {'UPGA-decay, $J_{\max}=5$', ...
        x5d, get('rate_UPGA_J5_decay'), get('crb_UPGA_J5_decay'), get('obj_UPGA_J5_decay'), ...
        COL.purple, '-', '^', true};
end

if isfield(d,'rate_UPGA_J10_decay')
    x10d = d.iter_outer_x_J10_decay(:);
    series{end+1} = {'UPGA-decay, $J_{\max}=10$', ...
        x10d, get('rate_UPGA_J10_decay'), get('crb_UPGA_J10_decay'), get('obj_UPGA_J10_decay'), ...
        COL.purple, '-', 'd', true};
end

if isempty(series)
    error('No recognized plotting fields found in the MAT file.');
end

%% ------------------------------------------------------------------------
%  Create figure
% -------------------------------------------------------------------------
fig = figure( ...
    'Units', 'centimeters', ...
    'Position', [2 2 FIG_WIDTH_CM FIG_HEIGHT_CM], ...
    'Color', 'w');

set(fig, ...
    'DefaultAxesFontName', FONT_NAME, ...
    'DefaultTextFontName', FONT_NAME, ...
    'DefaultAxesFontSize', FONT_SIZE, ...
    'DefaultAxesTickLabelInterpreter', 'latex', ...
    'DefaultTextInterpreter', 'latex', ...
    'DefaultLegendInterpreter', 'latex');

tl = tiledlayout(fig, 1, 3, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

titles = {'(a) Objective Function', ...
          '(b) Sum Rate', ...
          '(c) $CRLB^{-1}$'};

ylabs = {'$\omega R - CRLB^{-1}$', ...
         '$R$ [bits/s/Hz]', ...
         '$CRLB^{-1}$'};

% series columns: 3 = rate, 4 = crb, 5 = objective
data_id = [5, 3, 4];

ax = gobjects(1,3);

for p = 1:3

    ax(p) = nexttile(tl);
    hold(ax(p), 'on');
    box(ax(p), 'on');
    grid(ax(p), 'on');

    set(ax(p), ...
        'FontName', FONT_NAME, ...
        'FontSize', FONT_SIZE, ...
        'LineWidth', 1.0, ...
        'TickDir', 'in', ...
        'XMinorTick', 'off', ...
        'YMinorTick', 'off', ...
        'GridLineStyle', ':', ...
        'GridAlpha', 0.35, ...
        'Layer', 'top');

    for k = 1:numel(series)

        s = series{k};

        label = s{1};
        xk    = s{2}(:);
        yk    = s{data_id(p)}(:);
        col   = s{6};
        ls    = s{7};
        mk    = s{8};
        high  = s{9};

        % For CRLB^{-1}, convert from log-domain if your file stores log values.
        % If your crb_* values are already CRLB^{-1}, replace the next block with:
        % yk = yk;
        if p == 3
            yk = log_info_to_info(yk);
        end

        lw = LINE_WIDTH;
        ms = MARKER_SIZE;
        marker_face = 'w';

        if high
            lw = LINE_WIDTH + 0.55;
            ms = MARKER_SIZE + 0.8;
            marker_face = col;
        end

        marker_idx = 1:MARK_EVERY:numel(xk);

        if strcmpi(mk, 'none')
            plot(ax(p), xk, yk, ...
                'LineStyle', ls, ...
                'Color', col, ...
                'LineWidth', lw, ...
                'DisplayName', label);
        else
            plot(ax(p), xk, yk, ...
                'LineStyle', ls, ...
                'Marker', mk, ...
                'Color', col, ...
                'LineWidth', lw, ...
                'MarkerSize', ms, ...
                'MarkerIndices', marker_idx, ...
                'MarkerFaceColor', marker_face, ...
                'MarkerEdgeColor', col, ...
                'DisplayName', label);
        end
    end

    xlabel(ax(p), 'Iteration number $(I)$', ...
        'FontSize', LABEL_SIZE, ...
        'Interpreter', 'latex');

    ylabel(ax(p), ylabs{p}, ...
        'FontSize', LABEL_SIZE, ...
        'Interpreter', 'latex');

    title(ax(p), titles{p}, ...
        'FontSize', TITLE_SIZE, ...
        'FontWeight', 'normal', ...
        'Interpreter', 'latex');

    xlim(ax(p), [min(x), max(x)]);
end

%% ------------------------------------------------------------------------
%  Optional axis limits
%  Uncomment and adjust if you want fixed axes matching your paper.
% -------------------------------------------------------------------------
% ylim(ax(1), [16.5 19.0]);
% ylim(ax(2), [10 30]);
% ylim(ax(3), [1.0e7 3.5e7]);

% Scientific notation for CRLB^{-1}
ax(3).YAxis.Exponent = 7;

%% ------------------------------------------------------------------------
%  Shared legend — larger, cleaner, no clipping
% -------------------------------------------------------------------------
h = findobj(ax(1), 'Type', 'Line');
h = flipud(h);

lgd = legend(ax(1), h, ...
    'Orientation', 'horizontal', ...
    'Location', 'southoutside', ...
    'NumColumns', 3, ...
    'FontSize', LEGEND_SIZE, ...
    'Interpreter', 'latex', ...
    'Box', 'off');

lgd.Layout.Tile = 'south';
lgd.ItemTokenSize = [22, 10];

%% ------------------------------------------------------------------------
%  Export high-quality files
% -------------------------------------------------------------------------
if SAVE_FIG

    out_dir = fileparts(MAT_FILE);

    if isempty(out_dir)
        out_dir = pwd;
    end

    pdf_file = fullfile(out_dir, [OUT_NAME '.pdf']);
    eps_file = fullfile(out_dir, [OUT_NAME '.eps']);
    png_file = fullfile(out_dir, [OUT_NAME '.png']);

    set(fig, 'PaperUnits', 'centimeters');
    set(fig, 'PaperSize', [FIG_WIDTH_CM FIG_HEIGHT_CM]);
    set(fig, 'PaperPosition', [0 0 FIG_WIDTH_CM FIG_HEIGHT_CM]);

    exportgraphics(fig, pdf_file, 'ContentType', 'vector');
    exportgraphics(fig, eps_file, 'ContentType', 'vector');
    exportgraphics(fig, png_file, 'Resolution', 600);

    fprintf('Saved:\n%s\n%s\n%s\n', pdf_file, eps_file, png_file);
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

function y = log_info_to_info(log_val)

    % Use this function only if the stored CRB-related values are log(CRLB^{-1}).
    % If your data already stores CRLB^{-1}, replace the body of this function by:
    %
    % y = log_val;

    y = nan(size(log_val));
    idx = isfinite(log_val);
    y(idx) = exp(log_val(idx));

end