%% plot_iter_results.m
%  Loads the outer-iteration results saved by main_iter.py and produces a
%  publication-quality three-panel figure (Objective | Rate | CRB).
%
%  Usage: run from the matlab/ folder, or adjust MAT_FILE accordingly.

clearvars; close all; clc;

% -------------------------------------------------------------------------
%  Configuration – edit these to match your Python run
% -------------------------------------------------------------------------
MAT_FILE = '../sim_results/64TX_4UE_4RF/iter_results_64_0.05.mat';
OMEGA    = 0.05;          % weighting factor (must match Python run)

% -------------------------------------------------------------------------
%  Figure aesthetics
% -------------------------------------------------------------------------
FONT_SIZE   = 10;         % axis tick / label font size
LEGEND_SIZE = 8;
LINE_WIDTH  = 1.1;
MARKER_SIZE = 6;
FIG_WIDTH   = 18;         % cm
FIG_HEIGHT  = 5.5;        % cm
MARK_EVERY  = 5;          % plot a marker every N data points

% Colour palette (distinguishable, print-safe)
COL = struct( ...
    'blue',   [0.00 0.45 0.70], ...
    'orange', [0.90 0.62 0.00], ...
    'green',  [0.00 0.62 0.45], ...
    'red',    [0.84 0.15 0.16], ...
    'purple', [0.49 0.18 0.56], ...
    'cyan',   [0.34 0.71 0.91], ...
    'teal',   [0.00 0.62 0.62], ...
    'black',  [0.00 0.00 0.00]  ...
);

% -------------------------------------------------------------------------
%  Load data
% -------------------------------------------------------------------------
if ~isfile(MAT_FILE)
    error('File not found: %s\nRun main_iter.py first.', MAT_FILE);
end
d = load(MAT_FILE);
x = d.iter_outer_x;        % common x-axis  [1 x n_outer]

% Helper: safely extract a field, returning [] if absent
get = @(name) getfield_safe(d, name);

% -------------------------------------------------------------------------
%  Build series table  {label, x, rate, crb, obj, color, linestyle, marker}
% -------------------------------------------------------------------------
series = {};

if isfield(d,'rate_conv_PGA_J1')
    series{end+1} = {'PGA ($J\!=\!1$)',        x, get('rate_conv_PGA_J1'), get('crb_conv_PGA_J1'), get('obj_conv_PGA_J1'), COL.blue,   '--',  'none', false};
end
if isfield(d,'rate_conv_PGA_J5')
    series{end+1} = {'PGA ($J\!=\!5$)',        x, get('rate_conv_PGA_J5'), get('crb_conv_PGA_J5'), get('obj_conv_PGA_J5'), COL.orange, '-.',  'none', false};
end
if isfield(d,'rate_conv_PGA_J10')
    series{end+1} = {'PGA ($J\!=\!10$)',       x, get('rate_conv_PGA_J10'), get('crb_conv_PGA_J10'), get('obj_conv_PGA_J10'), COL.black, '--',  '*', false};
end
if isfield(d,'rate_conv_PGA_J20')
    series{end+1} = {'PGA ($J\!=\!20$)',       x, get('rate_conv_PGA_J20'), get('crb_conv_PGA_J20'), get('obj_conv_PGA_J20'), COL.black, '-.',  's', false};
end
if isfield(d,'rate_UPGA_J1')
    series{end+1} = {'UPGA ($J\!=\!1$)',       x, get('rate_UPGA_J1'), get('crb_UPGA_J1'), get('obj_UPGA_J1'), COL.cyan,   '-',   'o', false};
end
if isfield(d,'rate_UPGA_J5')
    series{end+1} = {'UPGA ($J\!=\!5$)',       x, get('rate_UPGA_J5'), get('crb_UPGA_J5'), get('obj_UPGA_J5'), COL.orange, '--',  'd', false};
end
if isfield(d,'rate_UPGA_J10')
    series{end+1} = {'UPGA ($J\!=\!10$)',      x, get('rate_UPGA_J10'), get('crb_UPGA_J10'), get('obj_UPGA_J10'), COL.red,    ':',   '*', false};
end
if isfield(d,'rate_UPGA_J20')
    series{end+1} = {'UPGA ($J\!=\!20$)',      x, get('rate_UPGA_J20'), get('crb_UPGA_J20'), get('obj_UPGA_J20'), COL.red,    '-',   'none', false};
end
if isfield(d,'rate_UPGA_J10_PRCDN')
    series{end+1} = {'UPGA PRCDN ($J\!=\!10$)',x, get('rate_UPGA_J10_PRCDN'), get('crb_UPGA_J10_PRCDN'), get('obj_UPGA_J10_PRCDN'), COL.green, ':',  '*', false};
end
if isfield(d,'rate_UPGA_J10_RMSProp')
    series{end+1} = {'UPGA RMSProp ($J\!=\!10$)',x,get('rate_UPGA_J10_RMSProp'),get('crb_UPGA_J10_RMSProp'),get('obj_UPGA_J10_RMSProp'),COL.green,':','none', false};
end
if isfield(d,'rate_UPGA_J5_decay')
    x5d = d.iter_outer_x_J5_decay;
    series{end+1} = {'UPGA decay ($J_{\max}\!=\!5$)', x5d, get('rate_UPGA_J5_decay'), get('crb_UPGA_J5_decay'), get('obj_UPGA_J5_decay'), COL.purple,'-','none', true};
end
if isfield(d,'rate_UPGA_J10_decay')
    x10d = d.iter_outer_x_J10_decay;
    series{end+1} = {'UPGA decay ($J_{\max}\!=\!10$)',x10d,get('rate_UPGA_J10_decay'),get('crb_UPGA_J10_decay'),get('obj_UPGA_J10_decay'),COL.purple,'-','d', true};
end
if isfield(d,'rate_UPGA_J20_decay')
    x20d = d.iter_outer_x_J20_decay;
    series{end+1} = {'UPGA decay ($J_{\max}\!=\!20$)',x20d,get('rate_UPGA_J20_decay'),get('crb_UPGA_J20_decay'),get('obj_UPGA_J20_decay'),COL.purple,'-','none', true};
end
if isfield(d,'rate_UPGA_J_GradReuse')
    series{end+1} = {'UPGA GradReuse ($J\!=\!10$)',x,get('rate_UPGA_J_GradReuse'),get('crb_UPGA_J_GradReuse'),get('obj_UPGA_J_GradReuse'),COL.teal,':','^', false};
end

if isempty(series)
    error('No recognised series found in %s.', MAT_FILE);
end

% -------------------------------------------------------------------------
%  Create figure
% -------------------------------------------------------------------------
fig = figure('Units','centimeters','Position',[2 2 FIG_WIDTH FIG_HEIGHT]);
set(fig,'defaultAxesFontName','Times New Roman', ...
        'defaultAxesFontSize', FONT_SIZE,         ...
        'defaultTextInterpreter','latex',         ...
        'defaultAxesTickLabelInterpreter','latex', ...
        'defaultLegendInterpreter','latex');

subplot_titles  = {'(a) Objective Function', '(b) Sum Rate', '(c) $I(\theta)$'};
subplot_ylabels = {'$\omega R + I(\theta)$', ...
                   '$R$ [bits/s/Hz]',                  ...
                   '$I(\theta)$'};
DATA_IDX = [5, 3, 4];   % column index in series cell: obj, rate, crb

ax = gobjects(1,3);
for p = 1:3
    ax(p) = subplot(1,3,p);
    hold(ax(p),'on');
    box(ax(p),'on');
    grid(ax(p),'on');
    set(ax(p), 'GridLineStyle',       ':',   ...
               'GridAlpha',           0.4,   ...
               'MinorGridLineStyle',  ':',   ...
               'XMinorGrid',          'off', ...
               'YMinorGrid',          'off', ...
               'TickDir',             'in',  ...
               'LineWidth',           0.8);

    for k = 1:numel(series)
        s    = series{k};
        lbl  = s{1};
        xk   = s{2}(:);
        yk   = s{DATA_IDX(p)}(:);
        if p == 3
            yk = log_info_to_info(yk);
        end
        col  = s{6};
        ls   = s{7};
        mk   = s{8};
        is_decay = (numel(s) >= 9) && logical(s{9});
        line_width_k = LINE_WIDTH;
        marker_size_k = MARKER_SIZE;
        marker_face_color_k = 'none';
        if is_decay
            line_width_k = LINE_WIDTH + 0.9;
            marker_size_k = MARKER_SIZE + 1.0;
            marker_face_color_k = col;
        end

        % Subsample markers for clarity
        mi   = false(size(xk));
        mi(1:MARK_EVERY:end) = true;

        if strcmp(mk,'none')
            plot(ax(p), xk, yk, ls, ...
                'Color', col, 'LineWidth', line_width_k, ...
                'DisplayName', lbl);
        else
            plot(ax(p), xk, yk, [ls mk], ...
                'Color', col, 'LineWidth', line_width_k, ...
                'MarkerSize', marker_size_k, ...
                'MarkerIndices', find(mi), ...
                'MarkerFaceColor', marker_face_color_k, ...
                'DisplayName', lbl);
        end
    end

    xlabel(ax(p), 'Iteration number $(I)$', 'FontSize', FONT_SIZE);
    ylabel(ax(p), subplot_ylabels{p},        'FontSize', FONT_SIZE);
    title(ax(p),  subplot_titles{p},         'FontSize', FONT_SIZE, 'FontWeight', 'normal');
end

% Shared legend below all panels
lgd = legend(ax(1), 'Location','southoutside',       ...
             'Orientation','horizontal',              ...
             'NumColumns', min(numel(series), 4),     ...
             'FontSize',   LEGEND_SIZE,               ...
             'Box',        'off');
lgd.ItemTokenSize = [12, 8];
lgd.Position(1) = (1 - lgd.Position(3)) / 2;  % centre horizontally
lgd.Position(2) = 0.01;

% Improve subplot spacing
set(fig,'Units','normalized');
for p = 1:3
    ax(p).Position(2) = 0.30;   % leave room for legend at bottom
    ax(p).Position(4) = 0.60;
end

% -------------------------------------------------------------------------
%  Export
% -------------------------------------------------------------------------
out_dir = fileparts(MAT_FILE);
base    = fullfile(out_dir, 'iter_results_panel');

print(fig, base, '-dpdf',  '-r300', '-bestfit');
print(fig, base, '-depsc', '-r300');
fprintf('Saved:\n  %s.pdf\n  %s.eps\n', base, base);

% -------------------------------------------------------------------------
%  Helper function
% -------------------------------------------------------------------------
function v = getfield_safe(s, name)
    if isfield(s, name)
        v = s.(name)(:);
    else
        v = [];
    end
end

function y = log_info_to_info(log_i_theta)
    % get_crb_fe stores log(I(theta)); convert back via exp(log(I(theta))).
    y = nan(size(log_i_theta));
    mask = isfinite(log_i_theta);
    y(mask) = exp(log_i_theta(mask));
end
