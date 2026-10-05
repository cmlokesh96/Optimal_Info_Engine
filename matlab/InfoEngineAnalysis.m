classdef InfoEngineAnalysis
    % InfoEngineAnalysis - load and plot analysis.py::export_for_matlab()
    % output for manuscript figures.
    %
    % export_for_matlab() saves the EXACT arrays plot_x_lambda_grid()/
    % plot_work_grid() already plot in Python - per state, one struct
    % (t_s, x_nm, lambda_nm, W_cum_kT, n_events, m0, mt) - plus scalar
    % metadata (kappa_N_per_m, T_K, protocol_dt_s). This class just loads
    % that .mat and re-exposes the same plots/summaries in MATLAB, with no
    % re-derivation of the extraction/work-formula logic.
    %
    % A .mat saved via analyze_batch_from_stage.py (matching Main_script.
    % ipynb's "4b. Batch-folder analysis" section) additionally carries,
    % per state when available: W_jump_kT (the jump's own work, kT),
    % predicted_work_kT/trigger_reference_kT (scalar reference lines), and
    % stage_nm/stage_t_s/stage_protocol_nm (physical Stage-channel
    % readback vs. the loaded protocol) - plotTrapRelative/plotWorkSplit/
    % plotStageVsProtocol below use these the same way Python's
    % plot_trap_relative_grid/plot_work_split_grid/plot_stage_vs_protocol_grid
    % do; an older .mat saved before this extension simply won't have
    % these fields, and those methods fall back to a "not available"
    % message per state instead of erroring.
    %
    % Field naming: MATLAB struct field names can't hold '(', ',', or '-',
    % so each state (m0, mt) is saved as state_<m0>_<mt> with '-' written
    % as 'neg' (e.g. (1,1) -> state_1_1, (-1,-1) -> state_neg1_neg1) - see
    % analysis.py::export_for_matlab()'s _field() helper, which
    % stateFieldName() below must keep matching exactly.
    %
    % Example
    % -------
    %   obj = InfoEngineAnalysis('D:\data\check01_events_work.mat');
    %   obj.states()
    %   obj.plotXLambda(1, 1);
    %   obj.plotWork(1, 1);
    %   obj.plotAllStatesGrid();               % x/lambda grid, all states
    %   obj.plotAllStatesGrid('work');          % cumulative-work grid
    %   T = obj.workSummaryTable()
    %
    %   % Same 3 plots as Main_script.ipynb's section 4b / analyze_batch_from_stage.py:
    %   obj.plotAllStatesGrid('trap_relative');   % trap-relative X and lambda
    %   obj.plotAllStatesGrid('work_split');      % jump vs. total work, with reference lines
    %   obj.plotAllStatesGrid('stage');           % measured Stage readback vs. loaded protocol
    %
    %   % Compare e.g. single- vs two-measurement runs for state (1,1):
    %   objSingle = InfoEngineAnalysis('D:\data\single01_events_work.mat');
    %   objTwo    = InfoEngineAnalysis('D:\data\check01_events_work.mat');
    %   InfoEngineAnalysis.compare({objTwo, objSingle}, ...
    %       {'two-measurement', 'single-measurement'}, 1, 1);

    properties
        data            % struct: state field name -> struct(t_s,x_nm,lambda_nm,W_cum_kT,n_events,m0,mt)
        kappa_N_per_m
        T_K
        protocol_dt_s
        source_path
    end

    methods
        function obj = InfoEngineAnalysis(matfile_path)
            raw = load(matfile_path);
            obj.source_path   = matfile_path;
            obj.kappa_N_per_m = raw.kappa_N_per_m;
            obj.T_K           = raw.T_K;
            obj.protocol_dt_s = raw.protocol_dt_s;

            obj.data = struct();
            fns = fieldnames(raw);
            for i = 1:numel(fns)
                fn = fns{i};
                if startsWith(fn, 'state_')
                    obj.data.(fn) = raw.(fn);
                end
            end
        end

        function names = states(obj)
            % Every state field name actually present in this .mat (not
            % every state that was ever requested - a state with 0 real
            % events is still present, with n_events==0).
            names = fieldnames(obj.data);
        end

        function s = getState(obj, m0, mt)
            key = InfoEngineAnalysis.stateFieldName(m0, mt);
            if ~isfield(obj.data, key)
                error('InfoEngineAnalysis:noState', ...
                    'State (%d,%d) not found in %s', m0, mt, obj.source_path);
            end
            s = obj.data.(key);
        end

        function plotXLambda(obj, m0, mt)
            % Mean x(t) (+/- std band) and lambda(t) overlaid, one state.
            s = obj.getState(m0, mt);
            figure;
            InfoEngineAnalysis.plotXLambdaAxes(gca, s, m0, mt);
        end

        function plotWork(obj, m0, mt)
            % Mean cumulative work(t) (+/- std band), one state.
            s = obj.getState(m0, mt);
            figure;
            InfoEngineAnalysis.plotWorkAxes(gca, s, m0, mt);
        end

        function plotTrapRelative(obj, m0, mt)
            % Trap-relative X(t) = x_nm + lambda_nm (added from sample 3
            % onward, matching Python's plot_trap_relative_grid - this
            % hardware's confirmed extra ~1-frame actuation delay means
            % samples 1-2 are both still pre-jump) overlaid with
            % lambda(t), its own instantaneous jump drawn as a dashed
            % marker. One state.
            s = obj.getState(m0, mt);
            figure;
            InfoEngineAnalysis.plotTrapRelativeAxes(gca, s, m0, mt);
        end

        function plotWorkSplit(obj, m0, mt)
            % Cumulative TOTAL work(t), with the jump's own share drawn as
            % an explicit dashed step from (t(1),0) to (t(2),<W_jump>),
            % plus the predicted/trigger-reference horizontal lines when
            % the .mat has them. One state.
            s = obj.getState(m0, mt);
            figure;
            InfoEngineAnalysis.plotWorkSplitAxes(gca, s, m0, mt);
        end

        function plotStageVsProtocol(obj, m0, mt)
            % Measured Stage-channel readback (+/- std band) vs. the
            % loaded protocol's stage-frame curve, one state - needs
            % stage_nm/stage_t_s/stage_protocol_nm in the .mat (saved by
            % analyze_batch_from_stage.py; an older .mat without them
            % shows a "not available" placeholder instead of erroring).
            s = obj.getState(m0, mt);
            figure;
            InfoEngineAnalysis.plotStageVsProtocolAxes(gca, s, m0, mt);
        end

        function plotAllStatesGrid(obj, quantity)
            % One figure, one panel per state - quantity is 'x_lambda'
            % (default), 'work', 'trap_relative', 'work_split', or
            % 'stage', same as plotXLambda()/plotWork()/plotTrapRelative()/
            % plotWorkSplit()/plotStageVsProtocol() but for every state in
            % this .mat at once. Grid layout as square as possible,
            % matching analysis.py::_grid_dims()'s rule.
            if nargin < 2
                quantity = 'x_lambda';
            end
            names = obj.states();
            n = numel(names);
            ncols = ceil(sqrt(n));
            nrows = ceil(n / ncols);
            figure;
            for i = 1:n
                s = obj.data.(names{i});
                ax = subplot(nrows, ncols, i);
                switch quantity
                    case 'work'
                        InfoEngineAnalysis.plotWorkAxes(ax, s, s.m0, s.mt);
                    case 'trap_relative'
                        InfoEngineAnalysis.plotTrapRelativeAxes(ax, s, s.m0, s.mt);
                    case 'work_split'
                        InfoEngineAnalysis.plotWorkSplitAxes(ax, s, s.m0, s.mt);
                    case 'stage'
                        InfoEngineAnalysis.plotStageVsProtocolAxes(ax, s, s.m0, s.mt);
                    otherwise
                        InfoEngineAnalysis.plotXLambdaAxes(ax, s, s.m0, s.mt);
                end
            end
        end

        function T = workSummaryTable(obj)
            % One row per state: m0, mt, n_events, mean/std of final
            % (t=t_s(end)) cumulative work in kT.
            names = obj.states();
            n = numel(names);
            m0 = zeros(n, 1); mt = zeros(n, 1); n_events = zeros(n, 1);
            mean_final_work_kT = nan(n, 1); std_final_work_kT = nan(n, 1);
            for i = 1:n
                s = obj.data.(names{i});
                m0(i) = s.m0; mt(i) = s.mt; n_events(i) = s.n_events;
                if s.n_events > 0
                    final = s.W_cum_kT(:, end);
                    mean_final_work_kT(i) = mean(final);
                    std_final_work_kT(i)  = std(final);
                end
            end
            T = table(m0, mt, n_events, mean_final_work_kT, std_final_work_kT);
        end
    end

    methods (Static)
        function key = stateFieldName(m0, mt)
            key = sprintf('state_%d_%d', m0, mt);
            key = strrep(key, '-', 'neg');
        end

        function fillBand(t, lo, hi, color)
            % Shared +/- std band fill, used by plotXLambdaAxes/plotWorkAxes.
            fill([t, fliplr(t)], [lo, fliplr(hi)], color, ...
                'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        end

        function plotXLambdaAxes(ax, s, m0, mt)
            axes(ax); hold(ax, 'on'); %#ok<LAXES>
            if s.n_events == 0
                text(0.5, 0.5, 'no events', 'HorizontalAlignment', 'center', ...
                    'Units', 'normalized', 'Parent', ax);
            else
                x_mean = mean(s.x_nm, 1);
                x_std  = std(s.x_nm, 0, 1);
                InfoEngineAnalysis.fillBand(s.t_s, x_mean - x_std, x_mean + x_std, [0.3 0.5 0.9]);
                plot(ax, s.t_s, x_mean, 'Color', [0 0.2 0.8], 'LineWidth', 1.5, 'DisplayName', '<x>');
                plot(ax, s.t_s, s.lambda_nm, 'Color', [0 0.6 0.2], 'LineWidth', 1.5, 'DisplayName', '\lambda');
                legend(ax, 'show');
            end
            xlabel(ax, 't (s)'); ylabel(ax, 'Position (nm)');
            title(ax, sprintf('m_0=%d, m_t=%d  n=%d', m0, mt, s.n_events));
            grid(ax, 'on'); hold(ax, 'off');
        end

        function plotWorkAxes(ax, s, m0, mt)
            axes(ax); hold(ax, 'on'); %#ok<LAXES>
            if s.n_events == 0
                text(0.5, 0.5, 'no events', 'HorizontalAlignment', 'center', ...
                    'Units', 'normalized', 'Parent', ax);
            else
                W_mean = mean(s.W_cum_kT, 1);
                W_std  = std(s.W_cum_kT, 0, 1);
                InfoEngineAnalysis.fillBand(s.t_s, W_mean - W_std, W_mean + W_std, [0.9 0.4 0.3]);
                plot(ax, s.t_s, W_mean, 'Color', [0.8 0.1 0.1], 'LineWidth', 1.5, 'DisplayName', '<W>');
                legend(ax, 'show');
            end
            xlabel(ax, 't (s)'); ylabel(ax, 'W / kT');
            title(ax, sprintf('m_0=%d, m_t=%d  n=%d', m0, mt, s.n_events));
            grid(ax, 'on'); hold(ax, 'off');
        end

        function plotTrapRelativeAxes(ax, s, m0, mt)
            % Mirrors analysis.py::plot_trap_relative_grid for one state.
            axes(ax); hold(ax, 'on'); %#ok<LAXES>
            if s.n_events == 0
                text(0.5, 0.5, 'no events', 'HorizontalAlignment', 'center', ...
                    'Units', 'normalized', 'Parent', ax);
            else
                lam = s.lambda_nm(:)';
                X_true = s.x_nm;
                if numel(lam) > 2
                    X_true(:, 3:end) = X_true(:, 3:end) + lam(3:end);
                end
                X_mean = mean(X_true, 1);
                X_std  = std(X_true, 0, 1);
                InfoEngineAnalysis.fillBand(s.t_s, X_mean - X_std, X_mean + X_std, [0.3 0.5 0.9]);
                plot(ax, s.t_s, X_mean, 'Color', [0 0.2 0.8], 'LineWidth', 1.5, 'DisplayName', '<X>');
                plot(ax, s.t_s, lam, 'Color', [0 0.6 0.2], 'LineWidth', 1.5, 'DisplayName', '\lambda');
                if ~isempty(lam) && lam(1) ~= 0
                    plot(ax, [s.t_s(1), s.t_s(1)], [0.0, lam(1)], 'Color', [0 0.6 0.2], ...
                        'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
                    plot(ax, s.t_s(1), 0.0, 'o', 'MarkerSize', 5, 'MarkerFaceColor', 'white', ...
                        'MarkerEdgeColor', [0 0.6 0.2], 'HandleVisibility', 'off');
                end
                legend(ax, 'show');
            end
            xlabel(ax, 't (s)'); ylabel(ax, 'Position (nm)');
            title(ax, sprintf('m_0=%d, m_t=%d  n=%d', m0, mt, s.n_events));
            grid(ax, 'on'); hold(ax, 'off');
        end

        function plotWorkSplitAxes(ax, s, m0, mt)
            % Mirrors analysis.py::plot_work_split_grid for one state -
            % needs W_jump_kT (saved by analyze_batch_from_stage.py);
            % predicted_work_kT/trigger_reference_kT reference lines are
            % drawn only when present in the .mat.
            axes(ax); hold(ax, 'on'); %#ok<LAXES>
            if s.n_events == 0
                text(0.5, 0.5, 'no events', 'HorizontalAlignment', 'center', ...
                    'Units', 'normalized', 'Parent', ax);
            elseif ~isfield(s, 'W_jump_kT')
                text(0.5, 0.5, 'W_jump_kT not in this .mat', 'HorizontalAlignment', 'center', ...
                    'Units', 'normalized', 'Parent', ax);
            else
                W_mean = mean(s.W_cum_kT, 1);
                W_jump_mean = mean(s.W_jump_kT(:));

                if numel(s.t_s) > 1 && W_jump_mean ~= 0
                    plot(ax, [s.t_s(1), s.t_s(2)], [0.0, W_jump_mean], 'Color', [0.85 0.5 0.1], ...
                        'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
                    plot(ax, s.t_s(1), 0.0, 'o', 'MarkerSize', 5, 'MarkerFaceColor', 'white', ...
                        'MarkerEdgeColor', [0.85 0.5 0.1], 'HandleVisibility', 'off');
                end

                plot(ax, s.t_s, W_mean, 'Color', [0.8 0.1 0.1], 'LineWidth', 1.5, 'DisplayName', '<W_{total}>');

                if isfield(s, 'predicted_work_kT')
                    yline(ax, s.predicted_work_kT, 'Color', [0.5 0.5 0.5], 'LineStyle', '--', ...
                        'LineWidth', 2, 'DisplayName', 'predicted (analytical)');
                end
                if isfield(s, 'trigger_reference_kT')
                    yline(ax, s.trigger_reference_kT, 'Color', [0.55 0.35 0.2], 'LineStyle', ':', ...
                        'LineWidth', 2, 'DisplayName', '<V_{ij}>');
                end
                legend(ax, 'show');
            end
            xlabel(ax, 't (s)'); ylabel(ax, 'W / kT');
            title(ax, sprintf('m_0=%d, m_t=%d  n=%d', m0, mt, s.n_events));
            grid(ax, 'on'); hold(ax, 'off');
        end

        function plotStageVsProtocolAxes(ax, s, m0, mt)
            % Mirrors analysis.py::plot_stage_vs_protocol_grid for one
            % state - needs stage_nm/stage_t_s (saved by
            % analyze_batch_from_stage.py); stage_protocol_nm is optional
            % (drawn only when present).
            axes(ax); hold(ax, 'on'); %#ok<LAXES>
            if ~isfield(s, 'stage_nm')
                text(0.5, 0.5, 'stage data not in this .mat', 'HorizontalAlignment', 'center', ...
                    'Units', 'normalized', 'Parent', ax);
            elseif s.n_events == 0
                text(0.5, 0.5, 'no events', 'HorizontalAlignment', 'center', ...
                    'Units', 'normalized', 'Parent', ax);
            else
                s_mean = mean(s.stage_nm, 1);
                s_std  = std(s.stage_nm, 0, 1);
                InfoEngineAnalysis.fillBand(s.stage_t_s, s_mean - s_std, s_mean + s_std, [0.6 0.4 0.8]);
                plot(ax, s.stage_t_s, s_mean, 'Color', [0.4 0.1 0.6], 'LineWidth', 1.5, ...
                    'DisplayName', 'Stage (measured)');
                if isfield(s, 'stage_protocol_nm')
                    plot(ax, s.stage_t_s, s.stage_protocol_nm, 'Color', [0.9 0.5 0.1], ...
                        'LineStyle', '--', 'LineWidth', 1.5, 'DisplayName', 'protocol (loaded)');
                end
                legend(ax, 'show');
            end
            xlabel(ax, 't (s)'); ylabel(ax, 'Relative stage position (nm)');
            title(ax, sprintf('m_0=%d, m_t=%d  n=%d', m0, mt, s.n_events));
            grid(ax, 'on'); hold(ax, 'off');
        end

        function compare(objs, labels, m0, mt)
            % Bar chart (+/- std error bars) of mean final cumulative work
            % for one state, across several InfoEngineAnalysis objects -
            % e.g. single- vs two-measurement, or several batches.
            %   InfoEngineAnalysis.compare({objTwo, objSingle}, ...
            %       {'two-measurement', 'single-measurement'}, 1, 1);
            n = numel(objs);
            means = nan(1, n);
            stds  = nan(1, n);
            for i = 1:n
                s = objs{i}.getState(m0, mt);
                if s.n_events > 0
                    final = s.W_cum_kT(:, end);
                    means(i) = mean(final);
                    stds(i)  = std(final);
                end
            end
            figure;
            bar(means);
            hold on;
            errorbar(1:n, means, stds, 'k.', 'LineWidth', 1.2);
            set(gca, 'XTick', 1:n, 'XTickLabel', labels);
            ylabel('Final W / kT');
            title(sprintf('Work comparison, m_0=%d, m_t=%d', m0, mt));
            grid on; hold off;
        end
    end
end
