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

        function plotAllStatesGrid(obj, quantity)
            % One figure, one panel per state - quantity is 'x_lambda'
            % (default) or 'work', same as plotXLambda()/plotWork() but
            % for every state in this .mat at once. Grid layout as square
            % as possible, matching analysis.py::_grid_dims()'s rule.
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
                if strcmp(quantity, 'work')
                    InfoEngineAnalysis.plotWorkAxes(ax, s, s.m0, s.mt);
                else
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
