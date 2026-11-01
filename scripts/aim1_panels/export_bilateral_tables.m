% EXPORT_BILATERAL_TABLES  Read-only export of the MATLAB tables saved by the
% Newton_SC_2026 bilateral analyses (peer repo; never modified) to CSV, for
% the Aim 1 panel builders B7 and B9 (scripts/aim1_panels/build_B7.py,
% build_B9.py). No analysis is recomputed here.
% Output: results/aim1_panels/_bilateral_source/*.csv
% Run: matlab -batch "run('scripts/aim1_panels/export_bilateral_tables.m')"
src = 'D:\Google Drive\HermanLab\data\Newton_SC_2026\data';
out = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'results', 'aim1_panels', '_bilateral_source');
if ~exist(out, 'dir'), mkdir(out); end
ses = {'Newton_08202026', 'Newton_08192026'};
% competition_choice: per-window table (added CP, LR p, reward t)
C = load(fullfile(src, 'bilateral', 'competition_choice.mat')).res;
for s = 1:2
    tc = C.(ses{s}).tc; tc.session = repmat(string(ses{s}), height(tc), 1);
    writetable(tc, fullfile(out, sprintf('competition_choice_tc_%s.csv', ses{s})));
    B = C.(ses{s}).B; writetable(B, fullfile(out, sprintf('competition_choice_trials_%s.csv', ses{s})));
end
% competition_normalisation: per-unit coefficients and null/planted sims of the mean
N = load(fullfile(src, 'bilateral', 'competition_normalisation.mat')).res;
U = table(); S = table();
for s = 1:2
    if ~isfield(N, ses{s}), continue, end
    for ep = ["visual", "delay", "presacc"]
        if ~isfield(N.(ses{s}), ep), continue, end
        r = N.(ses{s}).(ep);
        u = r.U; u.session = repmat(string(ses{s}), height(u), 1); u.epoch = repmat(ep, height(u), 1);
        U = [U; u]; %#ok<AGROW>
        g = [0 0.1 0.2];
        for gi = 1:3
            for m = 1:2
                v = r.sims(:, m, gi);
                S = [S; table(repmat(string(ses{s}), numel(v), 1), repmat(ep, numel(v), 1), ...
                    repmat(g(gi), numel(v), 1), repmat(string(char('a' + m - 1)), numel(v), 1), (1:numel(v))', v, ...
                    'VariableNames', {'session', 'epoch', 'gamma', 'model', 'sim', 'mean_comp_coef'})]; %#ok<AGROW>
            end
        end
        fprintf('%s %s: realA %+.4f realB %+.4f lo %s powA %s powB %s n %d\n', ses{s}, ep, r.realA, r.realB, ...
            mat2str(r.lo, 3), mat2str(r.powA, 3), mat2str(r.powB, 3), height(r.U));
    end
end
writetable(U, fullfile(out, 'competition_normalisation_units.csv'));
writetable(S, fullfile(out, 'competition_normalisation_sims.csv'));
% intercollicular: S1 relay simulation table, CCG split table, bound table
E = load(fullfile(src, 'qc', 'onset_estimators.mat'));
o = E.out; if istable(o), writetable(o, fullfile(out, 'onset_estimators_S1.csv')); else, disp(class(o)); end
G = load(fullfile(src, 'bilateral', 'intercollicular_ccg_split.mat'));
writetable(G.R, fullfile(out, 'intercollicular_ccg_split_R.csv'));
Bd = load(fullfile(src, 'bilateral', 'intercollicular_bound.mat'));
writetable(Bd.B, fullfile(out, 'intercollicular_bound_B.csv'));
L = load(fullfile(src, 'bilateral', 'intercollicular_latency.mat')).res;
for nm = ["act", "supp"]
    R = L.valid.(nm);
    u = R.unit; 
    try, writetable(u, fullfile(out, sprintf('intercollicular_latency_units_%s.csv', nm))); catch e, disp(e.message); end
    c = R.combos; T = table();
    for k = 1:numel(c)
        T = [T; table(string(c(k).session), c(k).hemi, string(c(k).xSide), string(c(k).ySide), c(k).nTrials, ...
            R.onsets(k, 1), R.onsets(k, 2), R.onsets(k, 3), R.onsets(k, 4), 'VariableNames', ...
            {'session', 'hemi', 'xSide', 'ySide', 'nTrials', 'x_rise_s', 'x_dep_s', 'y_rise_s', 'y_dep_s'})]; %#ok<AGROW>
    end
    writetable(T, fullfile(out, sprintf('intercollicular_latency_combos_%s.csv', nm)));
end
disp('done')
