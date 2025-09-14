function ppg_morse_cwt_pipeline()
%% ====================== CONFIG (edit me) ======================
csvPath   = "E:\Update_data\dataset\P7_calm_F0RHR\ppg_accel.CSV";  % CSV with PPG + 3 accel
Fs        = 50;                                % sampling rate (Hz)
personID  = "ptest";                              % for output path
emotion   = "calm";                           % angry|calm|happy|sad
rootOut   = "E:\Update_data\dataset\AFTER";    % dataset root

% Processing / CWT params
Fpass     = [0.5 10.0];        % bandpass for PPG & accelerometer (Hz)
Fkeep     = [0.5 10.0];        % keep these freqs for the scalogram (Hz)
winSec    = 5.0;               % window length (s)
overlap   = 0.5;               % 50% overlap
voices    = 16;                % VoicesPerOctave for CWT
outSize   = [128 128];         % CNN input size
clip_pct  = [1 99];            % percentile clip for log-power normalization

% PREVIEW colormap (on-screen only) and SAVE colormap (files)
preview_colormap = 'turbo';    % color map for PREVIEW
save_colormap    = 'turbo';    % color map for SAVED images (keep fixed!)

% Saving
PNG_BITDEPTH = 8;              % 8 or 16 (per channel) for saved RGB PNGs
SAVE_PNG     = true;           % save color PNGs (RGB)
SAVE_MAT     = false;          % also save float tensors (.mat)

% Preview
preview_seconds = 25;          % how many seconds to preview

%% =================== PREVIEW (stages + CWT) ==================
preview_stages_and_cwt(csvPath, Fs, Fpass, voices, Fkeep, ...
    winSec, overlap, preview_seconds, preview_colormap, clip_pct);

%% ============ MAKE WINDOWS → CWT → SAVE (color PNG) =========
save_cwt_windows(csvPath, Fs, rootOut, personID, emotion, ...
    Fpass, Fkeep, winSec, overlap, voices, outSize, clip_pct, ...
    save_colormap, PNG_BITDEPTH, SAVE_PNG, SAVE_MAT);
end

%% ====================== PREVIEW FUNCTION ======================
function preview_stages_and_cwt(csvPath, Fs, Fpass, voices, Fkeep, ...
                                winSec, overlap, Tshow, cmapname, clip_pct)
[PPG_raw, AccX_raw, AccY_raw, AccZ_raw] = load_ir_acc(csvPath);

% High-pass (remove baseline)
Fn = Fs/2; [b_hp,a_hp] = butter(3, 0.5/Fn, 'high');
PPG_hp  = make_finite(filtfilt(b_hp, a_hp, PPG_raw));
AccX_hp = make_finite(filtfilt(b_hp, a_hp, AccX_raw));
AccY_hp = make_finite(filtfilt(b_hp, a_hp, AccY_raw));
AccZ_hp = make_finite(filtfilt(b_hp, a_hp, AccZ_raw));

% Band-pass 0.5–10 Hz
[b_bp,a_bp] = butter(6, Fpass/Fn, 'bandpass');
PPG_bp  = make_finite(filtfilt(b_bp, a_bp, PPG_hp));
AccX_bp = make_finite(filtfilt(b_bp, a_bp, AccX_hp));
AccY_bp = make_finite(filtfilt(b_bp, a_bp, AccY_hp));
AccZ_bp = make_finite(filtfilt(b_bp, a_bp, AccZ_hp));

% Normalize
PPG  = make_finite(normalize(PPG_bp));
AccX = make_finite(normalize(AccX_bp));
AccY = make_finite(normalize(AccY_bp));
AccZ = make_finite(normalize(AccZ_bp));

% RLS artifact removal (diagnostic only)
[PPG_clean, noise_used, best_axis, max_corr] = rls_clean(PPG, AccX, AccY, AccZ);

% ===== Stages figure: RAW → BP+Norm → RLS =====
Ns = min(numel(PPG_raw), round(Tshow*Fs));
t  = (0:Ns-1)/Fs;
raw_view = make_finite(PPG_raw(1:Ns) - median(PPG_raw(1:Ns),'omitnan'));

figure('Name','Stages: RAW → BP+Norm → RLS','Color','w');
subplot(3,1,1); plot(t, raw_view, 'Color',[0.25 0.25 0.25]); grid on;
title('RAW PPG (demeaned)'); xlabel('Time (s)'); ylabel('a.u.');
subplot(3,1,2); plot(t, PPG(1:Ns), 'k'); grid on;
title('Band-pass (0.5–10 Hz) + Normalize'); xlabel('Time (s)'); ylabel('a.u.');
subplot(3,1,3); plot(t, PPG_clean(1:Ns), 'b'); grid on;
title('RLS-cleaned (diagnostic)'); xlabel('Time (s)'); ylabel('a.u.');

% ===== PPG vs Noise vs Cleaned (diagnostic) =====
figure('Name','PPG Before vs After (diagnostic)','Color','w');
subplot(3,1,1); plot(t, PPG(1:Ns),'k'); grid on;
title('PPG (bandpassed & normalized)'); xlabel('s'); ylabel('a.u.');
subplot(3,1,2); plot(t, noise_used(1:Ns)); grid on;
title(sprintf('Estimated Motion Noise (axis: %s, |corr|=%.3f)', best_axis, max_corr));
xlabel('s'); ylabel('a.u.');
subplot(3,1,3); plot(t, PPG_clean(1:Ns),'b'); grid on;
title('PPG after RLS'); xlabel('s'); ylabel('a.u.');

% ===== Morse CWT PREVIEW in COLOR (AFTER BP+NORM) =====
winS = winSec; hopS = winS*(1-overlap);
W = round(winS*Fs); H = max(1, round(hopS*Fs));
starts = 1:H:(min(numel(PPG), round(Tshow*Fs))-W+1);   % use PPG (BP+Norm)
nshow = min(5, numel(starts));

figure('Name','Preview: Morse CWT (COLOR, after BP+Norm)','Color','w');
for i = 1:nshow
    seg = make_finite(PPG(starts(i):starts(i)+W-1));     % use PPG (not RLS)
    [wt,f] = cwt(seg, Fs, 'morse', 'VoicesPerOctave', voices);
    P = abs(wt).^2;

    mask = f>=Fkeep(1) & f<=Fkeep(2);
    P = P(mask,:); f2 = f(mask);

    I = normalize_log_power(P, clip_pct);
    subplot(nshow,1,i);
    imagesc(linspace(0,winS,size(I,2)), f2, I); axis xy tight;
    colormap(cmapname); colorbar;
    xlabel('Time (s)'); ylabel('Hz');
    title(sprintf('Window %d (%.1f–%.1f Hz, Morse, color PREVIEW after BP+Norm)', i, Fkeep(1), Fkeep(2)));
end
end

%% ===================== SAVE WINDOWS (color PNG / MAT) =====================
function save_cwt_windows(csvPath, Fs, rootOut, personID, emotion, ...
                          Fpass, Fkeep, winSec, overlap, voices, outSize, ...
                          clip_pct, save_cmap, PNG_BITDEPTH, SAVE_PNG, SAVE_MAT)
[PPG_raw, AccX_raw, AccY_raw, AccZ_raw] = load_ir_acc(csvPath);

% HP + BP + Normalize (0.5–10 Hz)
Fn = Fs/2; [b_hp,a_hp] = butter(3, 0.5/Fn, 'high');
PPG_hp  = make_finite(filtfilt(b_hp, a_hp, PPG_raw));
AccX_hp = make_finite(filtfilt(b_hp, a_hp, AccX_raw));
AccY_hp = make_finite(filtfilt(b_hp, a_hp, AccY_raw));
AccZ_hp = make_finite(filtfilt(b_hp, a_hp, AccZ_raw));

[b_bp,a_bp] = butter(6, Fpass/Fn, 'bandpass');
PPG  = make_finite(normalize(filtfilt(b_bp, a_bp, PPG_hp)));
AccX = make_finite(normalize(filtfilt(b_bp, a_bp, AccX_hp)));
AccY = make_finite(normalize(filtfilt(b_bp, a_bp, AccY_hp)));
AccZ = make_finite(normalize(filtfilt(b_bp, a_bp, AccZ_hp)));

% (Optional) RLS clean for reporting only
[PPG_clean, ~, best_axis, max_corr] = rls_clean(PPG, AccX, AccY, AccZ);
fprintf('[INFO] (diagnostic) RLS axis=%s, |corr|=%.3f\n', best_axis, max_corr);

% Windowing will use PPG (bandpass+normalized)
W = round(winSec*Fs);
H = max(1, round(W*(1-overlap)));
starts = 1:H:(numel(PPG)-W+1);
if isempty(starts)
    warning('Signal too short for a single window. Skipping save.'); return;
end

% Output folder
outDir = fullfile(rootOut, personID, emotion, "CWT");
if ~exist(outDir,'dir'), mkdir(outDir); end

% Precompute freq mask for 0.5–10 Hz
tmp = make_finite(PPG(1:W));
[~, f0] = cwt(tmp, Fs, 'morse', 'VoicesPerOctave', voices);
fmask = (f0>=Fkeep(1) & f0<=Fkeep(2));

nSaved = 0;
for i = 1:numel(starts)
    seg = make_finite(PPG(starts(i):starts(i)+W-1));
    [wt,f] = cwt(seg, Fs, 'morse', 'VoicesPerOctave', voices);
    pow = abs(wt).^2;

    if numel(f) ~= numel(f0), fmask = (f>=Fkeep(1) & f<=Fkeep(2)); end
    pow = pow(fmask, :);

    % Normalize to [0,1] with log-power + percentile clip
    I01 = normalize_log_power(pow, clip_pct);
    if ~all(isfinite(I01(:))), continue; end

    % Resize
    I01 = imresize(I01, outSize, 'bicubic');

    % === SAVE as COLOR PNG (RGB) using fixed colormap ===
    if SAVE_PNG
        pngPath = fullfile(outDir, sprintf('win_%04d.png', i));
        RGB = to_colormap_rgb(I01, save_cmap, PNG_BITDEPTH);
        imwrite(RGB, pngPath);        % uint8 or uint16 MxNx3
    end

    % Optional: also save float tensor
    if SAVE_MAT
        I = single(I01);
        matPath = fullfile(outDir, sprintf('win_%04d.mat', i));
        save(matPath, 'I', '-v7.3');
    end

    nSaved = nSaved + 1;
end
fprintf('[DONE] Saved %d window(s) into %s (BP+Norm → CWT, keep %.1f–%.1f Hz, color PNG)\n', ...
    nSaved, outDir, Fkeep(1), Fkeep(2));
end

%% ========================= HELPERS =========================
function [PPG, AccX, AccY, AccZ] = load_ir_acc(csvPath)
T = readtable(csvPath);
V = lower(T.Properties.VariableNames);

% Prefer explicit names
ppg_idx = find(ismember(V, {'ir','ppg','ppg_value','signal','value'}), 1);

% Time columns to ignore when falling back
is_time = contains(V,'time') | contains(V,'stamp') | contains(V,'date');

% Try accel by common names
ax_idx = find(ismember(V, {'ax','acc_x','accx','a1','accx_mg'}), 1);
ay_idx = find(ismember(V, {'ay','acc_y','accy','a2','accy_mg'}), 1);
az_idx = find(ismember(V, {'az','acc_z','accz','a3','accz_mg'}), 1);

numMask = varfun(@isnumeric, T, 'OutputFormat','uniform');
numIdx  = find(numMask & ~is_time);

% Fallbacks if needed
if isempty(ppg_idx)
    if ~isempty(numIdx), ppg_idx = numIdx(1);
    else, error('No numeric columns for PPG found.'); end
end
if isempty(ax_idx) || isempty(ay_idx) || isempty(az_idx)
    rest = setdiff(numIdx, ppg_idx, 'stable');
    if numel(rest) < 3, error('Need 3 accel columns; found %d.', numel(rest)); end
    ax_idx = rest(1); ay_idx = rest(2); az_idx = rest(3);
end

PPG  = make_finite(single(T{:, ppg_idx}));
AccX = make_finite(single(T{:, ax_idx}));
AccY = make_finite(single(T{:, ay_idx}));
AccZ = make_finite(single(T{:, az_idx}));
end

function [PPG_clean, Noise, best_axis, g] = rls_clean(PPG, AccX, AccY, AccZ)
% Sanitize
PPG  = make_finite(PPG);
AccX = make_finite(AccX);
AccY = make_finite(AccY);
AccZ = make_finite(AccZ);

% Correlations
c1 = abs(corr(PPG, AccX, 'Rows','complete'));
c2 = abs(corr(PPG, AccY, 'Rows','complete'));
c3 = abs(corr(PPG, AccZ, 'Rows','complete'));
Acc = sqrt(AccX.^2 + AccY.^2 + AccZ.^2);
cr = abs(corr(PPG, Acc,  'Rows','complete'));
[g, whichAxis] = max([cr, c3, c2, c1]);
axesNames = {'resultant','Z','Y','X'};
best_axis = axesNames{whichAxis};

% RLS filter
M = 7; lambda = 1.0; delta = 0.999;
P0 = (1/delta)*eye(M, M, 'single');
rls = dsp.RLSFilter(M, 'ForgettingFactor', lambda, ...
                       'InitialInverseCovariance', P0);

if g < 0.05
    Noise = zeros(size(PPG), 'like', PPG);
    PPG_clean = PPG;
else
    switch whichAxis
        case 1, ref = Acc;
        case 2, ref = AccZ;
        case 3, ref = AccY;
        case 4, ref = AccX;
    end
    [Noise, PPG_hat] = rls(ref, PPG);
    PPG_clean = PPG_hat - Noise;
end

PPG_clean = make_finite(PPG_clean);
Noise     = make_finite(Noise);
end

function I01 = normalize_log_power(P, clip_pct)
% P is nonnegative power; convert to log-power and scale to [0,1]
Pl = 10*log10(max(P, 0) + 1e-12);
lohi = prctile(Pl(:), clip_pct);
Pl = min(max(Pl, lohi(1)), lohi(2));
I01 = (Pl - lohi(1)) / max(lohi(2) - lohi(1), 1e-9);
I01(~isfinite(I01)) = 0;
end

function RGB = to_colormap_rgb(I01, cmapname, bitdepth)
% Map [0,1] image I01 to RGB using a fixed colormap
I01 = min(max(I01,0),1);
idx = 1 + floor(I01*255);                 % 1..256
cmap = feval(cmapname, 256);              % 256x3, double in [0,1]
RGBd = ind2rgb(idx, cmap);                % double MxNx3 in [0,1]
switch bitdepth
    case 16
        RGB = uint16(round(RGBd*65535));
    otherwise % 8-bit
        RGB = uint8(round(RGBd*255));
end
end

function X = make_finite(X)
% Ensure finite; for vectors, also interpolate NaN/Inf
if isvector(X)
    x = double(X(:));
    bad = ~isfinite(x);
    if any(bad)
        t = (1:numel(x))'; good = ~bad;
        if any(good)
            x(bad) = interp1(t(good), x(good), t(bad), 'linear', 'extrap');
        else
            x(:) = 0;
        end
    end
    x(~isfinite(x)) = 0;
    X = x;
else
    X = double(X);
    X(~isfinite(X)) = 0;
end
end

