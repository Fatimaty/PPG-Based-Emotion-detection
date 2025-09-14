
% preprocess.m
% Preprocessing functions for PPG emotion recognition
% Includes: Bandpass filter + Normalization

function [filteredSignal, normSignal] = preprocess(signal, fs)
    % INPUT:
    %   signal : raw PPG signal (vector)
    %   fs     : sampling frequency (Hz)
    % OUTPUT:
    %   filteredSignal : bandpass filtered PPG
    %   normSignal     : normalized (zero-mean, unit variance)

    % ---- Bandpass filter (0.5 - 5 Hz) ----
    lowcut = 0.5;  % Hz
    highcut = 5.0; % Hz
    order = 4;

    [b, a] = butter(order, [lowcut, highcut]/(fs/2), 'bandpass');
    filteredSignal = filtfilt(b, a, double(signal));

    % ---- Normalization ----
    mu = mean(filteredSignal);
    sigma = std(filteredSignal) + 1e-8; % avoid div-by-zero
    normSignal = (filteredSignal - mu) / sigma;
end
