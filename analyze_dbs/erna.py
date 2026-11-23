"""ERNA measures from post-processed waveforms, following Sarah's incrementingExperiment_pull_wave_parameters.m:
latencies, peak-to-trough amplitudes, frequency, RMS and the coastline burst index, measured 3-20 ms after the
artifact peak (time 0, sample 98 of the post-processed waveform)."""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def time_axis_ms(n_samples, fs):
    """
    """
    samples = np.arange(0,n_samples)
    time_zero = round(0.002*fs) # sample 98, time zero
    time_ms = (samples - time_zero)/fs * 1000 # make samples count from artifact peak in ms
    return time_ms


def lowpass_600(segment, fs):
    """
    """
    b, a = butter(4, 600, 'lowpass', fs=fs,output='ba')
    filtered_segment = filtfilt(b, a, segment, padlen=12)
    return filtered_segment


def filtered_erna_window(signal, fs):
    """
    """
    t = time_axis_ms(len(signal), fs)
    window = (3 <= t)  & (t <= 20)
    windowed_time = t[window]
    windowed_signal = signal[window]
    filtered_window = lowpass_600(windowed_signal,fs)
    return windowed_time, filtered_window


def erna_rms(signal, fs):
    """
    """
    _, filtered_signal = filtered_erna_window(signal,fs)
    rms = np.sqrt(np.mean(filtered_signal**2))
    return rms


def find_extrema(times, filtered):
    """
    """
    peaks, _ = find_peaks()
    
    