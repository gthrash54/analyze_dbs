"""ERNA measures from post-processed waveforms, following Sarah's incrementingExperiment_pull_wave_parameters.m:
latencies, peak-to-trough amplitudes, frequency, RMS and the coastline burst index, measured 3-20 ms after the
artifact peak (time 0, sample 98 of the post-processed waveform)."""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def time_axis_ms(n_samples, fs):
    """
    The time of every sample of a post-processed waveform, in ms from the artifact peak.
    preprocess_dbs trims each channel so the (rounded) artifact peak sits at sample 98 counted from 0, which is
    round(0.002*fs) at 48.8 kHz. That sample is time 0; the samples before it are negative (about -2 ms at the start).
    Input:
    n_samples: how many samples the waveform has (2295 for a full one at 48.8 kHz)
    fs: sampling freq in Hz, the true one (48828.125), not Sarah's rounded 48828
    Returns: a 1-D array, n_samples long, of times in ms
    """
    samples = np.arange(0,n_samples) # 0, 1, 2, ... one per sample
    time_zero = round(0.002*fs) # sample 98, time zero
    time_ms = (samples - time_zero)/fs * 1000 # make samples count from artifact peak in ms
    return time_ms


def lowpass_600(segment, fs):
    """
    Sarah's 600 Hz low-pass for the ERNA measures (incrementingExperiment_pull_wave_parameters.m:577-583): a
    4th-order Butterworth run forwards and backwards (filtfilt), so the peaks aren't shifted in time.
    padlen=12 makes scipy pad the ends the way MATLAB's filtfilt does (3 x the filter order); scipy's own default (15)
    gives different values near the edges.
    One deliberate difference from Sarah: she designs the filter with fs rounded to 48828 (M:575); this uses the fs
    it is given.
    Input:
    segment: 1-D signal, already cut to the window to measure (it needs more than 12 samples)
    fs: sampling freq in Hz
    Returns: the filtered signal, same length (a new array)
    """
    b, a = butter(4, 600, 'lowpass', fs=fs,output='ba') # 'ba' form, as hers; passing fs lets the cutoff be in Hz
    filtered_segment = filtfilt(b, a, segment, padlen=12) # padlen=12 matches MATLAB's filtfilt
    return filtered_segment


def filtered_erna_window(signal, fs):
    """
    Cuts one post-processed waveform to the ERNA window, 3 to 20 ms after the artifact peak (both ends included), then
    low-passes it at 600 Hz. Sarah cuts first and filters second (M:582-583), so the filter's edge effects sit at 3
    and 20 ms; the order is kept to match her.
    Input:
    signal: one post-processed channel (1-D), artifact peak at sample 98, e.g. a row from postprocess_condition
    fs: sampling freq in Hz
    Returns: the window's times in ms from the artifact peak, and the filtered signal over that window (about 830
    samples each at 48.8 kHz)
    """
    t = time_axis_ms(len(signal), fs)
    window = (3 <= t)  & (t <= 20) # True for the samples from 3 to 20 ms
    windowed_time = t[window]
    windowed_signal = signal[window]
    filtered_window = lowpass_600(windowed_signal,fs) # cut first, then filter (her order)
    return windowed_time, filtered_window


def erna_rms(signal, fs):
    """
    Sarah's rms_20 (M:604, 626): the root mean square of the 600 Hz low-passed signal from 3 to 20 ms after the
    artifact peak. It is measured on the filtered signal, not the post-processed one (open question Q3).
    Input:
    signal: one post-processed channel (1-D), artifact peak at sample 98
    fs: sampling freq in Hz
    Returns: one number, in the signal's units (volts from postprocess_dbs; Sarah's tables are in mV)
    """
    _, filtered_signal = filtered_erna_window(signal,fs) # the times aren't needed here
    rms = np.sqrt(np.mean(filtered_signal**2))
    return rms


def find_extrema(times, filtered):
    """
    NOT FINISHED. Will find the peaks and troughs of the filtered ERNA window, as Sarah does with findpeaks on the
    signal and on its negative (M:586-594).
    """
    peaks, _ = find_peaks()
    
    