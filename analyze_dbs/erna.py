"""ERNA measures from post-processed waveforms, following Sarah's incrementingExperiment_pull_wave_parameters.m:
latencies, peak-to-trough amplitudes, frequency, RMS and the coastline burst index, measured 3-20 ms after the
artifact peak (time 0, sample 98 of the post-processed waveform)."""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def time_axis_ms(n_samples, fs, t0_ms=-2.0):
    """
    The time of every sample of a waveform, in ms from time 0 (the artifact peak in stage 1).
    Stage 1 (the default, t0_ms=-2.0): preprocess_dbs trims each channel so the (rounded) artifact peak sits at sample
    98 counted from 0, which is round(0.002*fs) at 48.8 kHz. That sample is time 0; the samples before it are negative.
    Stage 2 (BrainVision) epochs start 1 ms before time 0, so pass t0_ms=-1.0.
    Time 0 is put exactly on a sample, number round(-t0_ms/1000*fs) (a waveform shorter than that just never reaches
    it). So the first sample is exactly t0_ms only when
    that many ms is a whole number of samples: at 48.8 kHz the stage 1 axis starts at -2.007 ms, not -2.
    An exact half rounds up, as MATLAB's round does (Python's own round would send 2.5 to 2).
    This only rebuilds the axis from a length and a rate. Where the pipeline that made the waveform knows its real
    time axis (stage 2 especially), pass that one on instead.
    Input:
    n_samples: how many samples the waveform has (2295 for a full stage 1 one at 48.8 kHz)
    fs: sampling freq in Hz, the true one (48828.125 for stage 1), not Sarah's rounded 48828
    t0_ms: roughly when the waveform starts, in ms from time 0. Zero or negative
    Returns: a 1-D array, n_samples long, of times in ms
    Raises ValueError if t0_ms is above 0 (the waveform would start after time 0).
    """
    if t0_ms > 0:
        raise ValueError(f"t0_ms is {t0_ms}; the waveform has to start at or before time 0")
    samples = np.arange(0,n_samples) # 0, 1, 2, ... one per sample
    time_zero = int(np.floor(-t0_ms/1000*fs + 0.5)) # the sample at time zero: 98 for stage 1 at 48.8 kHz
    time_ms = (samples - time_zero)/fs * 1000 # make samples count from artifact peak in ms
    return time_ms


def lowpass_600(segment, fs):
    """
    Sarah's 600 Hz low-pass for the ERNA measures (incrementingExperiment_pull_wave_parameters.m:577-583): a
    4th-order Butterworth run forwards and backwards (filtfilt), so the filter itself adds no time delay.
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


def filtered_erna_window(signal, times, fs, t1=3, t2=20):
    """
    Cuts one post-processed waveform to the ERNA window, t1 to t2 ms after time 0 (3 to 20 ms by default, both ends
    included), then low-passes it at 600 Hz. Sarah cuts first and filters second (M:582-583), so the filter's edge
    effects sit at the window's two ends; the order is kept to match her.
    The caller passes the time axis, so the same function serves stage 1 and stage 2 (see time_axis_ms).
    Input:
    signal: one post-processed channel (1-D), e.g. a row from postprocess_condition
    times: the time of each sample of signal in ms from time 0, e.g. time_axis_ms(len(signal), fs)
    fs: sampling freq in Hz
    t1, t2: the window's start and end in ms
    Returns: the window's times in ms, and the filtered signal over that window (830 samples each for 3 to 20 ms at
    48.8 kHz)
    Raises ValueError if signal or times isn't 1-D, if they differ in length, if t1 isn't before t2, if times has a
    gap, a NaN or a spacing that doesn't match fs, if the waveform doesn't reach both ends of the window (e.g. a
    burst_1..9 average, which stops before 20 ms), if the window is too short to filter, or if the signal has a NaN
    or infinite value inside the window.
    """
    signal = np.asarray(signal)
    t = np.asarray(times)
    #fail loudly: a channels x time array would be filtered along the wrong direction without a word
    if signal.ndim != 1 or t.ndim != 1:
        raise ValueError(f"signal and times have to be 1-D (one channel); got shapes {signal.shape} and {t.shape}")
    #fail loudly: a mask from a times array of another length would pick the wrong samples or error unclearly
    if len(signal) != len(t):
        raise ValueError(f"signal has {len(signal)} samples but times has {len(t)}")
    if not t1 < t2:
        raise ValueError(f"the window start ({t1} ms) has to be before its end ({t2} ms)")
    if len(t) == 0:
        raise ValueError(f"the signal is empty, too short for the {t1} to {t2} ms window")
    #fail loudly: the filter treats neighbouring samples as one sample period apart. A gap, a NaN or an axis made
    #with another sampling rate would break that silently
    one_sample_ms = 1000/fs
    #(0.1% leeway: plenty for rounding in the times, far less than a missing sample or another sampling rate)
    if not np.all(np.isfinite(t)) or not np.allclose(np.diff(t), one_sample_ms, rtol=1e-3, atol=0):
        raise ValueError(f"times has to step evenly by one sample ({one_sample_ms:.5f} ms at fs={fs}), with no gaps "
                         f"or NaN")
    #fail loudly: a waveform that stops early would otherwise be measured over a shorter window without a word.
    #One sample of slack, because the samples rarely land exactly on t1 and t2 (less a sliver, so rounding error in
    #the sums can't let a missing edge sample through)
    slack_ms = one_sample_ms * (1 - 1e-6)
    if t[0] >= t1 + slack_ms or t[-1] <= t2 - slack_ms:
        raise ValueError(f"the signal covers {t[0]:.3f} to {t[-1]:.3f} ms, too short for the {t1} to {t2} ms window")
    window = (t1 <= t)  & (t <= t2) # True for the samples from t1 to t2 ms
    #fail loudly: the filter pads each end with 12 samples taken from the signal, so it needs more than 12
    if np.count_nonzero(window) <= 12:
        raise ValueError(f"the {t1} to {t2} ms window holds {np.count_nonzero(window)} samples; the 600 Hz filter "
                         f"needs more than 12")
    windowed_time = t[window]
    windowed_signal = signal[window]
    #fail loudly: one NaN would spread through the filter, give a NaN RMS and no peaks at all, which reads as "no ERNA"
    if not np.all(np.isfinite(windowed_signal)):
        raise ValueError(f"the signal has NaN or infinite values inside the {t1} to {t2} ms window")
    filtered_window = lowpass_600(windowed_signal,fs) # cut first, then filter (her order)
    return windowed_time, filtered_window


def erna_rms(signal, times, fs):
    """
    Sarah's rms_20 (M:604, 626): the root mean square of the 600 Hz low-passed signal from 3 to 20 ms after time 0
    (the artifact peak in stage 1). It is measured on the filtered signal, not the post-processed one (open question
    Q3). This is the stage 1 measure; Sarah's stage 2 script used a different filter and window, not ported here.
    Input:
    signal: one post-processed channel (1-D)
    times: the time of each sample of signal in ms from time 0, e.g. time_axis_ms(len(signal), fs)
    fs: sampling freq in Hz
    Returns: one number, in the signal's units (volts from preprocess_dbs; Sarah's tables are in mV)
    Raises ValueError for anything filtered_erna_window refuses, e.g. a waveform that stops before 20 ms.
    """
    _, filtered_signal = filtered_erna_window(signal, times, fs) # the window's times aren't needed here
    rms = np.sqrt(np.mean(filtered_signal**2))
    return rms


def find_extrema(times, filtered, fs):
    """
    Finds the peaks and troughs of the filtered ERNA window and lists them in time order, as Sarah does with findpeaks
    on the signal and on its negative (M:586-594). A peak is a sample higher than both neighbours (or a flat top
    with lower samples on both sides), a trough the same upside down; the first and last samples are never either.
    Two peaks (or two troughs) can't be closer than a 600 Hz wave allows: Sarah's MinPeakDistance = round(fs/600), 81
    samples (1.66 ms) at 48.8 kHz. MATLAB keeps peaks MORE than 81 apart and scipy keeps peaks AT LEAST `distance`
    apart, so scipy gets 82. Where two are too close, the taller peak (deeper trough) is kept. The rule is applied to
    peaks and to troughs separately: a peak and a trough can be any distance apart.
    Nothing else filters them (no height or prominence), so small wiggles count. Sarah's later rules (alternation,
    gaps, the first extremum by 7 ms) do the weeding.
    Not yet matched to MATLAB, two things:
    - The detector. On a flat-topped peak scipy picks the middle sample and MATLAB the leftmost, and when two
      equally tall peaks are too close MATLAB keeps the earlier one while scipy doesn't promise which. A
      matlab_findpeaks port is planned; until then this is scipy's answer.
    - The times. These are the true sample times. Sarah rebuilds each time from the frame number with her rounded fs
      and rounds it to 0.01 ms (M:589-594), so an extremum at 7.004 ms here is 7.00 in her table. That matters for
      her "first extremum after 7 ms" and "gap over 3.3 ms" rules; how to handle it is still to be decided.
    Input:
    times: the time of each sample in ms, e.g. the first output of filtered_erna_window
    filtered: the 600 Hz low-passed signal over the same samples (1-D), e.g. its second output
    fs: sampling freq in Hz (sets the minimum distance)
    Returns three arrays, one entry per extremum, in time order (all empty if there are none):
    - the times in ms
    - the values, in the signal's units (troughs keep their real sign, usually negative)
    - the kinds, "peak" or "trough"
    """
    #Sarah's MinPeakDistance is round(fs/600) = 81 samples; +1 because scipy's rule is "at least", MATLAB's "more than".
    #floor(x + 0.5) rounds an exact half up, as MATLAB's round does (Python's round would send 42.5 to 42)
    min_distance = int(np.floor(fs/600 + 0.5))+1
    peak_positions, _ = find_peaks(filtered, distance=min_distance) #positions in filtered; the _ is unused extras
    peak_times, peak_values = times[peak_positions], filtered[peak_positions]
    #find_peaks only finds tops, so flip the signal: its tops are the troughs. The positions are the same either way
    trough_positions, _ = find_peaks(-filtered, distance=min_distance)
    trough_times, trough_values = times[trough_positions], filtered[trough_positions] #read from the unflipped signal
    #one list of everything: all the peaks, then all the troughs
    all_times, all_values = np.concat((peak_times, trough_times)), np.concat((peak_values, trough_values))
    all_kinds = np.array(["peak"]*len(peak_positions) + ["trough"]*len(trough_positions), dtype=str)
    #the positions that put the times in order; reading all three arrays with them keeps each value and kind with its
    #own time. "stable" keeps peaks ahead of troughs if two times were ever equal
    time_order = np.argsort(all_times, kind="stable")
    extrema_times, extrema_values, extrema_kinds = all_times[time_order], all_values[time_order], all_kinds[time_order]
    return extrema_times, extrema_values, extrema_kinds
