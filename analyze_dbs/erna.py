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
    Two differences from MATLAB:
    - The detector. On a flat-topped peak scipy picks the middle sample and MATLAB the leftmost, and when two
      equally tall peaks are too close MATLAB keeps the earlier one while scipy doesn't promise which. Both need
      neighbouring or competing samples to be exactly equal, which should be very rare in a low-pass filtered
      signal, so no MATLAB-style peak finder was written (decided 2026-11-24). Instead this function refuses any
      signal with a flat-topped peak or flat-bottomed trough. Two separate, exactly equal peaks closer than the
      minimum distance are NOT caught; only the comparison with her outputs would show one.
    - The times. These are the true sample times. Sarah rebuilds each time from the frame number with her rounded fs
      and rounds it to 0.01 ms (M:589-594), so an extremum at 7.004 ms here is 7.00 in her table. That matters for
      her "first extremum after 7 ms" and "gap over 3.3 ms" rules. Decided 2026-11-23: keep the true sample times
      and let the comparison with her outputs name the rows her rounding explains (on the list for Walker).
    Input:
    times: the time of each sample in ms, e.g. the first output of filtered_erna_window
    filtered: the 600 Hz low-passed signal over the same samples (1-D), e.g. its second output
    fs: sampling freq in Hz (sets the minimum distance)
    Returns three arrays, one entry per extremum, in time order (all empty if there are none):
    - the times in ms
    - the values, in the signal's units (troughs keep their real sign, usually negative)
    - the kinds, "peak" or "trough"
    Raises ValueError if any peak has a flat top or any trough a flat bottom (two or more exactly equal samples).
    """
    #fail loudly: scipy would put a flat-topped peak at its middle and MATLAB at its left edge. plateau_size=2 asks
    #scipy for only the peaks whose top is 2 or more samples wide; a real filtered signal has none
    for flipped, what in ((filtered, "peak with a flat top"), (-filtered, "trough with a flat bottom")):
        flat_positions, _ = find_peaks(flipped, plateau_size=2)
        if len(flat_positions) > 0:
            raise ValueError(f"the signal has a {what} at sample {flat_positions[0]} ({times[flat_positions[0]]:.3f} "
                             f"ms): scipy and MATLAB can place it differently, so it isn't measured")
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


def truncate_extrema(times, values, kinds, max_gap_ms=3.3, first_max_ms=7.0):
    """
    Keeps only the extrema Sarah counts as the ERNA (M:601-617), from find_extrema's time-ordered list.
    Two rules, in her order:
    1. No ERNA at all if there are fewer than 2 extrema, or the first one comes after 7 ms (first_max_ms). Nothing is
       kept. An extremum at exactly 7 ms is still in time.
    2. Otherwise walk down the list and stop at the first extremum that is the same kind as the one before it (two
       peaks or two troughs in a row), or more than 3.3 ms after it (max_gap_ms; her comment calls this 150 Hz).
       That one and everything after it are dropped, even if later ones behave again. A gap of exactly 3.3 ms is
       still fine.
    Both limits get a sliver of leeway (a billionth of a ms), so a time that sits exactly on a limit isn't pushed
    over it by rounding error in the subtraction (8.3 - 5.0 comes out as 3.3000000000000007).
    The first extremum is never dropped by rule 2, so the result can be a single extremum.
    One difference from Sarah: these are the true sample times. She applies both rules to times rebuilt with her
    rounded fs and rounded to 0.01 ms, so an extremum at 7.004 ms passes her rule 1 and fails here (see find_extrema;
    on the list for Walker).
    Input:
    times: the extrema times in ms, in time order
    values: their values
    kinds: "peak" or "trough" for each
    max_gap_ms: the longest allowed time between one extremum and the next
    first_max_ms: the latest the first extremum may come
    Returns: the times, values and kinds that are kept, as arrays (the front of each input, possibly all of it or
    none of it). They share memory with the inputs when those are arrays; they are not copies.
    """
    #plain lists would compare as two whole lists below (one True or False for everything), so make them arrays
    times, values, kinds = np.asarray(times), np.asarray(values), np.asarray(kinds)
    leeway_ms = 1e-9 #far below one sample; only there to absorb rounding error at an exact limit
    #rule 1. The length is checked first, so an empty list never reaches times[0]
    if len(times) < 2 or times[0] > first_max_ms + leeway_ms:
        return times[:0], values[:0], kinds[:0]
    #rule 2. True where an extremum is a different kind from the one BEFORE it and within max_gap_ms of it. One slot
    #shorter than the inputs: slot 0 is about extremum 1
    follows_rules_mask = (kinds[1:] != kinds[:-1]) & (np.diff(times) <= max_gap_ms + leeway_ms)
    breaks = np.flatnonzero(~follows_rules_mask) #the slots where a rule is broken
    if len(breaks) == 0: #nothing breaks the rules: keep them all
        return times, values, kinds
    #slot 0 of the mask is about extremum 1, so a break at slot b means extremum b+1 is the bad one: keep 0..b
    n_keep = breaks[0] + 1
    return times[:n_keep], values[:n_keep], kinds[:n_keep]


def _fixed_slots(numbers, n_slots):
    """
    Puts numbers into a box of exactly n_slots slots: the first n_slots of them if there are more, blanks (NaN) after
    them if there are fewer. An array can't grow, so the box is made full-size first and the numbers copied into its
    front. Returns the box (a new array; numbers isn't changed).
    """
    box = np.full(n_slots, np.nan)
    n_stored = min(n_slots, len(numbers)) #how many fit: all of them, or n_slots if there are more
    box[:n_stored] = numbers[:n_stored]
    return box


def wave_parameters(times, values, max_peaks=8):
    """
    Sarah's wave measures from the extrema truncate_extrema kept (M:620-634).
    - Latencies: the time of every other extremum, starting with the first. So they are peak times if a peak came
      first and trough times if a trough did (open question Q5).
    - Amplitudes: the size of the swing from each of those extrema to the next one (1st to 2nd, 3rd to 4th, ...).
    - Inter-peak intervals: the time from each latency to the next, so one full cycle each.
    - Number of peaks: how many latencies there are.
    - Average frequency: 1000 / the mean interval, in Hz. NaN when there are fewer than 2 latencies. Sarah rounds it
      to a whole Hz; this doesn't. Against her tables expect up to 0.5 Hz from that rounding, and sometimes a little
      more because her intervals come from times rounded to 0.01 ms (e.g. a true 299.56 Hz is 299 in her table).
    - Duration: the time of the last latency, counted from time 0, not from the first latency (Q5).
    The number of peaks, the frequency and the duration use EVERY latency (every odd-numbered kept extremum; a
    trailing even one affects none of them). Only what is stored is capped.
    No extrema at all (truncate_extrema kept nothing) is not an error: 0 peaks and everything else NaN.
    Sarah stores 4 latencies, 4 amplitudes and 3 intervals; this stores max_peaks, max_peaks and max_peaks - 1, with
    max_peaks 8 by default (Garrett, 2026-11-24; on the list for Walker).
    The first 4 latencies, 4 amplitudes and 3 intervals are the ones she stores, so comparisons with her tables use
    those. Most waveforms have fewer than 8, so slots 5-8 are usually blank: never drop a row because they are (Q8).
    Input:
    times: the kept extrema times in ms, in time order, e.g. the first output of truncate_extrema
    values: their values, e.g. its second output
    max_peaks: how many latencies and amplitudes to store (and one fewer intervals). At least 1
    Returns, in this order:
    - latencies: exactly max_peaks slots, in ms, NaN where there was no extremum
    - amplitudes: exactly max_peaks slots, in the values' units, NaN where there was none
    - inter-peak intervals: exactly max_peaks - 1 slots, in ms, NaN where there was none
    - number of peaks
    - average frequency in Hz
    - duration in ms
    Raises ValueError if max_peaks is below 1.
    """
    if max_peaks < 1:
        raise ValueError(f"max_peaks is {max_peaks}; it has to be at least 1")
    all_latencies = times[::2] #every other extremum from the first ([::2]: the 3rd number is the step)
    all_amplitudes = np.abs(np.diff(values))[::2] #every swing between neighbours, then every other one
    all_intervals = np.diff(all_latencies) #time between one latency and the next
    #these use ALL the latencies, not the stored ones: the stored arrays have blanks and a fixed length
    n_peaks = len(all_latencies)
    #with 0 or 1 latency there are no intervals to average: say NaN directly, so np.mean isn't asked to average nothing
    if len(all_intervals) == 0:
        average_frequency = np.nan
    else: average_frequency = 1000/np.mean(all_intervals) # Hz from mean interval in ms
    #what gets stored: fixed-length arrays, blank (NaN) where there was nothing. One fewer interval than latencies
    latencies = _fixed_slots(all_latencies, max_peaks)
    amplitudes = _fixed_slots(all_amplitudes, max_peaks)
    inter_peak_intervals = _fixed_slots(all_intervals, max_peaks - 1)
    #the last latency, read from the full list (the stored box usually ends in blanks). NaN if there are no latencies
    if len(all_latencies) == 0:
        duration = np.nan
    else: duration = all_latencies[-1]
    
    return latencies, amplitudes, inter_peak_intervals, n_peaks, average_frequency, duration
