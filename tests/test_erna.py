import numpy as np
import pytest

FS = 48828.125
from scipy.signal import butter, filtfilt

from analyze_dbs.erna import erna_rms, filtered_erna_window, find_extrema, lowpass_600, time_axis_ms

N = 2295 #a full stage 1 post-processed waveform at 48.8 kHz


def decaying_sine(times, hz=300):
    """A made-up ERNA: nothing before time 0, then a sine at hz that dies away."""
    return np.where(times > 0, np.exp(-times/8) * np.sin(2*np.pi*hz*times/1000), 0.0)


#time_axis_ms

def test_time_axis_stage1_zero_is_sample_98():
    times = time_axis_ms(N, FS)
    assert round(0.002*FS) == 98
    assert times[98] == 0.0 #exactly, not nearly
    assert len(times) == N
    assert np.allclose(np.diff(times), 1000/FS) #one sample period between neighbours
    #the start is 98 samples before time 0, which is 2.007 ms, not 2
    assert times[0] == pytest.approx(-98/FS*1000)


def test_time_axis_stage2_starts_at_minus_1ms():
    #at a sampling rate where 1 ms is a whole number of samples, the first sample is exactly -1 ms
    times = time_axis_ms(500, 25000, t0_ms=-1.0)
    assert times[0] == -1.0
    assert times[25] == 0.0


def test_time_axis_rounds_a_half_sample_up_like_matlab():
    #1 ms at 2.5 kHz is 2.5 samples: MATLAB's round gives 3, Python's own round would give 2
    times = time_axis_ms(20, 2500, t0_ms=-1.0)
    assert times[3] == 0.0


def test_time_axis_can_start_at_zero():
    times = time_axis_ms(10, 1000, t0_ms=0)
    assert times[0] == 0.0
    assert times[-1] == 9.0


def test_time_axis_refuses_a_start_after_zero():
    with pytest.raises(ValueError, match="at or before time 0"):
        time_axis_ms(N, FS, t0_ms=2.0)


#lowpass_600

def test_lowpass_600_matches_filtfilt_with_padlen_12():
    rng = np.random.default_rng(0)
    segment = rng.standard_normal(830)
    b, a = butter(4, 600, "lowpass", fs=FS)
    assert np.array_equal(lowpass_600(segment, FS), filtfilt(b, a, segment, padlen=12))
    #scipy's own default padding (15) gives something else, which is why padlen is set
    assert not np.allclose(lowpass_600(segment, FS), filtfilt(b, a, segment))


def test_lowpass_600_passes_slow_and_blocks_fast():
    t = np.arange(4000)/FS
    middle = slice(1000, 3000) #away from the edge effects
    slow = np.sin(2*np.pi*100*t)
    fast = np.sin(2*np.pi*3000*t)
    assert np.allclose(lowpass_600(slow, FS)[middle], slow[middle], atol=1e-3)
    assert np.max(np.abs(lowpass_600(fast, FS)[middle])) < 1e-3


#filtered_erna_window

def test_window_edges_are_inclusive():
    #2 kHz, so the samples sit exactly on half ms: the ones at 3 and at 20 both have to be kept
    times = time_axis_ms(120, 2000, t0_ms=-2.0)
    signal = np.arange(120.0)
    window_times, filtered = filtered_erna_window(signal, times, 2000, t1=3, t2=20)
    assert window_times[0] == 3.0
    assert window_times[-1] == 20.0
    assert len(window_times) == 35 #3, 3.5, ... 20
    assert len(filtered) == 35


def test_window_stage1_is_830_samples_cut_then_filtered():
    times = time_axis_ms(N, FS)
    signal = decaying_sine(times)
    window_times, filtered = filtered_erna_window(signal, times, FS)
    assert len(window_times) == 830
    assert window_times[0] >= 3 and window_times[0] - 1000/FS < 3 #the first sample at or after 3 ms
    assert window_times[-1] <= 20 and window_times[-1] + 1000/FS > 20 #the last sample at or before 20 ms
    #cut first, then filter (Sarah's order): the same as filtering the cut piece
    cut = signal[(times >= 3) & (times <= 20)]
    assert np.array_equal(filtered, lowpass_600(cut, FS))
    #and not the same as filtering the whole waveform and cutting after
    filter_first = lowpass_600(signal, FS)[(times >= 3) & (times <= 20)]
    assert not np.allclose(filtered, filter_first)


def test_window_custom_edges():
    times = time_axis_ms(N, FS)
    window_times, filtered = filtered_erna_window(decaying_sine(times), times, FS, t1=5, t2=10)
    #exactly the samples from 5 to 10 ms, no fewer
    assert np.array_equal(window_times, times[(times >= 5) & (times <= 10)])
    assert window_times[0] - 1000/FS < 5 and window_times[-1] + 1000/FS > 10
    assert len(filtered) == len(window_times)


def test_window_refuses_a_signal_that_stops_early():
    #like a burst_1..9 average: about 7.6 ms long, so it never reaches 20 ms
    times = time_axis_ms(470, FS)
    with pytest.raises(ValueError, match="too short for the 3 to 20 ms window"):
        filtered_erna_window(np.zeros(470), times, FS)


def test_window_refuses_a_signal_that_starts_late():
    times = time_axis_ms(N, FS)
    late = times > 5
    with pytest.raises(ValueError, match="too short"):
        filtered_erna_window(np.zeros(N)[late], times[late], FS)


def test_window_accepts_a_signal_ending_on_the_last_window_sample():
    #cut right after the last sample inside 20 ms: the whole window is still there
    times = time_axis_ms(N, FS)
    keep = times <= 20
    window_times, _ = filtered_erna_window(np.zeros(N)[keep], times[keep], FS)
    assert len(window_times) == 830


def test_window_refuses_one_sample_short():
    times = time_axis_ms(N, FS)
    keep = times <= 20 - 1000/FS #the last window sample is missing
    with pytest.raises(ValueError, match="too short"):
        filtered_erna_window(np.zeros(N)[keep], times[keep], FS)


def test_window_refuses_mismatched_lengths():
    times = time_axis_ms(N, FS)
    with pytest.raises(ValueError, match="2294 samples but times has 2295"):
        filtered_erna_window(np.zeros(N - 1), times, FS)


def test_window_refuses_too_few_samples_to_filter():
    times = time_axis_ms(N, FS)
    with pytest.raises(ValueError, match="needs more than 12"):
        filtered_erna_window(np.zeros(N), times, FS, t1=3, t2=3.2) #about 10 samples


def test_window_needs_13_samples_and_refuses_12():
    #2 kHz, samples on the half ms: 3 to 9 ms holds 13 samples, 3 to 8.5 ms holds 12
    times = time_axis_ms(120, 2000)
    window_times, _ = filtered_erna_window(np.zeros(120), times, 2000, t1=3, t2=9)
    assert len(window_times) == 13
    with pytest.raises(ValueError, match="holds 12 samples"):
        filtered_erna_window(np.zeros(120), times, 2000, t1=3, t2=8.5)


def test_window_refuses_a_2d_signal():
    times = time_axis_ms(N, FS)
    with pytest.raises(ValueError, match="1-D"):
        filtered_erna_window(np.zeros((N, 14)), times, FS)


def test_window_refuses_times_with_a_gap_or_nan():
    times = time_axis_ms(N, FS)
    gap = np.delete(times, 500) #one sample missing in the middle
    with pytest.raises(ValueError, match="step evenly"):
        filtered_erna_window(np.zeros(N - 1), gap, FS)
    bad = times.copy()
    bad[500] = np.nan
    with pytest.raises(ValueError, match="step evenly"):
        filtered_erna_window(np.zeros(N), bad, FS)


def test_window_refuses_times_made_with_another_fs():
    times = time_axis_ms(N, 25000) #axis from one rate, filter told another
    with pytest.raises(ValueError, match="step evenly"):
        filtered_erna_window(np.zeros(N), times, FS)


def test_window_refuses_nan_in_the_signal_inside_the_window():
    times = time_axis_ms(N, FS)
    signal = decaying_sine(times)
    signal[np.argmin(np.abs(times - 10))] = np.nan #one bad sample at 10 ms
    with pytest.raises(ValueError, match="NaN or infinite"):
        filtered_erna_window(signal, times, FS)
    with pytest.raises(ValueError, match="NaN or infinite"):
        erna_rms(signal, times, FS)
    #a bad sample outside the window doesn't matter
    signal = decaying_sine(times)
    signal[-1] = np.nan #at 45 ms
    filtered_erna_window(signal, times, FS)


def test_window_accepts_float32_times():
    times = time_axis_ms(N, FS).astype(np.float32) #rounder numbers, same axis
    window_times, _ = filtered_erna_window(np.zeros(N), times, FS)
    assert len(window_times) == 830


def test_window_refuses_reversed_edges():
    times = time_axis_ms(N, FS)
    with pytest.raises(ValueError, match="has to be before"):
        filtered_erna_window(np.zeros(N), times, FS, t1=20, t2=3)


def test_window_missing_edge_sample_is_caught_despite_rounding():
    #5 kHz, samples every 0.2 ms starting at 0.3: the one at 0.1 ms is missing. 0.1 + 0.2 comes out a hair above 0.3
    #in floating point, which let this through before the check got its sliver of tolerance
    times = 0.3 + np.arange(200)/5000*1000
    with pytest.raises(ValueError, match="too short"):
        filtered_erna_window(np.zeros(200), times, 5000, t1=0.1, t2=20)


def test_window_refuses_an_empty_signal():
    with pytest.raises(ValueError, match="is empty"):
        filtered_erna_window(np.array([]), np.array([]), FS)


#erna_rms

def test_rms_matches_an_independent_calculation():
    times = time_axis_ms(N, FS)
    signal = np.sin(2*np.pi*300*times/1000) #a steady 300 Hz sine
    #worked out separately on the same cut, filtered samples
    cut = signal[(times >= 3) & (times <= 20)]
    b, a = butter(4, 600, "lowpass", fs=FS)
    expected = np.sqrt(np.mean(filtfilt(b, a, cut, padlen=12)**2))
    assert erna_rms(signal, times, FS) == pytest.approx(expected, rel=1e-12)
    #close to a sine's 1/sqrt(2), but not exactly: the filter trims 300 Hz a little and the cut isn't whole cycles
    assert erna_rms(signal, times, FS) == pytest.approx(1/np.sqrt(2), rel=0.05)


def test_rms_scales_with_the_signal_and_is_zero_for_a_flat_one():
    times = time_axis_ms(N, FS)
    signal = decaying_sine(times)
    assert erna_rms(3*signal, times, FS) == pytest.approx(3*erna_rms(signal, times, FS))
    assert erna_rms(np.zeros(N), times, FS) == 0.0


def test_rms_stage2_axis_measures_the_same_window():
    #the same samples described with a stage 2 style axis (starting at -1 ms) give the same answer as slicing by hand
    fs = 25000
    times = time_axis_ms(1500, fs, t0_ms=-1.0)
    signal = decaying_sine(times)
    cut = signal[(times >= 3) & (times <= 20)]
    assert erna_rms(signal, times, fs) == pytest.approx(np.sqrt(np.mean(lowpass_600(cut, fs)**2)))


#find_extrema

def test_extrema_of_a_decaying_sine_alternate_in_time_order():
    times = time_axis_ms(N, FS)
    window_times, filtered = filtered_erna_window(decaying_sine(times), times, FS)
    ext_times, ext_values, ext_kinds = find_extrema(window_times, filtered, FS)
    #300 Hz from 3 to 20 ms: 5 peaks and 5 troughs, strictly taking turns, peak first
    assert list(ext_kinds) == ["peak", "trough"]*5
    assert np.all(np.diff(ext_times) > 0) #in time order
    assert np.allclose(ext_times, [4.14, 5.80, 7.48, 9.13, 10.79, 12.47, 14.13, 15.79, 17.47, 19.15], atol=0.01)
    #each value stayed with its own time and kind: peaks above zero, troughs below, and the wave dies away
    assert np.all(ext_values[ext_kinds == "peak"] > 0)
    assert np.all(ext_values[ext_kinds == "trough"] < 0)
    assert np.all(np.diff(np.abs(ext_values)) < 0)
    #and each value is the signal at that time
    for when, value in zip(ext_times, ext_values):
        assert value == filtered[window_times == when][0]


def test_extrema_values_are_read_at_their_own_positions():
    #hand-made: a peak at position 2, a trough at 5, a peak at 8
    filtered = np.array([0, 1, 4, 1, 0, -3, 0, 1, 2, 1, 0.0])
    times = np.arange(11)*10.0
    ext_times, ext_values, ext_kinds = find_extrema(times, filtered, 600) #fs 600: the distance is 2 samples
    assert list(ext_times) == [20, 50, 80]
    assert list(ext_values) == [4, -3, 2]
    assert list(ext_kinds) == ["peak", "trough", "peak"]


def test_extrema_a_trough_above_zero_is_still_a_trough():
    #the whole wave rides above zero, so the sign of the value can't say what it is
    filtered = np.array([5, 9, 6, 8, 5.0])
    ext_times, ext_values, ext_kinds = find_extrema(np.arange(5.0), filtered, 600)
    assert list(ext_values) == [9, 6, 8]
    assert list(ext_kinds) == ["peak", "trough", "peak"]


def test_extrema_81_apart_loses_the_smaller_peak_82_keeps_both():
    #Sarah's MinPeakDistance=81 keeps peaks MORE than 81 samples apart
    filtered = np.zeros(400)
    filtered[100], filtered[181] = 5, 3 #exactly 81 apart
    _, values, kinds = find_extrema(np.arange(400.0), filtered, FS)
    assert list(values[kinds == "peak"]) == [5] #the taller one stays
    filtered = np.zeros(400)
    filtered[100], filtered[182] = 5, 3 #82 apart
    _, values, kinds = find_extrema(np.arange(400.0), filtered, FS)
    assert list(values[kinds == "peak"]) == [5, 3]


def test_extrema_distance_follows_fs():
    #at 25 kHz the distance is round(25000/600) = 42 samples: 42 apart loses one, 43 keeps both
    filtered = np.zeros(300)
    filtered[100], filtered[142] = 5, 3
    _, values, kinds = find_extrema(np.arange(300.0), filtered, 25000)
    assert list(values[kinds == "peak"]) == [5]
    filtered = np.zeros(300)
    filtered[100], filtered[143] = 5, 3
    _, values, kinds = find_extrema(np.arange(300.0), filtered, 25000)
    assert list(values[kinds == "peak"]) == [5, 3]


def test_extrema_a_peak_and_a_trough_can_be_close():
    #the distance rule is within peaks and within troughs, not between a peak and a trough
    filtered = np.zeros(400)
    filtered[100], filtered[110] = 5, -4
    ext_times, _, ext_kinds = find_extrema(np.arange(400.0), filtered, FS)
    assert list(ext_times) == [100, 110]
    assert list(ext_kinds) == ["peak", "trough"]


def test_extrema_none_found_gives_three_empty_arrays():
    ramp = np.arange(50.0) #always rising: no tops, no bottoms, and the ends never count
    ext_times, ext_values, ext_kinds = find_extrema(np.arange(50.0), ramp, FS)
    assert len(ext_times) == 0 and len(ext_values) == 0 and len(ext_kinds) == 0
    assert ext_kinds.dtype.kind == "U" #still text, so later comparisons with "peak" work


def test_extrema_distance_rounds_a_half_up_like_matlab():
    #25500/600 = 42.5: MATLAB's round gives 43, so peaks have to be more than 43 apart (Python's round would give 42)
    filtered = np.zeros(300)
    filtered[100], filtered[143] = 5, 3 #43 apart: one goes
    _, values, kinds = find_extrema(np.arange(300.0), filtered, 25500)
    assert list(values[kinds == "peak"]) == [5]
    filtered = np.zeros(300)
    filtered[100], filtered[144] = 5, 3 #44 apart: both stay
    _, values, kinds = find_extrema(np.arange(300.0), filtered, 25500)
    assert list(values[kinds == "peak"]) == [5, 3]


def test_extrema_troughs_81_apart_lose_the_shallower_one():
    filtered = np.zeros(400)
    filtered[100], filtered[181] = -5, -3
    _, values, kinds = find_extrema(np.arange(400.0), filtered, FS)
    assert list(values[kinds == "trough"]) == [-5]
