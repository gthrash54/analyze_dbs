import numpy as np
import pytest

FS = 48828.125
from scipy.signal import butter, filtfilt

import analyze_dbs.erna as erna_module
from analyze_dbs.erna import (UnmeasurableWaveform, _fixed_slots, coastline_index, erna_measures, erna_rms, filtered_erna_window, find_extrema,
                              lowpass_600, time_axis_ms, truncate_extrema, wave_parameters)

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
    #like a burst_1..9 average: it ends at about 7.6 ms, so it never reaches 20 ms
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

def ramp_with(n, spikes):
    """A gently rising line, n samples long, with single-sample spikes at the given positions ({position: value}).
    The slope is there so nothing is flat: find_extrema refuses flat tops and flat bottoms."""
    signal = np.linspace(0, 0.001, n)
    for position, value in spikes.items():
        signal[position] = value
    return signal


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
    filtered = ramp_with(400, {100: 5, 181: 3}) #exactly 81 apart
    _, values, kinds = find_extrema(np.arange(400.0), filtered, FS)
    assert list(values[kinds == "peak"]) == [5] #the taller one stays
    filtered = ramp_with(400, {100: 5, 182: 3}) #82 apart
    _, values, kinds = find_extrema(np.arange(400.0), filtered, FS)
    assert list(values[kinds == "peak"]) == [5, 3]


def test_extrema_distance_follows_fs():
    #at 25 kHz the distance is round(25000/600) = 42 samples: 42 apart loses one, 43 keeps both
    filtered = ramp_with(300, {100: 5, 142: 3})
    _, values, kinds = find_extrema(np.arange(300.0), filtered, 25000)
    assert list(values[kinds == "peak"]) == [5]
    filtered = ramp_with(300, {100: 5, 143: 3})
    _, values, kinds = find_extrema(np.arange(300.0), filtered, 25000)
    assert list(values[kinds == "peak"]) == [5, 3]


def test_extrema_a_peak_and_a_trough_can_be_close():
    #the distance rule is within peaks and within troughs, not between a peak and a trough
    filtered = ramp_with(400, {100: 5, 110: -4})
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
    filtered = ramp_with(300, {100: 5, 143: 3}) #43 apart: one goes
    _, values, kinds = find_extrema(np.arange(300.0), filtered, 25500)
    assert list(values[kinds == "peak"]) == [5]
    filtered = ramp_with(300, {100: 5, 144: 3}) #44 apart: both stay
    _, values, kinds = find_extrema(np.arange(300.0), filtered, 25500)
    assert list(values[kinds == "peak"]) == [5, 3]


def test_extrema_troughs_81_apart_lose_the_shallower_one():
    filtered = ramp_with(400, {100: -5, 181: -3})
    _, values, kinds = find_extrema(np.arange(400.0), filtered, FS)
    assert list(values[kinds == "trough"]) == [-5]


def test_extrema_refuses_a_flat_topped_peak():
    #scipy would say position 4 (the middle), MATLAB position 3 (the left edge)
    filtered = np.array([0, 2, 0, 3, 3, 3, 0, 1, 0.5])
    with pytest.raises(ValueError, match="peak with a flat top at sample 4"):
        find_extrema(np.arange(9.0), filtered, FS)


def test_extrema_refuses_a_flat_bottomed_trough():
    filtered = -np.array([0, 2, 0, 3, 3, 3, 0, 1, 0.5])
    with pytest.raises(ValueError, match="trough with a flat bottom"):
        find_extrema(np.arange(9.0), filtered, FS)


def test_extrema_a_flat_signal_is_not_refused():
    #all zeros has no top or bottom at all (a flat stretch touching the ends is never a peak): no extrema, no error
    ext_times, _, _ = find_extrema(np.arange(100.0), np.zeros(100), FS)
    assert len(ext_times) == 0


#truncate_extrema

def extrema(times, kinds):
    """Hand-made extrema for the truncation tests: the times, made-up values (+1 for a peak, -1 for a trough) and the
    kinds, written "p" or "t" for short."""
    kinds = np.array(["peak" if k == "p" else "trough" for k in kinds], dtype=str)
    values = np.where(kinds == "peak", 1.0, -1.0)
    return np.array(times, dtype=float), values, kinds


def test_truncate_keeps_a_clean_oscillation_whole():
    times, values, kinds = extrema([4, 6, 8, 10, 12], "ptptp")
    kept_times, kept_values, kept_kinds = truncate_extrema(times, values, kinds)
    assert np.array_equal(kept_times, times)
    assert np.array_equal(kept_values, values)
    assert np.array_equal(kept_kinds, kinds)


def test_truncate_a_trough_first_is_fine():
    #her rule is about taking turns, not about which kind starts
    kept_times, _, kept_kinds = truncate_extrema(*extrema([4, 6, 8], "tpt"))
    assert list(kept_times) == [4, 6, 8]
    assert list(kept_kinds) == ["trough", "peak", "trough"]


def test_truncate_fewer_than_two_extrema_keeps_nothing():
    for times, kinds in (([], ""), ([4], "p")):
        kept_times, kept_values, kept_kinds = truncate_extrema(*extrema(times, kinds))
        assert len(kept_times) == 0 and len(kept_values) == 0 and len(kept_kinds) == 0


def test_truncate_a_late_first_extremum_keeps_nothing():
    kept_times, kept_values, kept_kinds = truncate_extrema(*extrema([7.01, 9, 11], "ptp"))
    assert len(kept_times) == 0 and len(kept_values) == 0 and len(kept_kinds) == 0
    #exactly 7 ms is still in time (her test is "> 7")
    kept_times, _, _ = truncate_extrema(*extrema([7.0, 9, 11], "ptp"))
    assert list(kept_times) == [7, 9, 11]


def test_truncate_cuts_at_two_of_a_kind_in_a_row():
    #the second peak in a row (at 10) is the first bad one: it and everything after go, even though 12 and 14 behave
    kept_times, kept_values, kept_kinds = truncate_extrema(*extrema([4, 6, 8, 10, 12, 14], "ptpptp"))
    assert list(kept_times) == [4, 6, 8]
    assert list(kept_values) == [1, -1, 1]
    assert list(kept_kinds) == ["peak", "trough", "peak"]


def test_truncate_cuts_at_a_gap_over_3p3_ms():
    #3.4 ms from 8 to 11.4: the one at 11.4 is the first bad one
    kept_times, _, kept_kinds = truncate_extrema(*extrema([4, 6, 8, 11.4, 13], "ptptp"))
    assert list(kept_times) == [4, 6, 8]
    assert list(kept_kinds) == ["peak", "trough", "peak"]


def test_truncate_a_gap_of_exactly_3p3_ms_is_kept():
    times, values, kinds = extrema([4.0, 7.3, 9.0], "ptp")
    assert times[1] - times[0] == 3.3 #the test only means something if this is exact
    kept_times, _, _ = truncate_extrema(times, values, kinds)
    assert list(kept_times) == [4.0, 7.3, 9.0]


def test_truncate_a_break_at_the_second_extremum_leaves_one():
    #the first extremum always survives rule 2
    kept_times, _, kept_kinds = truncate_extrema(*extrema([4, 6, 8], "ppt"))
    assert list(kept_times) == [4]
    assert list(kept_kinds) == ["peak"]


def test_truncate_uses_the_first_break_only():
    #a gap break at 12.5, then a kind break later: the cut is at the earlier one
    kept_times, _, _ = truncate_extrema(*extrema([4, 6, 8, 12.5, 14, 15], "ptptpp"))
    assert list(kept_times) == [4, 6, 8]


def test_truncate_custom_limits():
    times, values, kinds = extrema([4, 6, 8, 10], "ptpt")
    assert len(truncate_extrema(times, values, kinds, first_max_ms=3.9)[0]) == 0
    assert list(truncate_extrema(times, values, kinds, max_gap_ms=1.0)[0]) == [4]
    assert list(truncate_extrema(times, values, kinds, max_gap_ms=2.0)[0]) == [4, 6, 8, 10]


def test_truncate_after_find_extrema_on_a_decaying_sine():
    times = time_axis_ms(N, FS)
    window_times, filtered = filtered_erna_window(decaying_sine(times), times, FS)
    found = find_extrema(window_times, filtered, FS)
    kept_times, kept_values, kept_kinds = truncate_extrema(*found)
    #300 Hz: first peak at 4.14 ms, 1.67 ms between extrema, strictly taking turns, so all 10 stay
    assert len(kept_times) == 10
    assert np.array_equal(kept_times, found[0])


def test_truncate_exactly_two_extrema():
    #a good pair stays whole
    assert list(truncate_extrema(*extrema([4, 6], "pt"))[0]) == [4, 6]
    #a bad second one leaves the first (there is no second "fewer than 2" check after the cut, as in her code)
    assert list(truncate_extrema(*extrema([4, 6], "pp"))[0]) == [4]
    assert list(truncate_extrema(*extrema([4, 8], "pt"))[0]) == [4]


def test_truncate_a_3p3_ms_gap_survives_rounding_error():
    #66 samples at 20 kHz is 3.3 ms, but 8.3 - 5.0 comes out a hair above 3.3 in floating point
    times, values, kinds = extrema([5.0, 8.3, 10.0], "ptp")
    assert times[1] - times[0] > 3.3
    assert list(truncate_extrema(times, values, kinds)[0]) == [5.0, 8.3, 10.0]
    #a real 3.4 ms gap is still cut
    assert list(truncate_extrema(*extrema([5.0, 8.4, 10.0], "ptp"))[0]) == [5.0]


def test_truncate_accepts_plain_lists():
    #two peaks in a row must still be caught when the kinds arrive as a list, not an array
    kept_times, kept_values, kept_kinds = truncate_extrema([4, 6, 8], [1, 1, -1], ["peak", "peak", "trough"])
    assert list(kept_times) == [4]
    assert list(kept_kinds) == ["peak"]


#_fixed_slots and wave_parameters

def test_fixed_slots_pads_cuts_and_fits():
    assert np.array_equal(_fixed_slots(np.array([1.0, 2.0]), 4), [1, 2, np.nan, np.nan], equal_nan=True) #too few
    assert np.array_equal(_fixed_slots(np.array([1.0, 2.0, 3.0]), 2), [1, 2]) #too many: the first ones
    assert np.array_equal(_fixed_slots(np.array([1.0, 2.0]), 2), [1, 2]) #just right
    assert np.array_equal(_fixed_slots(np.array([]), 3), [np.nan]*3, equal_nan=True) #nothing: all blank


def test_fixed_slots_leaves_its_input_alone():
    numbers = np.array([1.0, 2.0, 3.0])
    box = _fixed_slots(numbers, 2)
    box[0] = 99
    assert list(numbers) == [1, 2, 3]


def test_wave_parameters_by_hand():
    #peak, trough, peak, trough, peak, trough: latencies are the 1st, 3rd and 5th
    times = np.array([4.0, 5.5, 7.0, 8.5, 10.5, 12.0])
    values = np.array([3.0, -2.0, 2.0, -1.0, 1.0, -0.5])
    latencies, amplitudes, intervals, n_peaks, frequency, duration = wave_parameters(times, values)
    blank = np.nan
    assert np.array_equal(latencies, [4.0, 7.0, 10.5, blank, blank, blank, blank, blank], equal_nan=True)
    #swings 1st to 2nd, 3rd to 4th, 5th to 6th
    assert np.array_equal(amplitudes, [5.0, 3.0, 1.5, blank, blank, blank, blank, blank], equal_nan=True)
    assert np.array_equal(intervals, [3.0, 3.5, blank, blank, blank, blank, blank], equal_nan=True)
    assert n_peaks == 3
    assert frequency == pytest.approx(1000/3.25) #the mean interval is 3.25 ms
    assert duration == 10.5 #the last latency, from time 0


def test_wave_parameters_trough_first_measures_the_troughs():
    #Sarah's "peak" latencies are whatever kind comes first (Q5)
    times = np.array([4.0, 5.5, 7.0])
    values = np.array([-2.0, 3.0, -1.0])
    latencies, amplitudes, _, n_peaks, _, _ = wave_parameters(times, values, max_peaks=2)
    assert list(latencies) == [4.0, 7.0]
    assert np.array_equal(amplitudes, [5.0, np.nan], equal_nan=True)
    assert n_peaks == 2


def test_wave_parameters_odd_count_has_one_fewer_amplitude():
    #5 extrema: 3 latencies but only 2 complete swings
    times = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
    values = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    latencies, amplitudes, intervals, n_peaks, _, _ = wave_parameters(times, values, max_peaks=3)
    assert list(latencies) == [4, 6, 8]
    assert np.array_equal(amplitudes, [2, 2, np.nan], equal_nan=True)
    assert list(intervals) == [2, 2]
    assert n_peaks == 3


def test_wave_parameters_the_cap_changes_what_is_stored_not_what_is_measured():
    times = np.arange(4.0, 16.0) #12 extrema, 1 ms apart: 6 latencies, 2 ms apart
    values = np.tile([1.0, -1.0], 6)
    for cap in (4, 8):
        latencies, amplitudes, intervals, n_peaks, frequency, duration = wave_parameters(times, values, max_peaks=cap)
        assert len(latencies) == cap and len(amplitudes) == cap and len(intervals) == cap - 1
        assert n_peaks == 6 #all of them, whatever the cap
        assert frequency == pytest.approx(500)
        assert duration == 14.0 #the 6th latency, even when only 4 are stored
    #with 4, the stored ones are the first 4: the same numbers Sarah keeps
    latencies, _, intervals, _, _, _ = wave_parameters(times, values, max_peaks=4)
    assert list(latencies) == [4, 6, 8, 10]
    assert list(intervals) == [2, 2, 2]


def test_wave_parameters_default_stores_8():
    latencies, amplitudes, intervals, _, _, _ = wave_parameters(np.array([4.0, 5.0]), np.array([1.0, -1.0]))
    assert len(latencies) == 8 and len(amplitudes) == 8 and len(intervals) == 7


def test_wave_parameters_empty_input_is_all_missing_without_warnings(recwarn):
    latencies, amplitudes, intervals, n_peaks, frequency, duration = wave_parameters(np.array([]), np.array([]))
    assert np.all(np.isnan(latencies)) and len(latencies) == 8
    assert np.all(np.isnan(amplitudes)) and len(amplitudes) == 8
    assert np.all(np.isnan(intervals)) and len(intervals) == 7
    assert n_peaks == 0
    assert np.isnan(frequency) and np.isnan(duration)
    assert len(recwarn) == 0 #no "mean of empty slice"


def test_wave_parameters_one_extremum_has_a_latency_and_duration_but_no_frequency(recwarn):
    latencies, amplitudes, intervals, n_peaks, frequency, duration = wave_parameters(np.array([4.5]), np.array([2.0]))
    assert latencies[0] == 4.5 and np.all(np.isnan(latencies[1:]))
    assert np.all(np.isnan(amplitudes)) and np.all(np.isnan(intervals))
    assert n_peaks == 1
    assert np.isnan(frequency)
    assert duration == 4.5
    assert len(recwarn) == 0


def test_wave_parameters_on_a_decaying_300_hz_sine():
    times = time_axis_ms(N, FS)
    window_times, filtered = filtered_erna_window(decaying_sine(times), times, FS)
    kept_times, kept_values, _ = truncate_extrema(*find_extrema(window_times, filtered, FS))
    latencies, amplitudes, intervals, n_peaks, frequency, duration = wave_parameters(kept_times, kept_values)
    assert n_peaks == 5
    assert np.allclose(latencies[:5], [4.14, 7.48, 10.79, 14.13, 17.47], atol=0.01)
    assert np.all(np.isnan(latencies[5:]))
    assert np.all(np.diff(amplitudes[:5]) < 0) #each swing smaller than the last
    #the first swing, worked out separately: the first peak's height plus the first trough's depth
    first_peak = filtered[np.argmin(np.abs(window_times - 4.137))]
    first_trough = filtered[np.argmin(np.abs(window_times - 5.796))]
    assert first_peak > 0.5 and first_trough < -0.4
    assert amplitudes[0] == pytest.approx(first_peak - first_trough)
    assert np.all(np.isnan(amplitudes[5:])) and np.all(np.isnan(intervals[4:]))
    assert np.allclose(intervals[:4], 1000/300, atol=0.03)
    assert frequency == pytest.approx(300, abs=0.5)
    assert duration == latencies[4]


def test_wave_parameters_frequency_and_amplitudes_use_everything_past_the_cap():
    #uneven intervals (2, 2, 2, 4, 4 ms), so a frequency from only the stored ones would come out different
    times = np.array([4.0, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 19])
    values = np.array([6.0, -6, 5, -5, 4, -4, 3, -3, 2, -2, 1, -1])
    latencies, amplitudes, intervals, n_peaks, frequency, duration = wave_parameters(times, values, max_peaks=4)
    assert list(latencies) == [4, 6, 8, 10]
    assert list(amplitudes) == [12, 10, 8, 6] #the first 4 swings; the 5th and 6th (4 and 2) aren't stored
    assert list(intervals) == [2, 2, 2]
    assert n_peaks == 6
    assert frequency == pytest.approx(1000/2.8) #the mean of all five intervals, not 1000/2
    assert duration == 18.0


def test_wave_parameters_two_extrema():
    #one latency, one complete swing: an amplitude but no interval and no frequency (as in her code)
    latencies, amplitudes, intervals, n_peaks, frequency, duration = wave_parameters(
        np.array([4.0, 5.5]), np.array([1.5, -0.5]), max_peaks=2)
    assert np.array_equal(latencies, [4.0, np.nan], equal_nan=True)
    assert np.array_equal(amplitudes, [2.0, np.nan], equal_nan=True)
    assert np.all(np.isnan(intervals)) and len(intervals) == 1
    assert n_peaks == 1
    assert np.isnan(frequency)
    assert duration == 4.0


def test_wave_parameters_max_peaks_limits():
    times, values = np.array([4.0, 5, 6, 7]), np.array([1.0, -1, 1, -1])
    latencies, amplitudes, intervals, n_peaks, frequency, _ = wave_parameters(times, values, max_peaks=1)
    assert list(latencies) == [4] and list(amplitudes) == [2] and len(intervals) == 0
    assert n_peaks == 2 and frequency == pytest.approx(500) #still measured from everything
    with pytest.raises(ValueError, match="at least 1"):
        wave_parameters(times, values, max_peaks=0)


def test_wave_parameters_accepts_plain_lists():
    latencies, amplitudes, _, n_peaks, _, _ = wave_parameters([4.0, 5.5, 7.0], [3.0, -2.0, 2.0], max_peaks=2)
    assert list(latencies) == [4.0, 7.0]
    assert np.array_equal(amplitudes, [5.0, np.nan], equal_nan=True)
    assert n_peaks == 2


#erna_measures

def measured_columns(max_peaks=8):
    """The peak columns erna_measures should add, in order."""
    return ([f"latencies_{k}" for k in range(1, max_peaks + 1)] + [f"amplitudes_{k}" for k in range(1, max_peaks + 1)]
            + [f"intervals_{k}" for k in range(1, max_peaks)])


def test_erna_measures_decaying_sine_matches_the_functions_it_chains():
    times = time_axis_ms(N, FS)
    signal = decaying_sine(times)
    row = erna_measures(signal, times, FS)
    assert list(row) == ["rms", "reason", "n_peaks", "freq", "duration"] + measured_columns()
    assert row["reason"] == ""
    assert row["rms"] == pytest.approx(erna_rms(signal, times, FS))
    #worked out separately, one function at a time
    window_times, filtered = filtered_erna_window(signal, times, FS)
    kept_times, kept_values, _ = truncate_extrema(*find_extrema(window_times, filtered, FS))
    latencies, amplitudes, intervals, n_peaks, frequency, duration = wave_parameters(kept_times, kept_values)
    assert row["n_peaks"] == n_peaks == 5
    assert row["freq"] == pytest.approx(frequency) and row["duration"] == pytest.approx(duration)
    stored = [row[f"latencies_{k}"] for k in range(1, 9)]
    assert np.array_equal(stored, latencies, equal_nan=True)
    assert np.array_equal([row[f"amplitudes_{k}"] for k in range(1, 9)], amplitudes, equal_nan=True)
    assert np.array_equal([row[f"intervals_{k}"] for k in range(1, 8)], intervals, equal_nan=True)
    assert row["latencies_1"] == pytest.approx(4.14, abs=0.01) and np.isnan(row["latencies_6"])


def test_erna_measures_flat_trace_is_measured_with_no_erna():
    #the plan's test: 0 peaks and RMS 0, with the peak fields missing
    times = time_axis_ms(N, FS)
    row = erna_measures(np.zeros(N), times, FS)
    assert row["rms"] == 0.0
    assert row["n_peaks"] == 0
    assert np.isnan(row["freq"]) and np.isnan(row["duration"])
    assert all(np.isnan(row[c]) for c in measured_columns())
    assert row["reason"].startswith("no ERNA")


def test_erna_measures_short_trace_is_never_measured_and_says_why():
    #the plan's test: RMS missing, with its reason. n_peaks is missing too (NaN), not 0
    times = time_axis_ms(470, FS) #ends at about 7.6 ms, like a burst_1..9 average
    row = erna_measures(np.zeros(470), times, FS)
    assert np.isnan(row["rms"]) and np.isnan(row["n_peaks"])
    assert np.isnan(row["freq"]) and np.isnan(row["duration"])
    assert all(np.isnan(row[c]) for c in measured_columns())
    assert "too short for the 3 to 20 ms window" in row["reason"]
    assert len(row) == 28 #the same columns as a measured row


def test_erna_measures_flat_top_keeps_the_rms(monkeypatch):
    #a flat-topped peak makes find_extrema raise; the RMS was measured first and must survive
    def flat_top(*args, **kwargs):
        raise UnmeasurableWaveform("the signal has a peak with a flat top at sample 5 (3.113 ms)")
    monkeypatch.setattr(erna_module, "find_extrema", flat_top)
    times = time_axis_ms(N, FS)
    signal = decaying_sine(times)
    row = erna_measures(signal, times, FS)
    assert row["rms"] == pytest.approx(erna_rms(signal, times, FS))
    assert np.isnan(row["n_peaks"])
    assert all(np.isnan(row[c]) for c in measured_columns())
    assert "flat top" in row["reason"]


def test_erna_measures_late_first_extremum_is_no_erna(monkeypatch):
    #extrema that start after 7 ms: truncate_extrema keeps nothing, so 0 peaks and the "no ERNA" reason
    def late(window_times, filtered, fs):
        return np.array([7.5, 9.0, 10.5]), np.array([1.0, -1.0, 1.0]), np.array(["peak", "trough", "peak"])
    monkeypatch.setattr(erna_module, "find_extrema", late)
    times = time_axis_ms(N, FS)
    row = erna_measures(decaying_sine(times), times, FS)
    assert row["n_peaks"] == 0
    assert row["reason"].startswith("no ERNA")
    assert row["rms"] > 0 #still measured, as in Sarah's table


def test_erna_measures_max_peaks_sets_the_columns():
    times = time_axis_ms(N, FS)
    signal = decaying_sine(times)
    row4 = erna_measures(signal, times, FS, max_peaks=4)
    assert list(row4) == ["rms", "reason", "n_peaks", "freq", "duration"] + measured_columns(4)
    row8 = erna_measures(signal, times, FS)
    for k in range(1, 5): #the first 4 are the same numbers either way
        assert row4[f"latencies_{k}"] == row8[f"latencies_{k}"]
        assert row4[f"amplitudes_{k}"] == row8[f"amplitudes_{k}"]
    assert row4["n_peaks"] == row8["n_peaks"] == 5 #measured from everything, not just what is stored
    assert row4["freq"] == row8["freq"] and row4["duration"] == row8["duration"]


def test_erna_measures_bad_max_peaks_is_an_error_not_a_reason():
    times = time_axis_ms(N, FS)
    with pytest.raises(ValueError, match="at least 1"):
        erna_measures(decaying_sine(times), times, FS, max_peaks=0)


def test_erna_measures_other_errors_still_stop_the_run():
    #only ValueError becomes a reason: a wrong input type is a bug in the call and must not be hidden
    times = time_axis_ms(N, FS)
    with pytest.raises(TypeError):
        erna_measures(decaying_sine(times), times, "48828")


def test_erna_measures_rows_become_a_table():
    import pandas as pd
    times = time_axis_ms(N, FS)
    rows = [erna_measures(decaying_sine(times), times, FS), erna_measures(np.zeros(N), times, FS),
            erna_measures(np.zeros(470), time_axis_ms(470, FS), FS)]
    table = pd.DataFrame(rows)
    assert table.shape == (3, 28)
    assert list(table["n_peaks"].isna()) == [False, False, True]


def test_erna_measures_real_flat_top_goes_through_the_detector(monkeypatch):
    #a window with a genuine flat-topped peak, so find_extrema itself (not a stand-in) has to refuse it
    window_times = np.arange(3.0, 20.0, 0.5)
    window_signal = np.sin(np.arange(len(window_times)) / 2.0)
    window_signal[5:8] = 2.0 #three equal samples: a flat top
    monkeypatch.setattr(erna_module, "filtered_erna_window", lambda **kwargs: (window_times, window_signal))
    row = erna_measures(np.zeros(N), time_axis_ms(N, FS), FS)
    assert "flat top" in row["reason"]
    assert row["rms"] == pytest.approx(np.sqrt(np.mean(window_signal**2)))
    assert np.isnan(row["n_peaks"])


def test_erna_measures_single_kept_extremum(monkeypatch):
    #the 2nd extremum is 5 ms after the 1st (over 3.3 ms), so only the 1st is kept: one latency, no frequency
    def found(window_times, filtered, fs):
        return np.array([4.0, 9.0, 10.5]), np.array([1.0, -1.0, 1.0]), np.array(["peak", "trough", "peak"])
    monkeypatch.setattr(erna_module, "find_extrema", found)
    times = time_axis_ms(N, FS)
    row = erna_measures(decaying_sine(times), times, FS)
    assert row["n_peaks"] == 1
    assert row["latencies_1"] == 4.0 and row["duration"] == 4.0
    assert np.isnan(row["freq"]) and np.isnan(row["amplitudes_1"]) and np.isnan(row["intervals_1"])
    assert row["reason"] == "" #nothing went wrong; a one-peak ERNA just has no frequency


def test_erna_measures_a_mistake_in_the_call_is_not_a_reason():
    #mismatched lengths are the caller's mistake: they must raise, not become a missing row
    times = time_axis_ms(N, FS)
    with pytest.raises(ValueError, match="2295 samples but times has 2294"):
        erna_measures(decaying_sine(times), times[:-1], FS)
    with pytest.raises(ValueError, match="1-D"):
        erna_measures(np.zeros((2, N)), times, FS)


def test_erna_measures_a_bug_further_down_is_not_a_reason(monkeypatch):
    #a ValueError from inside the chain (not an UnmeasurableWaveform) means a bug, and must stop the run
    def broken(*args, **kwargs):
        raise ValueError("something went wrong inside wave_parameters")
    monkeypatch.setattr(erna_module, "wave_parameters", broken)
    times = time_axis_ms(N, FS)
    with pytest.raises(ValueError, match="something went wrong"):
        erna_measures(decaying_sine(times), times, FS)


def test_unmeasurable_waveform_is_still_a_value_error():
    #so anything written to catch ValueError keeps working
    assert issubclass(UnmeasurableWaveform, ValueError)
    with pytest.raises(UnmeasurableWaveform):
        filtered_erna_window(np.zeros(470), time_axis_ms(470, FS), FS)


#coastline_index

def hand_trace():
    """1 kHz, 0 to 29 ms, so every sample sits on a whole ms and the windows can be worked out by hand."""
    times = np.arange(30.0)
    signal = np.zeros(30)
    signal[3:8] = [0, 2, 1, 3, 0] #3-7 ms: steps 2, 1, 2, 3 -> sum 8 over 4 steps
    signal[2] = signal[8] = 5 #just outside the window: a step into or out of it would add 5
    signal[25] = 10 #just outside the default noise window (25, 29]: it would add a step of 10
    signal[26:30] = [0, 1, 0, 2] #26-29 ms: steps 1, 1, 2 -> mean 4/3
    return signal, times


def test_coastline_matches_a_hand_calculation_with_a_noise_correction():
    #the plan's test: 8 - 4 x 4/3 = 8/3
    signal, times = hand_trace()
    assert coastline_index(signal, times, t1=3, t2=7, noise_ms=4) == pytest.approx(8/3)


def test_coastline_end_ms_moves_the_noise_window():
    #the plan's test: an end time passed in (her latest time for that subject and condition) is used, not this
    #signal's own end. (24, 28] holds 25-28 ms: 10, 0, 1, 0 -> steps 10, 1, 1 -> mean 4 -> 8 - 4 x 4 = -8
    signal, times = hand_trace()
    assert coastline_index(signal, times, t1=3, t2=7, noise_ms=4, end_ms=28) == pytest.approx(-8)
    #negative values are kept as they are (her R code clamps them later, not this function)


def test_coastline_fewer_than_2_noise_samples_is_nan_without_a_warning(recwarn):
    signal, times = hand_trace()
    assert np.isnan(coastline_index(signal, times, t1=3, t2=7, noise_ms=4, end_ms=60)) #no samples at all
    assert np.isnan(coastline_index(signal, times, t1=3, t2=7, noise_ms=0.5)) #one sample (29 ms)
    assert len(recwarn) == 0


def test_coastline_decaying_sine_and_flat_trace():
    times = time_axis_ms(N, FS)
    signal = decaying_sine(times)
    window = signal[(times >= 3) & (times <= 20)]
    noise = signal[times > times[-1] - 7]
    expected = np.abs(np.diff(window)).sum() - (len(window) - 1) * np.abs(np.diff(noise)).mean()
    assert coastline_index(signal, times) == pytest.approx(expected)
    assert coastline_index(signal, times) == pytest.approx(5.811, abs=0.001)
    assert coastline_index(np.zeros(N), times) == 0.0


def test_coastline_scales_with_the_signal():
    #twice the signal, twice every step, twice the index
    times = time_axis_ms(N, FS)
    signal = decaying_sine(times)
    assert coastline_index(2 * signal, times) == pytest.approx(2 * coastline_index(signal, times))


def test_coastline_uses_the_signal_not_a_filtered_copy():
    #a fast 5 kHz wiggle inside 3-20 ms would vanish under the 600 Hz low-pass, but the coastline must count it
    times = time_axis_ms(N, FS)
    signal = decaying_sine(times)
    wiggle = np.where((times >= 3) & (times <= 20), 0.01 * np.sin(2 * np.pi * 5000 * times / 1000), 0.0)
    wiggly = signal + wiggle
    window = wiggly[(times >= 3) & (times <= 20)]
    noise = wiggly[times > times[-1] - 7]
    expected = np.abs(np.diff(window)).sum() - (len(window) - 1) * np.abs(np.diff(noise)).mean()
    assert coastline_index(wiggly, times) == pytest.approx(expected)
    assert coastline_index(wiggly, times) > coastline_index(signal, times) + 0.5 #the wiggle adds about 0.75


def test_coastline_short_trace_is_unmeasurable():
    times = time_axis_ms(470, FS) #ends at about 7.6 ms
    with pytest.raises(UnmeasurableWaveform, match="too short for the 3 to 20 ms coastline window"):
        coastline_index(np.zeros(470), times)


def test_coastline_nan_inside_a_window_is_unmeasurable_but_outside_is_fine():
    times = time_axis_ms(N, FS)
    signal = decaying_sine(times)
    in_window = signal.copy()
    in_window[np.argmin(np.abs(times - 10))] = np.nan
    with pytest.raises(UnmeasurableWaveform, match="NaN or infinite"):
        coastline_index(in_window, times)
    in_noise = signal.copy()
    in_noise[-1] = np.nan
    with pytest.raises(UnmeasurableWaveform, match="NaN or infinite"):
        coastline_index(in_noise, times)
    between = signal.copy()
    between[np.argmin(np.abs(times - 30))] = np.nan #30 ms: in neither window
    assert coastline_index(between, times) == pytest.approx(coastline_index(signal, times))


def test_coastline_mistakes_in_the_call_raise_plain_value_errors():
    times = time_axis_ms(N, FS)
    signal = decaying_sine(times)
    for bad_call, message in ((lambda: coastline_index(signal, times[:-1]), "same length"),
                              (lambda: coastline_index(np.zeros((2, N)), times), "1-D"),
                              (lambda: coastline_index(signal, times, t1=20, t2=3), "has to be before"),
                              (lambda: coastline_index(signal, times[::-1]), "rise")):
        with pytest.raises(ValueError, match=message) as caught:
            bad_call()
        assert not isinstance(caught.value, UnmeasurableWaveform) #a mistake, not a property of the waveform


def test_coastline_accepts_plain_lists():
    signal, times = hand_trace()
    assert coastline_index(list(signal), list(times), t1=3, t2=7, noise_ms=4) == pytest.approx(8/3)


def test_coastline_later_group_end_with_a_partly_filled_noise_window():
    #her end time is the latest across signals, so it can run past this signal's own end. (27, 31] holds only 28 and
    #29 ms here: values 0 and 2 -> one step of 2 -> mean 2 -> 8 - 4 x 2 = 0
    signal, times = hand_trace()
    assert coastline_index(signal, times, t1=3, t2=7, noise_ms=4, end_ms=31) == pytest.approx(0.0)


def test_coastline_broken_time_axis_is_a_mistake_in_the_call():
    times = time_axis_ms(N, FS)
    signal = decaying_sine(times)
    with_nan = times.copy()
    with_nan[500] = np.nan #inside 3-20 ms: the step would jump across the hole
    with_gap = np.delete(times, 500)
    uneven = times.copy()
    uneven[600:] += 0.5 * (times[1] - times[0])
    for bad_times, bad_signal in ((with_nan, signal), (with_gap, np.delete(signal, 500)), (uneven, signal)):
        with pytest.raises(ValueError, match="same step") as caught:
            coastline_index(bad_signal, bad_times)
        assert not isinstance(caught.value, UnmeasurableWaveform)


def test_coastline_bad_noise_window_arguments_raise():
    signal, times = hand_trace()
    for kwargs in ({"noise_ms": 0}, {"noise_ms": -1}, {"noise_ms": np.nan}, {"noise_ms": np.inf}, {"end_ms": np.nan}):
        with pytest.raises(ValueError, match="noise_ms has to be"):
            coastline_index(signal, times, t1=3, t2=7, **kwargs)


def test_coastline_a_window_with_fewer_than_2_samples_is_unmeasurable():
    signal, times = hand_trace()
    with pytest.raises(UnmeasurableWaveform, match="holds 1 samples"):
        coastline_index(signal, times, t1=3, t2=3.5) #only the 3 ms sample
