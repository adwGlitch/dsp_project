"""
tests/test_range_estimation.py
Tests for Range FFT, calibration, resolution, and distance estimation.
"""

import pytest
import numpy as np

from radar.config import RadarConfig
from radar.target_model import Target
from radar.waveform import simulate_radar_cpi
from radar.signal_processing import compute_range_fft, RangeDopplerProcessor
from radar.detection import detect_peaks_2d


def test_stationary_target_range_estimation():
    """Verify stationary target produces range estimate within half a range bin of true distance (Requirement 2)."""
    true_range = 45.2 # meters
    config = RadarConfig(
        bandwidth=150.0e6,      # delta_R = 3e8 / (2 * 150e6) = 1.0 m
        chirp_duration=25.0e-6,
        chirp_repetition_interval=30.0e-6,
        sampling_rate=10.0e6,
        num_chirps=32,
        snr_db=None,            # noiseless for pure algorithm test
    )

    targets = [Target(range_m=true_range, velocity_mps=0.0, name="Stationary Car")]
    cpi = simulate_radar_cpi(config, targets)

    processor = RangeDopplerProcessor(config, n_fft_range=1024, n_fft_doppler=64)
    res = processor.process(cpi.if_signal)

    det_res = detect_peaks_2d(res.range_doppler_map_db, res.range_axis, res.velocity_axis, threshold_db=-20.0)

    assert len(det_res.detected_targets) >= 1
    est_target = det_res.detected_targets[0]

    # Estimation error must be less than theoretical range resolution (1.0 m)
    abs_err = abs(est_target.range_m - true_range)
    assert abs_err < config.range_resolution, f"Error {abs_err:.3f} m exceeds resolution {config.range_resolution} m."
    # With sub-bin interpolation, it should be even closer (< 0.3 m)
    assert abs_err < 0.3, f"Range error too large: {abs_err:.3f} m."


def test_fft_axes_and_units_calibration():
    """Verify that Range and Velocity FFT axes correctly reflect sampling rate and slope (Requirement 9)."""
    config = RadarConfig(
        bandwidth=200.0e6,
        chirp_duration=40.0e-6,
        sampling_rate=10.0e6,
        chirp_repetition_interval=50.0e-6,
        num_chirps=64,
    )

    processor = RangeDopplerProcessor(config, n_fft_range=512, n_fft_doppler=64)
    dummy_signal = np.zeros((config.num_samples_per_chirp, config.num_chirps), dtype=np.complex128)
    res = processor.process(dummy_signal)

    # 1. Range axis starts at 0 and ends near max unambiguous range
    assert np.isclose(res.range_axis[0], 0.0)
    expected_max_range = config.max_unambiguous_range
    assert np.isclose(res.range_axis[-1], expected_max_range * (1.0 - 2.0 / 512), rtol=1e-2)

    # 2. Velocity axis is symmetric around 0 m/s
    expected_max_v = config.wavelength / (4.0 * config.chirp_repetition_interval)
    assert np.isclose(abs(res.velocity_axis[0]), expected_max_v, rtol=0.05)
    assert np.isclose(res.velocity_axis[len(res.velocity_axis) // 2], 0.0, atol=0.1)


def test_bandwidth_range_resolution_relationship():
    """Verify that theoretical range resolution delta_R = c / (2B) scales inversely with B (Requirement 7)."""
    c = 299792458.0

    b1 = 100.0e6
    config1 = RadarConfig(bandwidth=b1)
    res1 = config1.range_resolution
    assert np.isclose(res1, c / (2.0 * b1), rtol=1e-6)
    assert np.isclose(res1, 1.49896, atol=1e-3)

    b2 = 300.0e6
    config2 = RadarConfig(bandwidth=b2)
    res2 = config2.range_resolution
    assert np.isclose(res2, c / (2.0 * b2), rtol=1e-6)
    assert np.isclose(res2, 0.49965, atol=1e-3)

    # Doubling or tripling bandwidth must proportionally improve resolution
    assert np.isclose(res1 / res2, b2 / b1, rtol=1e-6)
