"""
tests/test_waveform.py
Tests for FMCW waveform parameters, signal generation, and parameter validation.
"""

import pytest
import numpy as np

from radar.config import RadarConfig
from radar.target_model import Target
from radar.waveform import generate_chirp_waveform, simulate_radar_cpi, add_awgn


def test_waveform_duration_and_bandwidth():
    """Verify that chirp has the intended duration, sampling points, and slope (Requirement 1)."""
    bandwidth = 150.0e6      # 150 MHz
    chirp_duration = 50.0e-6 # 50 us
    chirp_pri = 60.0e-6      # 60 us
    fs = 10.0e6              # 10 MS/s

    config = RadarConfig(
        bandwidth=bandwidth,
        chirp_duration=chirp_duration,
        chirp_repetition_interval=chirp_pri,
        sampling_rate=fs,
    )

    fast_time, tx_freq, tx_signal = generate_chirp_waveform(config)

    # 1. Check time duration
    assert len(fast_time) == config.num_samples_per_chirp
    assert np.isclose(fast_time[-1] + 1.0 / fs, chirp_duration, atol=1e-10)

    # 2. Check instantaneous frequency ramp
    expected_slope = bandwidth / chirp_duration
    actual_slope = (tx_freq[-1] - tx_freq[0]) / (fast_time[-1] - fast_time[0])
    assert np.isclose(actual_slope, expected_slope, rtol=1e-3)
    assert np.isclose(tx_freq[0], 0.0)
    assert np.isclose(tx_freq[-1] + expected_slope * (1.0 / fs), bandwidth, rtol=1e-2)

    # 3. Check constant envelope of complex analytic chirp
    envelope = np.abs(tx_signal)
    assert np.allclose(envelope, 1.0, atol=1e-6)


def test_invalid_radar_parameters_raise_errors():
    """Verify that invalid configurations produce clear validation errors (Requirement 8)."""
    # Negative bandwidth
    with pytest.raises(ValueError, match="Bandwidth must be positive"):
        RadarConfig(bandwidth=-10e6)

    # Chirp repetition interval shorter than active chirp
    with pytest.raises(ValueError, match="Chirp repetition interval .* cannot be smaller"):
        RadarConfig(chirp_duration=50e-6, chirp_repetition_interval=40e-6)

    # Negative sampling rate
    with pytest.raises(ValueError, match="Sampling rate must be positive"):
        RadarConfig(sampling_rate=-1e6)

    # Too few samples per chirp
    with pytest.raises(ValueError, match="Too few fast-time samples"):
        RadarConfig(chirp_duration=50e-6, chirp_repetition_interval=60e-6, sampling_rate=1e4)  # 1 sample


def test_reproducible_random_seed():
    """Verify simulation reproducibility when a fixed random seed is provided (Requirement 10)."""
    config = RadarConfig(snr_db=10.0, seed=12345)
    targets = [Target(range_m=40.0, velocity_mps=-15.0)]

    cpi1 = simulate_radar_cpi(config, targets, seed=12345)
    cpi2 = simulate_radar_cpi(config, targets, seed=12345)
    cpi3 = simulate_radar_cpi(config, targets, seed=99999)

    # Identical seed must produce bitwise identical IF signals
    np.testing.assert_array_equal(cpi1.if_signal, cpi2.if_signal)

    # Different seed must produce different noisy signal realization
    assert not np.array_equal(cpi1.if_signal, cpi3.if_signal)
