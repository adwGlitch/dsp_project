"""
tests/test_velocity_estimation.py
Tests for Doppler FFT, velocity resolution, and directional sign convention.
"""

import pytest
import numpy as np

from radar.config import RadarConfig
from radar.target_model import Target
from radar.waveform import simulate_radar_cpi
from radar.signal_processing import RangeDopplerProcessor
from radar.detection import detect_peaks_2d


def test_approaching_target_velocity_sign():
    """Verify approaching target (decreasing range) produces negative velocity sign (Requirement 3)."""
    true_range = 60.0
    true_vel = -20.0  # m/s (approaching, within v_max ~ 32.4 m/s)

    config = RadarConfig(
        bandwidth=150.0e6,
        chirp_duration=25.0e-6,
        chirp_repetition_interval=30.0e-6,
        sampling_rate=10.0e6,
        num_chirps=128,
        snr_db=None,
    )

    targets = [Target(range_m=true_range, velocity_mps=true_vel, name="Approaching Car")]
    cpi = simulate_radar_cpi(config, targets)

    processor = RangeDopplerProcessor(config, n_fft_range=512, n_fft_doppler=256)
    res = processor.process(cpi.if_signal)

    det_res = detect_peaks_2d(res.range_doppler_map_db, res.range_axis, res.velocity_axis, threshold_db=-15.0)

    assert len(det_res.detected_targets) >= 1
    est = det_res.detected_targets[0]

    # Verify sign is strictly negative
    assert est.velocity_mps < 0, f"Expected negative velocity for approaching target, got {est.velocity_mps}"
    # Verify value is accurate within velocity resolution
    vel_err = abs(est.velocity_mps - true_vel)
    assert vel_err < config.velocity_resolution * 1.5, f"Velocity error {vel_err:.2f} m/s exceeds tolerance."


def test_receding_target_velocity_sign():
    """Verify receding target (increasing range) produces positive velocity sign (Requirement 4)."""
    true_range = 40.0
    true_vel = +18.0  # m/s (receding, within v_max ~ 32.4 m/s)

    config = RadarConfig(
        bandwidth=150.0e6,
        chirp_duration=25.0e-6,
        chirp_repetition_interval=30.0e-6,
        sampling_rate=10.0e6,
        num_chirps=128,
        snr_db=None,
    )

    targets = [Target(range_m=true_range, velocity_mps=true_vel, name="Receding Car")]
    cpi = simulate_radar_cpi(config, targets)

    processor = RangeDopplerProcessor(config, n_fft_range=512, n_fft_doppler=256)
    res = processor.process(cpi.if_signal)

    det_res = detect_peaks_2d(res.range_doppler_map_db, res.range_axis, res.velocity_axis, threshold_db=-15.0)

    assert len(det_res.detected_targets) >= 1
    est = det_res.detected_targets[0]

    # Verify sign is strictly positive
    assert est.velocity_mps > 0, f"Expected positive velocity for receding target, got {est.velocity_mps}"
    vel_err = abs(est.velocity_mps - true_vel)
    assert vel_err < config.velocity_resolution * 1.5, f"Velocity error {vel_err:.2f} m/s exceeds tolerance."


def test_stationary_target_zero_velocity():
    """Verify stationary target produces zero Doppler velocity."""
    config = RadarConfig(num_chirps=64, snr_db=None)
    targets = [Target(range_m=30.0, velocity_mps=0.0)]
    cpi = simulate_radar_cpi(config, targets)

    processor = RangeDopplerProcessor(config, n_fft_range=512, n_fft_doppler=128)
    res = processor.process(cpi.if_signal)
    det_res = detect_peaks_2d(res.range_doppler_map_db, res.range_axis, res.velocity_axis, threshold_db=-15.0)

    assert len(det_res.detected_targets) >= 1
    est = det_res.detected_targets[0]
    assert abs(est.velocity_mps) < config.velocity_resolution
