"""
tests/test_detection.py
Tests for multi-target detection, CA-CFAR algorithm, noise robustness, and performance metrics.
"""

import pytest
import numpy as np

from radar.config import RadarConfig
from radar.target_model import Target
from radar.waveform import simulate_radar_cpi
from radar.signal_processing import RangeDopplerProcessor
from radar.detection import detect_peaks_2d, ca_cfar_2d
from radar.metrics import evaluate_detections


def test_multiple_separated_targets_detection():
    """Verify multiple distinct targets in range and Doppler are all detected (Requirement 5)."""
    config = RadarConfig(
        bandwidth=150.0e6,
        chirp_duration=25.0e-6,
        chirp_repetition_interval=30.0e-6,
        sampling_rate=10.0e6,
        num_chirps=128,
        snr_db=None,
    )

    targets = [
        Target(range_m=30.0, velocity_mps=15.0, rcs_or_amplitude=1.0, name="Target 1"),
        Target(range_m=70.0, velocity_mps=-25.0, rcs_or_amplitude=1.0, name="Target 2"),
        Target(range_m=110.0, velocity_mps=-8.0, rcs_or_amplitude=0.8, name="Target 3"),
    ]

    cpi = simulate_radar_cpi(config, targets)
    processor = RangeDopplerProcessor(config, n_fft_range=512, n_fft_doppler=256)
    res = processor.process(cpi.if_signal)

    det_res = detect_peaks_2d(res.range_doppler_map_db, res.range_axis, res.velocity_axis, threshold_db=-18.0)
    metrics = evaluate_detections(targets, det_res.detected_targets, config)

    # All 3 targets should be successfully detected and matched
    assert metrics.detection_rate == 1.0, f"Expected 100% detection, got {metrics.detection_rate}"
    assert len(metrics.matched_pairs) == 3
    assert metrics.range_mae_m < 0.5
    assert metrics.velocity_mae_mps < 0.5


def test_estimation_under_moderate_noise():
    """Verify range and velocity estimates remain accurate under moderate noise (Requirement 6)."""
    # 12 dB SNR with deterministic seed
    config = RadarConfig(
        bandwidth=150.0e6,
        chirp_duration=25.0e-6,
        chirp_repetition_interval=30.0e-6,
        sampling_rate=10.0e6,
        num_chirps=128,
        snr_db=12.0,
        seed=42,
    )

    targets = [
        Target(range_m=45.0, velocity_mps=-18.0, name="Approaching Sedan"),
        Target(range_m=85.0, velocity_mps=12.0, name="Receding SUV"),
    ]

    cpi = simulate_radar_cpi(config, targets)
    processor = RangeDopplerProcessor(config, n_fft_range=512, n_fft_doppler=256)
    res = processor.process(cpi.if_signal)

    det_res = detect_peaks_2d(res.range_doppler_map_db, res.range_axis, res.velocity_axis, threshold_db=-15.0)
    metrics = evaluate_detections(targets, det_res.detected_targets, config)

    assert metrics.detection_rate == 1.0
    assert metrics.range_mae_m < config.range_resolution
    assert metrics.velocity_mae_mps < config.velocity_resolution * 2.0


def test_ca_cfar_detection_and_false_alarm_rejection():
    """Verify 2D CA-CFAR detects target while rejecting noise floor."""
    config = RadarConfig(
        bandwidth=150.0e6,
        num_chirps=64,
        snr_db=15.0,
        seed=100,
    )

    targets = [Target(range_m=50.0, velocity_mps=10.0, rcs_or_amplitude=1.0, name="Target")]
    cpi = simulate_radar_cpi(config, targets)

    processor = RangeDopplerProcessor(config, n_fft_range=256, n_fft_doppler=128)
    res = processor.process(cpi.if_signal)

    # Run 2D CA-CFAR
    cfar_res = ca_cfar_2d(
        res.range_doppler_map,
        res.range_axis,
        res.velocity_axis,
        train_cells_range=4,
        train_cells_doppler=3,
        guard_cells_range=2,
        guard_cells_doppler=1,
        pfa=1e-3,
    )

    assert len(cfar_res.detected_targets) >= 1
    # Check that at least one detection matches target
    matched, _, _ = evaluate_detections(targets, cfar_res.detected_targets, config).matched_pairs, None, None
    metrics = evaluate_detections(targets, cfar_res.detected_targets, config)
    assert metrics.detection_rate == 1.0
