"""
FMCW Radar Signal Processing Package
Moving Object Detection Using Digital Signal Processing (DSP)

Carrier Frequency: 77 GHz Automotive Radar
"""

from .config import RadarConfig
from .target_model import Target, ScenarioPreset, get_scenario_presets
from .waveform import (
    generate_chirp_waveform,
    simulate_radar_cpi,
    mix_signals,
    add_awgn,
    RadarCPIResult,
)
from .signal_processing import (
    RangeDopplerProcessor,
    ProcessingResult,
    compute_range_fft,
    compute_doppler_fft,
    compute_range_doppler_map,
)
from .detection import (
    detect_peaks_2d,
    ca_cfar_2d,
    DetectionResult,
    DetectedTarget,
)
from .metrics import (
    evaluate_detections,
    associate_targets,
    EvaluationMetrics,
)

__all__ = [
    "RadarConfig",
    "Target",
    "ScenarioPreset",
    "get_scenario_presets",
    "generate_chirp_waveform",
    "simulate_radar_cpi",
    "mix_signals",
    "add_awgn",
    "RadarCPIResult",
    "RangeDopplerProcessor",
    "ProcessingResult",
    "compute_range_fft",
    "compute_doppler_fft",
    "compute_range_doppler_map",
    "detect_peaks_2d",
    "ca_cfar_2d",
    "DetectionResult",
    "DetectedTarget",
    "evaluate_detections",
    "associate_targets",
    "EvaluationMetrics",
]

