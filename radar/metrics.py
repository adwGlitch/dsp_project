"""
radar/metrics.py
Quantitative Performance Evaluation: Target Association, Estimation Errors, and Radar Metrics.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .config import RadarConfig
from .target_model import Target
from .detection import DetectedTarget


@dataclass
class MatchedPair:
    """Association pair between a ground-truth target and an estimated detection."""
    target: Target
    detection: DetectedTarget
    range_error_m: float
    velocity_error_mps: float
    normalized_distance: float


@dataclass
class EvaluationMetrics:
    """
    Comprehensive radar detection performance evaluation.

    Attributes:
        matched_pairs (List[MatchedPair]): Validated ground-truth to detection associations.
        unmatched_targets (List[Target]): Ground-truth targets missed by detection (False Negatives).
        unmatched_detections (List[DetectedTarget]): Detections with no matching target (False Positives).
        detection_rate (float): P_d = N_matched / N_true (0.0 to 1.0).
        false_alarm_count (int): Number of spurious detections.
        range_mae_m (float): Mean Absolute Error in range for matched targets.
        range_rmse_m (float): Root Mean Square Error in range for matched targets.
        velocity_mae_mps (float): Mean Absolute Error in velocity for matched targets.
        velocity_rmse_mps (float): Root Mean Square Error in velocity for matched targets.
    """
    matched_pairs: List[MatchedPair] = field(default_factory=list)
    unmatched_targets: List[Target] = field(default_factory=list)
    unmatched_detections: List[DetectedTarget] = field(default_factory=list)
    detection_rate: float = 0.0
    false_alarm_count: int = 0
    range_mae_m: float = 0.0
    range_rmse_m: float = 0.0
    velocity_mae_mps: float = 0.0
    velocity_rmse_mps: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert matched pairs and errors into a clean pandas DataFrame."""
        rows = []
        for pair in self.matched_pairs:
            rows.append({
                "Target Name": pair.target.name,
                "True Range (m)": pair.target.range_m,
                "Est. Range (m)": pair.detection.range_m,
                "Range Error (m)": round(pair.range_error_m, 3),
                "True Velocity (m/s)": pair.target.velocity_mps,
                "Est. Velocity (m/s)": pair.detection.velocity_mps,
                "Velocity Error (m/s)": round(pair.velocity_error_mps, 3),
                "Detection Power (dB)": pair.detection.power_db,
                "Est. SNR (dB)": pair.detection.snr_estimate_db,
            })
        for miss in self.unmatched_targets:
            rows.append({
                "Target Name": f"[MISSED] {miss.name}",
                "True Range (m)": miss.range_m,
                "Est. Range (m)": np.nan,
                "Range Error (m)": np.nan,
                "True Velocity (m/s)": miss.velocity_mps,
                "Est. Velocity (m/s)": np.nan,
                "Velocity Error (m/s)": np.nan,
                "Detection Power (dB)": np.nan,
                "Est. SNR (dB)": np.nan,
            })
        for fa in self.unmatched_detections:
            rows.append({
                "Target Name": f"[FALSE ALARM #{fa.id}]",
                "True Range (m)": np.nan,
                "Est. Range (m)": fa.range_m,
                "Range Error (m)": np.nan,
                "True Velocity (m/s)": np.nan,
                "Est. Velocity (m/s)": fa.velocity_mps,
                "Velocity Error (m/s)": np.nan,
                "Detection Power (dB)": fa.power_db,
                "Est. SNR (dB)": fa.snr_estimate_db,
            })
        return pd.DataFrame(rows)


def associate_targets(
    targets: List[Target],
    detections: List[DetectedTarget],
    config: RadarConfig,
    max_gate_cells: float = 4.0,
) -> Tuple[List[MatchedPair], List[Target], List[DetectedTarget]]:
    """
    Associate detections with ground-truth targets using Hungarian bipartite matching
    in normalized resolution-cell units.

    Gating Distance Metric:
        d_norm = sqrt( ((R_est - R_true) / delta_R)^2 + ((v_est - v_true) / delta_v)^2 )
    Detections exceeding max_gate_cells are rejected from pairing.

    Args:
        targets: List of ground-truth simulated targets.
        detections: List of detected peaks from radar processor.
        config: RadarConfig defining physical resolution delta_R and delta_v.
        max_gate_cells: Maximum allowable distance in resolution bins for association.

    Returns:
        Tuple of (matched_pairs, unmatched_targets, unmatched_detections).
    """
    if not targets or not detections:
        return [], list(targets), list(detections)

    delta_R = config.range_resolution
    delta_v = config.velocity_resolution

    n_targets = len(targets)
    n_dets = len(detections)

    cost_matrix = np.zeros((n_targets, n_dets))

    for i, t in enumerate(targets):
        for j, d in enumerate(detections):
            dr_norm = (d.range_m - t.range_m) / max(delta_R, 1e-6)
            dv_norm = (d.velocity_mps - t.velocity_mps) / max(delta_v, 1e-6)
            cost_matrix[i, j] = np.sqrt(dr_norm ** 2 + dv_norm ** 2)

    # Solve optimal 1-to-1 assignment via Hungarian algorithm
    target_ind, det_ind = linear_sum_assignment(cost_matrix)

    matched_pairs: List[MatchedPair] = []
    matched_target_indices = set()
    matched_det_indices = set()

    for ti, di in zip(target_ind, det_ind):
        dist = cost_matrix[ti, di]
        if dist <= max_gate_cells:
            t = targets[ti]
            d = detections[di]
            r_err = abs(d.range_m - t.range_m)
            v_err = abs(d.velocity_mps - t.velocity_mps)
            matched_pairs.append(
                MatchedPair(
                    target=t,
                    detection=d,
                    range_error_m=r_err,
                    velocity_error_mps=v_err,
                    normalized_distance=float(dist),
                )
            )
            matched_target_indices.add(ti)
            matched_det_indices.add(di)

    unmatched_targets = [targets[i] for i in range(n_targets) if i not in matched_target_indices]
    unmatched_detections = [detections[j] for j in range(n_dets) if j not in matched_det_indices]

    return matched_pairs, unmatched_targets, unmatched_detections


def evaluate_detections(
    targets: List[Target],
    detections: List[DetectedTarget],
    config: RadarConfig,
    max_gate_cells: float = 4.0,
) -> EvaluationMetrics:
    """
    Compute comprehensive quantitative performance evaluation metrics.

    Args:
        targets: List of simulated targets.
        detections: List of detected peaks.
        config: Active radar configuration.
        max_gate_cells: Gating distance threshold.

    Returns:
        EvaluationMetrics with statistical metrics (MAE, RMSE, P_d, False Alarms).
    """
    matched_pairs, unmatched_targets, unmatched_detections = associate_targets(
        targets, detections, config, max_gate_cells=max_gate_cells
    )

    n_true = len(targets)
    detection_rate = len(matched_pairs) / n_true if n_true > 0 else 1.0
    false_alarm_count = len(unmatched_detections)

    if matched_pairs:
        r_errors = np.array([p.range_error_m for p in matched_pairs])
        v_errors = np.array([p.velocity_error_mps for p in matched_pairs])

        range_mae = float(np.mean(r_errors))
        range_rmse = float(np.sqrt(np.mean(r_errors ** 2)))
        velocity_mae = float(np.mean(v_errors))
        velocity_rmse = float(np.sqrt(np.mean(v_errors ** 2)))
    else:
        range_mae = 0.0
        range_rmse = 0.0
        velocity_mae = 0.0
        velocity_rmse = 0.0

    return EvaluationMetrics(
        matched_pairs=matched_pairs,
        unmatched_targets=unmatched_targets,
        unmatched_detections=unmatched_detections,
        detection_rate=round(detection_rate, 4),
        false_alarm_count=false_alarm_count,
        range_mae_m=round(range_mae, 4),
        range_rmse_m=round(range_rmse, 4),
        velocity_mae_mps=round(velocity_mae, 4),
        velocity_rmse_mps=round(velocity_rmse, 4),
    )

