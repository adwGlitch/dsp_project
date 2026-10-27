"""
radar/detection.py
Target Detection Algorithms: Basic 2D Peak Finding and 2D CA-CFAR (Cell-Averaging Constant False Alarm Rate).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy.ndimage import maximum_filter


@dataclass
class DetectedTarget:
    """
    Represents an estimated target extracted from the radar detection pipeline.

    Attributes:
        range_m (float): Estimated range in meters.
        velocity_mps (float): Estimated radial velocity in m/s (v > 0 receding, v < 0 approaching).
        range_bin (int): Index of range bin in the Range-Doppler map.
        doppler_bin (int): Index of Doppler bin in the Range-Doppler map.
        power_db (float): Peak power in dB relative to map maximum.
        snr_estimate_db (float): Estimated local Signal-to-Noise / Signal-to-Clutter ratio in dB.
        id (int): Unique detection index.
    """
    range_m: float
    velocity_mps: float
    range_bin: int
    doppler_bin: int
    power_db: float
    snr_estimate_db: float
    id: int = 0


@dataclass
class DetectionResult:
    """
    Container for the full detection stage outputs.

    Attributes:
        detected_targets (List[DetectedTarget]): List of detected targets.
        detection_mask (np.ndarray): 2D boolean mask where detections occurred, shape (N_range, N_doppler).
        threshold_map (Optional[np.ndarray]): 2D matrix of CFAR detection thresholds (linear or dB).
        algorithm (str): Name of detection algorithm used ("Peak Threshold" or "2D CA-CFAR").
    """
    detected_targets: List[DetectedTarget]
    detection_mask: np.ndarray
    threshold_map: Optional[np.ndarray]
    algorithm: str


def detect_peaks_2d(
    rd_map_db: np.ndarray,
    range_axis: np.ndarray,
    velocity_axis: np.ndarray,
    threshold_db: float = -20.0,
    min_distance_range: int = 3,
    min_distance_doppler: int = 3,
) -> DetectionResult:
    """
    Basic 2D local maxima detection above a relative dB threshold with Non-Maximum Suppression (NMS).

    Args:
        rd_map_db (np.ndarray): 2D Range-Doppler magnitude in dB (peak is 0 dB).
        range_axis (np.ndarray): Calibrated range axis in meters.
        velocity_axis (np.ndarray): Calibrated velocity axis in m/s.
        threshold_db (float): Cutoff in dB below the peak (e.g. -25 dB).
        min_distance_range (int): NMS neighborhood radius in range bins.
        min_distance_doppler (int): NMS neighborhood radius in Doppler bins.

    Returns:
        DetectionResult object containing detected targets and boolean detection mask.
    """
    footprint_size = (2 * min_distance_range + 1, 2 * min_distance_doppler + 1)
    local_max = maximum_filter(rd_map_db, size=footprint_size, mode='constant', cval=-999.0)

    # A cell is a detected peak if it equals the local maximum and exceeds threshold
    peak_mask = (rd_map_db == local_max) & (rd_map_db >= threshold_db)

    # Avoid detecting at the exact edge bin boundary
    peak_mask[0, :] = False
    peak_mask[-1, :] = False
    peak_mask[:, 0] = False
    peak_mask[:, -1] = False

    detected_targets: List[DetectedTarget] = []
    r_indices, d_indices = np.where(peak_mask)

    # Sort detections by descending power
    sorted_order = np.argsort(rd_map_db[r_indices, d_indices])[::-1]

    for det_id, idx in enumerate(sorted_order, start=1):
        r_bin = int(r_indices[idx])
        d_bin = int(d_indices[idx])
        power_val = float(rd_map_db[r_bin, d_bin])

        # Parabolic interpolation for sub-bin range accuracy
        r_est = range_axis[r_bin]
        if 1 <= r_bin < len(range_axis) - 1:
            y1, y2, y3 = rd_map_db[r_bin - 1, d_bin], rd_map_db[r_bin, d_bin], rd_map_db[r_bin + 1, d_bin]
            denom = 2 * (2 * y2 - y1 - y3)
            if abs(denom) > 1e-6:
                delta_r = (y3 - y1) / denom
                if abs(delta_r) < 1.0:
                    delta_r_m = delta_r * (range_axis[1] - range_axis[0])
                    r_est = float(r_est + delta_r_m)

        # Parabolic interpolation for sub-bin velocity accuracy
        v_est = velocity_axis[d_bin]
        if 1 <= d_bin < len(velocity_axis) - 1:
            y1, y2, y3 = rd_map_db[r_bin, d_bin - 1], rd_map_db[r_bin, d_bin], rd_map_db[r_bin, d_bin + 1]
            denom = 2 * (2 * y2 - y1 - y3)
            if abs(denom) > 1e-6:
                delta_v = (y3 - y1) / denom
                if abs(delta_v) < 1.0:
                    delta_v_mps = delta_v * (velocity_axis[1] - velocity_axis[0])
                    v_est = float(v_est + delta_v_mps)

        # SNR estimate relative to threshold
        snr_est = float(power_val - threshold_db)

        detected_targets.append(
            DetectedTarget(
                range_m=round(float(r_est), 3),
                velocity_mps=round(float(v_est), 3),
                range_bin=r_bin,
                doppler_bin=d_bin,
                power_db=round(power_val, 2),
                snr_estimate_db=round(snr_est, 2),
                id=det_id,
            )
        )

    return DetectionResult(
        detected_targets=detected_targets,
        detection_mask=peak_mask,
        threshold_map=np.full_like(rd_map_db, threshold_db),
        algorithm="Peak Threshold NMS",
    )


def ca_cfar_2d(
    rd_map_mag: np.ndarray,
    range_axis: np.ndarray,
    velocity_axis: np.ndarray,
    train_cells_range: int = 6,
    train_cells_doppler: int = 4,
    guard_cells_range: int = 3,
    guard_cells_doppler: int = 2,
    pfa: float = 1e-4,
    offset_db: Optional[float] = None,
) -> DetectionResult:
    """
    2D Cell-Averaging Constant False Alarm Rate (CA-CFAR) Detector.

    Operating Principles:
    - Operates on linear power: P = |rd_map_mag|^2.
    - For each Cell Under Test (CUT), an inner 2D guard zone is excluded to avoid signal leakage.
    - Outer training cells estimate the local background noise floor P_noise.
    - Threshold multiplier:
        alpha = N_train * (pfa^(-1 / N_train) - 1)   (for exponential/Rayleigh noise)
    - Threshold: T_CUT = alpha * P_noise.
    - Edge cells where the full training window cannot be formed are omitted or clamped.

    Args:
        rd_map_mag (np.ndarray): 2D linear magnitude matrix (N_range, N_doppler).
        range_axis (np.ndarray): Calibrated range axis in meters.
        velocity_axis (np.ndarray): Calibrated velocity axis in m/s.
        train_cells_range (int): Number of training cells in range (one side).
        train_cells_doppler (int): Number of training cells in Doppler (one side).
        guard_cells_range (int): Number of guard cells in range (one side).
        guard_cells_doppler (int): Number of guard cells in Doppler (one side).
        pfa (float): Target probability of false alarm (e.g. 1e-4).
        offset_db (Optional[float]): Optional additive linear offset in dB to override analytical alpha.

    Returns:
        DetectionResult with detected targets, threshold map, and detection mask.
    """
    n_rows, n_cols = rd_map_mag.shape
    power_map = rd_map_mag ** 2

    # Dimensions
    Gr = guard_cells_range
    Gd = guard_cells_doppler
    Tr = train_cells_range
    Td = train_cells_doppler

    total_window_r = 2 * (Gr + Tr) + 1
    total_window_d = 2 * (Gd + Td) + 1
    guard_window_r = 2 * Gr + 1
    guard_window_d = 2 * Gd + 1

    num_training_cells = (total_window_r * total_window_d) - (guard_window_r * guard_window_d)
    if num_training_cells <= 0:
        raise ValueError("Invalid CFAR configuration: number of training cells is <= 0.")

    # Calculate analytical CFAR threshold scaling factor alpha
    if offset_db is not None:
        alpha = 10.0 ** (offset_db / 10.0)
    else:
        # Theoretical CA-CFAR multiplier for exponential power distribution
        alpha = num_training_cells * ((pfa ** (-1.0 / num_training_cells)) - 1.0)

    # Use 2D integral image / prefix sum for O(1) box sum per cell
    pad_r = Gr + Tr
    pad_d = Gd + Td

    # Padded power map with edge reflection or zero padding
    padded_power = np.pad(power_map, ((pad_r, pad_r), (pad_d, pad_d)), mode='edge')

    # Integral image
    integral = np.pad(np.cumsum(np.cumsum(padded_power, axis=0), axis=1), ((1, 0), (1, 0)), mode='constant')

    def box_sum(r1, r2, d1, d2):
        # Coordinates in padded space (inclusive [r1, r2], [d1, d2])
        # Integral table index is shifted by 1
        return (
            integral[r2 + 1, d2 + 1]
            - integral[r1, d2 + 1]
            - integral[r2 + 1, d1]
            + integral[r1, d1]
        )

    threshold_map = np.zeros_like(power_map)
    cfar_mask = np.zeros_like(power_map, dtype=bool)

    # Slide across CUT cells
    for r in range(n_rows):
        pr = r + pad_r
        outer_r1, outer_r2 = pr - (Gr + Tr), pr + (Gr + Tr)
        guard_r1, guard_r2 = pr - Gr, pr + Gr

        for d in range(n_cols):
            pd = d + pad_d
            outer_d1, outer_d2 = pd - (Gd + Td), pd + (Gd + Td)
            guard_d1, guard_d2 = pd - Gd, pd + Gd

            outer_sum = box_sum(outer_r1, outer_r2, outer_d1, outer_d2)
            guard_sum = box_sum(guard_r1, guard_r2, guard_d1, guard_d2)

            noise_sum = max(0.0, outer_sum - guard_sum)
            noise_floor = noise_sum / num_training_cells

            t_val = alpha * noise_floor
            threshold_map[r, d] = t_val

            cut_power = power_map[r, d]
            if cut_power > t_val:
                cfar_mask[r, d] = True

    # Apply 2D Non-Maximum Suppression to group cluster detections into distinct target peaks
    local_max = maximum_filter(power_map, size=(3, 3), mode='constant', cval=0.0)
    peak_mask = cfar_mask & (power_map == local_max)

    # Exclude outer edge boundary
    peak_mask[:pad_r, :] = False
    peak_mask[-pad_r:, :] = False
    peak_mask[:, :pad_d] = False
    peak_mask[:, -pad_d:] = False

    detected_targets: List[DetectedTarget] = []
    r_indices, d_indices = np.where(peak_mask)
    sorted_order = np.argsort(power_map[r_indices, d_indices])[::-1]

    # Convert linear power to normalized dB for reporting
    max_p = np.max(power_map) + 1e-12

    for det_id, idx in enumerate(sorted_order, start=1):
        r_bin = int(r_indices[idx])
        d_bin = int(d_indices[idx])
        p_cut = float(power_map[r_bin, d_bin])
        t_cut = float(threshold_map[r_bin, d_bin])

        # Range and velocity estimates
        r_est = range_axis[r_bin]
        v_est = velocity_axis[d_bin]

        # Power in dB relative to maximum
        power_db = 10.0 * np.log10(max(p_cut, 1e-12) / max_p)

        # Estimated SNR above noise floor: 10 * log10(P_cut / P_noise)
        noise_level = t_cut / alpha if alpha > 0 else 1e-12
        snr_est = 10.0 * np.log10(max(p_cut / max(noise_level, 1e-12), 1.0))

        detected_targets.append(
            DetectedTarget(
                range_m=round(float(r_est), 3),
                velocity_mps=round(float(v_est), 3),
                range_bin=r_bin,
                doppler_bin=d_bin,
                power_db=round(power_db, 2),
                snr_estimate_db=round(snr_est, 2),
                id=det_id,
            )
        )

    # Convert threshold map to dB normalized to max linear power for plotting
    thresh_db = 10.0 * np.log10(np.maximum(threshold_map, 1e-12) / max_p)
    thresh_db = np.clip(thresh_db, -60.0, 0.0)

    return DetectionResult(
        detected_targets=detected_targets,
        detection_mask=peak_mask,
        threshold_map=thresh_db,
        algorithm="2D CA-CFAR",
    )

