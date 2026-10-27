"""
radar/signal_processing.py
DSP Processing Pipeline: Windowing, Range FFT, Doppler FFT, and Range-Doppler Map Generation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy import signal as sp_signal

from .config import RadarConfig


@dataclass
class ProcessingResult:
    """
    Container for DSP pipeline outputs.

    Attributes:
        range_axis (np.ndarray): Range values in meters [0, R_max), shape (N_range,).
        velocity_axis (np.ndarray): Radial velocity values in m/s [-v_max, +v_max), shape (N_doppler,).
        range_profile_1d (np.ndarray): Single-chirp or coherent average 1D range magnitude in dB, shape (N_range,).
        doppler_spectrum_1d (np.ndarray): Doppler spectrum across chirps for selected range bin, shape (N_doppler,).
        range_doppler_map (np.ndarray): 2D complex or magnitude matrix, shape (N_range, N_doppler).
        range_doppler_map_db (np.ndarray): 2D logarithmic magnitude map (normalized or calibrated dB).
        peak_power_db (float): Peak power value in dB before normalization.
    """
    range_axis: np.ndarray
    velocity_axis: np.ndarray
    range_profile_1d: np.ndarray
    doppler_spectrum_1d: np.ndarray
    range_doppler_map: np.ndarray
    range_doppler_map_db: np.ndarray
    peak_power_db: float


def compute_range_fft(
    if_signal_1d: np.ndarray,
    config: RadarConfig,
    n_fft: Optional[int] = None,
    window_type: str = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D Range FFT on a single chirp IF beat signal.

    Mathematical Basis:
    For dechirped complex beat signal s_IF(t) = exp(j * 2*pi * f_b * t),
    the beat frequency f_b maps to target range via:
        f_b = (2 * S * R) / c  ==>  R = (c * f_b) / (2 * S)

    Args:
        if_signal_1d (np.ndarray): 1D fast-time samples of a single chirp (N_samples,).
        config (RadarConfig): Active radar configuration.
        n_fft (Optional[int]): FFT length (defaults to next power of 2 >= N_samples).
        window_type (str): "hann", "hamming", "blackman", or "rect".

    Returns:
        range_axis: Calibrated distance axis in meters.
        range_spectrum_mag: Magnitude spectrum of Range FFT.
    """
    num_samples = len(if_signal_1d)
    if n_fft is None:
        # Use next power of 2 for FFT efficiency and smooth peak interpolation
        n_fft = int(2 ** np.ceil(np.log2(max(num_samples, 64))))

    # Windowing to suppress sidelobes and spectral leakage
    if window_type.lower() == "hann":
        window = np.hanning(num_samples)
    elif window_type.lower() == "hamming":
        window = np.hamming(num_samples)
    elif window_type.lower() == "blackman":
        window = np.blackman(num_samples)
    elif window_type.lower() == "rect":
        window = np.ones(num_samples)
    else:
        raise ValueError(f"Unknown window type: {window_type}")

    # Coherent gain normalization factor
    window_gain = np.sum(window)
    windowed_signal = if_signal_1d * window

    # Complex analytic FFT along fast-time
    fft_vals = np.fft.fft(windowed_signal, n=n_fft) / (window_gain + 1e-12)

    # Range axis calibration:
    # Frequency f_k = k * (F_s / N_fft) for k = 0, ..., N_fft - 1
    # R_k = (c * f_k) / (2 * S)
    c = config.c
    S = config.chirp_slope
    freq_axis = np.fft.fftfreq(n_fft, d=1.0 / config.sampling_rate)

    # For positive frequencies (complex baseband beat signal)
    # Note: Complex baseband mixer produces positive beat frequencies for positive ranges
    range_axis = (c * freq_axis[:n_fft // 2]) / (2.0 * S)
    range_spectrum_mag = np.abs(fft_vals[:n_fft // 2])

    return range_axis, range_spectrum_mag


def compute_doppler_fft(
    slow_time_vector: np.ndarray,
    config: RadarConfig,
    n_fft: Optional[int] = None,
    window_type: str = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D Doppler FFT across slow-time chirps for a specific range bin.

    Mathematical Basis:
    Doppler phase accumulation across chirps m is:
        phi(m) = 2*pi * f_D * m * T_PRI, where f_D = 2 * v / lambda
    The radial velocity is recovered by:
        v = (lambda * f_D) / 2

    Convention:
        v > 0: Receding target (moving away).
        v < 0: Approaching target (moving closer).

    Args:
        slow_time_vector (np.ndarray): Complex values across chirps at one range bin, shape (N_chirps,).
        config (RadarConfig): Active radar configuration.
        n_fft (Optional[int]): Doppler FFT size.
        window_type (str): Windowing function.

    Returns:
        velocity_axis: Calibrated radial velocity in m/s (centered at zero Doppler).
        doppler_mag: Centered magnitude spectrum.
    """
    num_chirps = len(slow_time_vector)
    if n_fft is None:
        n_fft = int(2 ** np.ceil(np.log2(max(num_chirps, 32))))

    if window_type.lower() == "hann":
        window = np.hanning(num_chirps)
    elif window_type.lower() == "hamming":
        window = np.hamming(num_chirps)
    elif window_type.lower() == "rect":
        window = np.ones(num_chirps)
    else:
        window = np.ones(num_chirps)

    window_gain = np.sum(window)
    windowed = slow_time_vector * window

    # Doppler FFT with fftshift to place 0 velocity in the center
    fft_raw = np.fft.fft(windowed, n=n_fft) / (window_gain + 1e-12)
    fft_shifted = np.fft.fftshift(fft_raw)

    # Doppler frequency axis: [-PRF/2, +PRF/2)
    T_pri = config.chirp_repetition_interval
    freq_doppler = np.fft.fftshift(np.fft.fftfreq(n_fft, d=T_pri))

    # Velocity axis: v = lambda * f_D / 2
    velocity_axis = (config.wavelength * freq_doppler) / 2.0
    doppler_mag = np.abs(fft_shifted)

    return velocity_axis, doppler_mag


def compute_range_doppler_map(
    if_signal_2d: np.ndarray,
    config: RadarConfig,
    n_fft_range: Optional[int] = None,
    n_fft_doppler: Optional[int] = None,
    window_fast: str = "hann",
    window_slow: str = "hann",
    db_dynamic_range: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the complete 2D Range-Doppler Map via 2D FFT.

    Processing Pipeline:
    1. Fast-time Hann window along rows (samples).
    2. Range FFT along axis 0. Keep positive frequency half (analytic signal).
    3. Slow-time Hann window along columns (chirps).
    4. Doppler FFT along axis 1.
    5. fftshift along axis 1 to center zero Doppler velocity.
    6. Convert magnitude to calibrated dB scale with safe handling of near-zero values.

    Args:
        if_signal_2d (np.ndarray): 2D IF beat signal (N_samples, N_chirps).
        config (RadarConfig): Active radar configuration.
        n_fft_range (Optional[int]): Range FFT size.
        n_fft_doppler (Optional[int]): Doppler FFT size.
        window_fast (str): Fast-time window type ("hann", "hamming", "rect").
        window_slow (str): Slow-time window type ("hann", "hamming", "rect").
        db_dynamic_range (float): Max dynamic range in dB below peak to display.

    Returns:
        range_axis: 1D array of ranges [m], shape (N_range_bins,).
        velocity_axis: 1D array of radial velocities [m/s], shape (N_doppler_bins,).
        rd_map_mag: 2D linear magnitude matrix, shape (N_range_bins, N_doppler_bins).
        rd_map_db: 2D logarithmic magnitude map in dB, normalized so peak is 0 dB.
    """
    num_samples, num_chirps = if_signal_2d.shape

    if n_fft_range is None:
        n_fft_range = int(2 ** np.ceil(np.log2(max(num_samples, 64))))
    if n_fft_doppler is None:
        n_fft_doppler = int(2 ** np.ceil(np.log2(max(num_chirps, 32))))

    # 1. Fast-time Window
    if window_fast.lower() == "hann":
        w_fast = np.hanning(num_samples)[:, np.newaxis]
    elif window_fast.lower() == "hamming":
        w_fast = np.hamming(num_samples)[:, np.newaxis]
    else:
        w_fast = np.ones((num_samples, 1))

    # 2. Slow-time Window
    if window_slow.lower() == "hann":
        w_slow = np.hanning(num_chirps)[np.newaxis, :]
    elif window_slow.lower() == "hamming":
        w_slow = np.hamming(num_chirps)[np.newaxis, :]
    else:
        w_slow = np.ones((1, num_chirps))

    # Apply 2D separable window
    window_2d = w_fast * w_slow
    window_gain = np.sum(w_fast) * np.sum(w_slow)
    windowed_signal = if_signal_2d * window_2d

    # 3. Stage 1: Fast-time FFT (Range FFT) along axis 0
    range_fft = np.fft.fft(windowed_signal, n=n_fft_range, axis=0)

    # For positive frequencies (baseband complex IF representation):
    n_range_bins = n_fft_range // 2
    range_fft_pos = range_fft[:n_range_bins, :]

    # 4. Stage 2: Slow-time FFT (Doppler FFT) along axis 1
    rd_complex = np.fft.fft(range_fft_pos, n=n_fft_doppler, axis=1)

    # 5. Shift Doppler axis so 0 Doppler is in the middle
    rd_complex = np.fft.fftshift(rd_complex, axes=1) / (window_gain + 1e-12)

    # 6. Linear Magnitude
    rd_map_mag = np.abs(rd_complex)

    # 7. Safe dB calculation: 20 * log10(mag / max_mag)
    max_val = np.max(rd_map_mag)
    if max_val <= 1e-12:
        rd_map_db = np.zeros_like(rd_map_mag) - db_dynamic_range
    else:
        eps = 1e-12
        rd_map_db = 20.0 * np.log10(np.maximum(rd_map_mag, eps) / max_val)
        rd_map_db = np.clip(rd_map_db, -db_dynamic_range, 0.0)

    # 8. Calibrated Axes
    c = config.c
    S = config.chirp_slope
    T_pri = config.chirp_repetition_interval
    lam = config.wavelength

    # Fast-time frequency grid -> Range axis [m]
    fast_freqs = np.arange(n_range_bins) * (config.sampling_rate / n_fft_range)
    range_axis = (c * fast_freqs) / (2.0 * S)

    # Doppler frequency grid -> Velocity axis [m/s]
    doppler_freqs = np.fft.fftshift(np.fft.fftfreq(n_fft_doppler, d=T_pri))
    velocity_axis = (lam * doppler_freqs) / 2.0

    return range_axis, velocity_axis, rd_map_mag, rd_map_db


class RangeDopplerProcessor:
    """
    High-level processor that runs the complete DSP pipeline on raw CPI data.
    """
    def __init__(
        self,
        config: RadarConfig,
        n_fft_range: Optional[int] = None,
        n_fft_doppler: Optional[int] = None,
        window_fast: str = "hann",
        window_slow: str = "hann",
    ):
        self.config = config
        self.n_fft_range = n_fft_range
        self.n_fft_doppler = n_fft_doppler
        self.window_fast = window_fast
        self.window_slow = window_slow

    def process(self, if_signal: np.ndarray, selected_range_idx: Optional[int] = None) -> ProcessingResult:
        """
        Process the 2D IF signal and extract Range-Doppler representations.

        Args:
            if_signal: 2D array of shape (N_samples, N_chirps).
            selected_range_idx: Index of range bin for 1D Doppler slice. If None, bin with peak power is used.

        Returns:
            ProcessingResult with axes, 1D profiles, and 2D RD map.
        """
        range_axis, velocity_axis, rd_map_mag, rd_map_db = compute_range_doppler_map(
            if_signal,
            self.config,
            n_fft_range=self.n_fft_range,
            n_fft_doppler=self.n_fft_doppler,
            window_fast=self.window_fast,
            window_slow=self.window_slow,
        )

        # 1D Range Profile: Coherent sum or average across chirps
        # Or take the maximum magnitude across Doppler for each range bin
        range_profile_mag = np.max(rd_map_mag, axis=1)
        max_rp = np.max(range_profile_mag)
        if max_rp > 1e-12:
            range_profile_1d_db = 20.0 * np.log10(np.maximum(range_profile_mag, 1e-12) / max_rp)
            range_profile_1d_db = np.clip(range_profile_1d_db, -60.0, 0.0)
        else:
            range_profile_1d_db = np.zeros_like(range_profile_mag) - 60.0

        # Doppler slice across slow time for the selected range bin
        if selected_range_idx is None:
            # Pick range bin of global peak
            peak_r_idx = int(np.unravel_index(np.argmax(rd_map_mag), rd_map_mag.shape)[0])
        else:
            peak_r_idx = min(max(0, selected_range_idx), rd_map_mag.shape[0] - 1)

        doppler_slice_mag = rd_map_mag[peak_r_idx, :]
        max_dp = np.max(doppler_slice_mag)
        if max_dp > 1e-12:
            doppler_spectrum_1d = 20.0 * np.log10(np.maximum(doppler_slice_mag, 1e-12) / max_dp)
            doppler_spectrum_1d = np.clip(doppler_spectrum_1d, -60.0, 0.0)
        else:
            doppler_spectrum_1d = np.zeros_like(doppler_slice_mag) - 60.0

        peak_power_db = float(20.0 * np.log10(np.maximum(np.max(rd_map_mag), 1e-12)))

        return ProcessingResult(
            range_axis=range_axis,
            velocity_axis=velocity_axis,
            range_profile_1d=range_profile_1d_db,
            doppler_spectrum_1d=doppler_spectrum_1d,
            range_doppler_map=rd_map_mag,
            range_doppler_map_db=rd_map_db,
            peak_power_db=peak_power_db,
        )

