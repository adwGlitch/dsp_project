"""
radar/waveform.py
Waveform generation, target echo modeling, dechirping, and noise addition for FMCW Radar.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .config import RadarConfig
from .target_model import Target


@dataclass
class RadarCPIResult:
    """
    Container for the complete Coherent Processing Interval (CPI) simulation data.

    Attributes:
        config (RadarConfig): Active radar configuration.
        targets (List[Target]): List of simulated targets.
        fast_time (np.ndarray): Fast-time sampling grid within one chirp, shape (N_samples,).
        slow_time (np.ndarray): Slow-time chirp emission timestamps, shape (N_chirps,).
        tx_chirp_sample (np.ndarray): 1D array of one transmitted chirp (complex baseband representation).
        rx_chirp_sample (np.ndarray): 1D array of the composite received echo for the first chirp.
        if_signal_clean (np.ndarray): 2D clean dechirped intermediate frequency (IF) beat matrix (N_samples, N_chirps).
        if_signal (np.ndarray): 2D noisy IF beat matrix (N_samples, N_chirps).
        noise_power (float): Estimated or configured noise variance sigma^2.
        snr_db (Optional[float]): Achieved or configured SNR in dB.
    """
    config: RadarConfig
    targets: List[Target]
    fast_time: np.ndarray
    slow_time: np.ndarray
    tx_chirp_sample: np.ndarray
    rx_chirp_sample: np.ndarray
    if_signal_clean: np.ndarray
    if_signal: np.ndarray
    noise_power: float
    snr_db: Optional[float]


def generate_chirp_waveform(
    config: RadarConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the theoretical single FMCW chirp waveform in complex baseband.

    Note on 77 GHz Representation:
    An ordinary laptop cannot sample a physical 77 GHz RF carrier (which requires
    Nyquist rate > 154 GHz). Instead, the chirp is represented analytically at baseband
    with instantaneous frequency ramp f(t) = S * t from 0 to B. The physical carrier
    phase contribution 2*pi*f_c*tau is preserved in the analytical mixing equations.

    Returns:
        fast_time: Time array [0, T_c) with step 1/F_s, shape (N_samples,).
        tx_freq: Instantaneous baseband chirp frequency S * t [Hz].
        tx_signal: Complex analytic chirp exp(j * pi * S * t^2).
    """
    num_samples = config.num_samples_per_chirp
    fast_time = np.arange(num_samples) / config.sampling_rate
    S = config.chirp_slope

    # Instantaneous frequency ramp: 0 to B
    tx_freq = S * fast_time
    # Complex analytic baseband chirp
    tx_signal = np.exp(1j * np.pi * S * (fast_time ** 2))

    return fast_time, tx_freq, tx_signal


def mix_signals(
    tx_signal: np.ndarray,
    rx_signal: np.ndarray,
) -> np.ndarray:
    """
    Quadrature mixing (dechirping) of received echo with reference transmit signal:
    s_IF(t) = s_tx(t) * conj(s_rx(t))

    Returns:
        Complex intermediate frequency (IF) beat signal.
    """
    return tx_signal * np.conj(rx_signal)


def add_awgn(
    signal: np.ndarray,
    snr_db: Optional[float],
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    Add complex Circularly Symmetric Additive White Gaussian Noise (AWGN) to signal.

    Args:
        signal (np.ndarray): Complex input signal (1D or 2D).
        snr_db (Optional[float]): Desired Signal-to-Noise Ratio in dB. If None, signal is unchanged.
        seed (Optional[int]): Random seed for deterministic reproducibility.

    Returns:
        Tuple of (noisy_signal, noise_power_variance).
    """
    if snr_db is None:
        return signal.copy(), 0.0

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Calculate average signal power
    sig_power = np.mean(np.abs(signal) ** 2)
    if sig_power <= 0:
        return signal.copy(), 0.0

    # Noise power: SNR = 10 * log10(P_sig / P_noise)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear

    # Complex circular AWGN: sigma_real = sigma_imag = sqrt(noise_power / 2)
    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape))

    return signal + noise, float(noise_power)


def simulate_radar_cpi(
    config: RadarConfig,
    targets: List[Target],
    include_doppler_coupling: bool = True,
    seed: Optional[int] = None,
) -> RadarCPIResult:
    """
    Simulate a complete FMCW Radar Coherent Processing Interval (CPI).

    Mathematical Echo Model:
    For target k at initial range R_0 and radial velocity v (with convention:
    v > 0 receding, v < 0 approaching):
    At chirp m (m = 0, ..., N_c - 1) at slow time t_m = m * T_PRI:
        R_k(t_m) = R_{0,k} + v_k * t_m
        tau_k(t_m) = 2 * R_k(t_m) / c

    The analytical dechirped IF beat signal for chirp m and fast-time t is:
        s_IF,k(t, m) = A_k * exp( j * [ 2*pi*f_b*t + 2*pi*f_D*t_m + phi_0 + phi_rvp ] )

    Where:
        - Beat frequency: f_b = S * tau_k(t_m) = (2 * S * R_k(t_m)) / c
        - Doppler frequency: f_D = (2 * v_k) / lambda
        - Constant phase: phi_0 = (4 * pi * R_{0,k}) / lambda
        - Residual video phase: phi_rvp = -pi * S * tau^2 (negligibly small but included)
        - Fast-time Doppler coupling (optional): + (2 * v_k / lambda) * t

    Approximations:
        - "Stop-and-Go" delay within a single chirp: Target displacement during T_c (50 us)
          at 30 m/s is only 1.5 mm, negligible compared to range resolution (1.0 m).
          Slow-time phase accumulation across chirps accurately models Doppler.

    Args:
        config (RadarConfig): Validated radar configuration.
        targets (List[Target]): List of targets to simulate.
        include_doppler_coupling (bool): Whether to include fast-time Doppler coupling.
        seed (Optional[int]): Random seed for noise reproducibility.

    Returns:
        RadarCPIResult containing time axes, sampled signals, clean and noisy IF matrices.
    """
    config.validate()

    num_samples = config.num_samples_per_chirp
    num_chirps = config.num_chirps
    fast_time = np.arange(num_samples) / config.sampling_rate  # shape: (N_s,)
    slow_time = np.arange(num_chirps) * config.chirp_repetition_interval  # shape: (N_c,)

    S = config.chirp_slope
    c = config.c
    lam = config.wavelength
    f_c = config.carrier_frequency

    # Fast-time and slow-time 2D mesh grids
    # T_fast has shape (N_samples, N_chirps), T_slow has shape (N_samples, N_chirps)
    T_fast = fast_time[:, np.newaxis]
    T_slow = slow_time[np.newaxis, :]

    # Initialize composite clean IF matrix: (N_samples, N_chirps)
    if_signal_clean = np.zeros((num_samples, num_chirps), dtype=np.complex128)

    # For visualization of a single chirp
    tx_chirp_sample = np.exp(1j * np.pi * S * (fast_time ** 2))
    rx_chirp_sample = np.zeros(num_samples, dtype=np.complex128)

    for target in targets:
        A = target.rcs_or_amplitude
        R0 = target.range_m
        v = target.velocity_mps

        # Range at each chirp index: R(m) = R0 + v * T_slow
        R_m = R0 + v * T_slow  # shape (1, N_chirps)
        tau_m = 2.0 * R_m / c  # shape (1, N_chirps)

        # Fast-time beat frequency: f_b = S * tau_m
        # Phase from beat frequency: 2*pi * S * tau_m * t
        beat_phase = 2.0 * np.pi * S * tau_m * T_fast

        # Doppler phase across chirps: 2*pi * f_c * tau_m = 4*pi*R(m)/lambda
        # = 4*pi*R0/lambda + 2*pi * (2*v/lambda) * t_m
        doppler_phase = 2.0 * np.pi * f_c * tau_m

        # Residual video phase (RVP): -pi * S * tau^2
        rvp_phase = -np.pi * S * (tau_m ** 2)

        # Optional fast-time Doppler coupling: (2*v/lambda) * t
        if include_doppler_coupling:
            coupling_phase = 2.0 * np.pi * (2.0 * v / lam) * T_fast
        else:
            coupling_phase = 0.0

        total_if_phase = beat_phase + doppler_phase + rvp_phase + coupling_phase
        target_if = A * np.exp(1j * total_if_phase)
        if_signal_clean += target_if

        # First chirp Rx echo for diagnostic visualization
        tau_0 = 2.0 * R0 / c
        # Physical echo in baseband delayed by tau_0 with Doppler
        rx_target_chirp = A * np.exp(
            1j * (np.pi * S * ((fast_time - tau_0) ** 2) - 2.0 * np.pi * f_c * tau_0)
        )
        rx_chirp_sample += rx_target_chirp

    # Add AWGN if SNR is specified
    actual_seed = seed if seed is not None else config.seed
    if_signal, noise_power = add_awgn(if_signal_clean, config.snr_db, seed=actual_seed)

    return RadarCPIResult(
        config=config,
        targets=targets,
        fast_time=fast_time,
        slow_time=slow_time,
        tx_chirp_sample=tx_chirp_sample,
        rx_chirp_sample=rx_chirp_sample,
        if_signal_clean=if_signal_clean,
        if_signal=if_signal,
        noise_power=noise_power,
        snr_db=config.snr_db,
    )

