"""
radar/config.py
Configuration and parameter validation for 77 GHz FMCW Automotive Radar Simulation.
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class RadarConfig:
    """
    Radar configuration holding physical, waveform, and processing parameters.

    Attributes:
        carrier_frequency (float): Nominal carrier frequency in Hz (default: 77 GHz).
        c (float): Speed of light in m/s (default: 299,792,458 m/s).
        bandwidth (float): FMCW chirp sweep bandwidth in Hz (default: 150 MHz).
        chirp_duration (float): Active chirp ramp duration T_c in seconds (default: 25 us).
        sampling_rate (float): ADC sampling frequency F_s in Hz (default: 10 MHz).
        num_chirps (int): Number of chirps per coherent processing interval (CPI) (default: 128).
        chirp_repetition_interval (float): Chirp repetition interval T_PRI in seconds (default: 30 us).
        snr_db (Optional[float]): Target signal-to-noise ratio in dB (None for noiseless).
        seed (Optional[int]): Random seed for deterministic simulation reproducibility.
    """
    carrier_frequency: float = 77.0e9          # 77 GHz
    c: float = 299792458.0                     # Speed of light [m/s]
    bandwidth: float = 150.0e6                 # 150 MHz sweep bandwidth
    chirp_duration: float = 25.0e-6            # 25 microseconds active chirp
    sampling_rate: float = 10.0e6              # 10 MS/s baseband ADC rate
    num_chirps: int = 128                      # Chirps per frame / CPI
    chirp_repetition_interval: float = 30.0e-6 # 30 microseconds PRI (>= chirp_duration)
    snr_db: Optional[float] = None             # Configurable SNR in dB
    seed: Optional[int] = 42                   # Reproducibility seed

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """
        Validate all radar parameters against physical, numerical, and sampling constraints.
        Raises ValueError if any constraint is violated.
        """
        if self.carrier_frequency <= 0:
            raise ValueError(f"Carrier frequency must be positive, got {self.carrier_frequency} Hz.")
        if self.c <= 0:
            raise ValueError(f"Speed of light must be positive, got {self.c} m/s.")
        if self.bandwidth <= 0:
            raise ValueError(f"Bandwidth must be positive, got {self.bandwidth} Hz.")
        if self.chirp_duration <= 0:
            raise ValueError(f"Chirp duration must be positive, got {self.chirp_duration} s.")
        if self.sampling_rate <= 0:
            raise ValueError(f"Sampling rate must be positive, got {self.sampling_rate} Hz.")
        if self.chirp_repetition_interval < self.chirp_duration:
            raise ValueError(
                f"Chirp repetition interval ({self.chirp_repetition_interval*1e6:.1f} us) "
                f"cannot be smaller than active chirp duration ({self.chirp_duration*1e6:.1f} us)."
            )
        if self.num_chirps < 2:
            raise ValueError(f"num_chirps must be at least 2 for Doppler processing, got {self.num_chirps}.")

        num_samples = self.num_samples_per_chirp
        if num_samples < 16:
            raise ValueError(
                f"Too few fast-time samples ({num_samples}). Increase sampling_rate or chirp_duration."
            )

        # Check maximum beat frequency Nyquist limit for complex baseband
        # f_b_max = S * (2 * R_max / c) <= F_s
        if self.chirp_slope <= 0:
            raise ValueError("Chirp slope must be positive.")

    @property
    def wavelength(self) -> float:
        """Operating carrier wavelength lambda = c / f_c [m]."""
        return self.c / self.carrier_frequency

    @property
    def chirp_slope(self) -> float:
        """Chirp frequency slope S = B / T_c [Hz/s]."""
        return self.bandwidth / self.chirp_duration

    @property
    def num_samples_per_chirp(self) -> int:
        """Number of fast-time ADC samples per chirp N_s = round(F_s * T_c)."""
        return int(round(self.sampling_rate * self.chirp_duration))

    @property
    def range_resolution(self) -> float:
        """
        Theoretical range resolution delta_R = c / (2 * B) [m].
        Determined solely by the sweep bandwidth B.
        """
        return self.c / (2.0 * self.bandwidth)

    @property
    def max_unambiguous_range(self) -> float:
        """
        Maximum unambiguous range for positive baseband frequency limit (F_s / 2):
        f_b,max = F_s / 2  =>  R_max = (c * F_s) / (4 * S) [m].
        """
        return (self.c * self.sampling_rate) / (4.0 * self.chirp_slope)

    @property
    def velocity_resolution(self) -> float:
        """
        Theoretical velocity resolution delta_v = lambda / (2 * N_chirps * T_PRI) [m/s].
        Determined by the total CPI coherent integration time.
        """
        cpi_duration = self.num_chirps * self.chirp_repetition_interval
        return self.wavelength / (2.0 * cpi_duration)

    @property
    def max_unambiguous_velocity(self) -> float:
        """
        Maximum unambiguous radial velocity v_max = lambda / (4 * T_PRI) [m/s].
        Governed by the chirp repetition frequency PRF = 1 / T_PRI.
        """
        return self.wavelength / (4.0 * self.chirp_repetition_interval)

    @property
    def cpi_duration(self) -> float:
        """Total Coherent Processing Interval (CPI) duration [s]."""
        return self.num_chirps * self.chirp_repetition_interval

    def summary_dict(self) -> dict:
        """Return human-readable parameter dictionary for UI/reports."""
        return {
            "Carrier Frequency (GHz)": self.carrier_frequency / 1e9,
            "Wavelength (mm)": self.wavelength * 1e3,
            "Bandwidth (MHz)": self.bandwidth / 1e6,
            "Chirp Duration (us)": self.chirp_duration * 1e6,
            "Chirp Repetition Interval (us)": self.chirp_repetition_interval * 1e6,
            "Chirp Slope (MHz/us)": (self.chirp_slope / 1e12),
            "Sampling Rate (MS/s)": self.sampling_rate / 1e6,
            "Samples per Chirp": self.num_samples_per_chirp,
            "Chirps per Frame": self.num_chirps,
            "Range Resolution (m)": round(self.range_resolution, 3),
            "Max Unambiguous Range (m)": round(self.max_unambiguous_range, 2),
            "Velocity Resolution (m/s)": round(self.velocity_resolution, 3),
            "Max Unambiguous Velocity (m/s)": round(self.max_unambiguous_velocity, 2),
            "CPI Duration (ms)": round(self.cpi_duration * 1e3, 2),
        }
