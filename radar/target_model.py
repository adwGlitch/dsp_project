"""
radar/target_model.py
Target modeling and scenario presets for FMCW Radar simulation.

Conventions:
- Range (R): Distance from radar in meters (R >= 0).
- Radial Velocity (v): Rate of change of range dR/dt in m/s:
    * v > 0: Receding target (moving away from radar, distance increasing).
    * v < 0: Approaching target (moving towards radar, distance decreasing).
    * v = 0: Stationary object.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Target:
    """
    Represents a point target in radar range and Doppler space.

    Attributes:
        range_m (float): Target radial distance in meters (R >= 0).
        velocity_mps (float): Target radial velocity in m/s (v > 0: receding, v < 0: approaching).
        rcs_or_amplitude (float): Relative linear amplitude scale factor of reflection (default: 1.0).
        name (str): Descriptive label for the target.
    """
    range_m: float
    velocity_mps: float
    rcs_or_amplitude: float = 1.0
    name: str = "Target"

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if self.range_m < 0:
            raise ValueError(f"Target range cannot be negative, got {self.range_m} m.")
        if self.rcs_or_amplitude <= 0:
            raise ValueError(f"Target amplitude must be positive, got {self.rcs_or_amplitude}.")

    def range_at_time(self, t: float) -> float:
        """Calculate target range at slow-time t: R(t) = R_0 + v * t."""
        return max(0.0, self.range_m + self.velocity_mps * t)

    def delay_at_time(self, t: float, c: float = 299792458.0) -> float:
        """Calculate two-way round-trip delay tau(t) = 2 * R(t) / c."""
        return 2.0 * self.range_at_time(t) / c


@dataclass
class ScenarioPreset:
    """A preconfigured radar test scenario."""
    name: str
    description: str
    targets: List[Target] = field(default_factory=list)
    snr_db: Optional[float] = 15.0


def get_scenario_presets() -> List[ScenarioPreset]:
    """Return standard test scenarios covering various automotive conditions."""
    return [
        ScenarioPreset(
            name="1. Single Approaching Vehicle",
            description="Single car approaching at 72 km/h (-20.0 m/s) at 50 m distance.",
            targets=[
                Target(range_m=50.0, velocity_mps=-20.0, rcs_or_amplitude=1.0, name="Approaching Sedan"),
            ],
            snr_db=20.0,
        ),
        ScenarioPreset(
            name="2. Multi-Target Traffic Scenario",
            description="Lead car moving away (+15 m/s), fast oncoming truck (-24 m/s), and oncoming bike (-10 m/s).",
            targets=[
                Target(range_m=35.0, velocity_mps=15.0, rcs_or_amplitude=1.0, name="Receding Lead Car"),
                Target(range_m=70.0, velocity_mps=-24.0, rcs_or_amplitude=1.2, name="Fast Oncoming Truck"),
                Target(range_m=95.0, velocity_mps=-10.0, rcs_or_amplitude=0.6, name="Distant Oncoming Bike"),
            ],
            snr_db=15.0,
        ),
        ScenarioPreset(
            name="3. Stationary Obstacle + Moving Targets in Heavy Noise",
            description="Stationary barrier (0 m/s), approaching car (-18 m/s), and receding van (+12 m/s) with 8 dB SNR.",
            targets=[
                Target(range_m=25.0, velocity_mps=0.0, rcs_or_amplitude=1.2, name="Stationary Barrier"),
                Target(range_m=65.0, velocity_mps=-18.0, rcs_or_amplitude=0.8, name="Approaching Car"),
                Target(range_m=90.0, velocity_mps=12.0, rcs_or_amplitude=0.7, name="Receding Van"),
            ],
            snr_db=8.0,
        ),
        ScenarioPreset(
            name="4. Range Resolution Test (Two Close Targets)",
            description="Two targets at 48 m and 52 m with the same speed to verify range resolution limits.",
            targets=[
                Target(range_m=48.0, velocity_mps=10.0, rcs_or_amplitude=1.0, name="Target A (48 m)"),
                Target(range_m=52.0, velocity_mps=10.0, rcs_or_amplitude=1.0, name="Target B (52 m)"),
            ],
            snr_db=25.0,
        ),
    ]
