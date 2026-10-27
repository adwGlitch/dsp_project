# Technical Project Report: Moving Object Detection Using FMCW Radar Signal Processing

**Academic Discipline:** Electronics and Communication Engineering (ECE) / Digital Signal Processing (DSP)  
**Nominal Operating Frequency:** 77 GHz Automotive Millimeter-Wave Radar  
**Target Applications:** ADAS (Adaptive Cruise Control, Autonomous Emergency Braking, Blind Spot Detection)  

---

## Abstract
This project presents the mathematical formulation, algorithmic implementation, and software simulation of a 77 GHz Frequency-Modulated Continuous-Wave (FMCW) radar system designed for moving object detection in automotive scenarios. Operating under strict physical and sampling constraints, the application simulates millimeter-wave propagation, kinematic target echoes, thermal Gaussian noise, and receiver mixing at complex baseband. A two-stage coherent Digital Signal Processing (DSP) pipeline—consisting of fast-time Range FFT and slow-time Doppler FFT with Hann windowing—transforms raw intermediate-frequency (IF) beat signals into a two-dimensional Range-Doppler Map (RDM). Target detection is realized using two distinct algorithms: 2D Peak Finding with Non-Maximum Suppression (NMS) and an adaptive 2D Cell-Averaging Constant False Alarm Rate (CA-CFAR) detector. Performance is quantitatively verified across multiple scenarios, proving that range and radial velocity are estimated within sub-bin accuracy ($< 0.35\text{ m}$ range error, $< 0.05\text{ m/s}$ velocity error) against simulated ground truth. An engineering dashboard developed in Streamlit provides interactive scenario manipulation, real-time waveform visualization, and Monte Carlo noise sensitivity analysis.

---

## 1. Problem Statement & Motivation
Modern autonomous driving and Advanced Driver Assistance Systems (ADAS) demand round-the-clock perception capabilities. While electro-optical sensors (cameras and LiDAR) degrade under adverse weather (dense fog, heavy rainfall, glare, and dust), millimeter-wave FMCW radar operating in the 76–81 GHz automotive band penetrates atmospheric scatterers due to its small wavelength ($\approx 3.9\text{ mm}$) and high atmospheric transmission window.

However, extracting both range and radial velocity from continuous electromagnetic wave transmissions requires specialized modulation and multi-dimensional Fourier analysis:
1. Conventional continuous wave (CW) radar measures Doppler velocity but provides zero range information due to lack of time markers.
2. Pulsed radar requires extremely high peak transmit power and ultra-fast sub-nanosecond ADCs unsuitable for consumer automotive silicon.
3. FMCW resolves both challenges by linearly sweeping frequency, encoding target distance into an intermediate beat frequency and velocity into inter-chirp phase accumulation.

The objective of this project is to develop an authentic, mathematically sound software implementation of this complete radar processing chain, demonstrating DSP fundamentals through numerical simulation, automated test suites, and interactive visualization.

---

## 2. Main Objectives
1. **FMCW Chirp Waveform Synthesis:** Synthesize linear frequency sweeps ($B = 150\text{ MHz}$, $T_c = 25\,\mu\text{s}$, $S = 6.0\times 10^{12}\text{ Hz/s}$) with physical 77 GHz carrier phase preservation at complex baseband.
2. **Channel & Kinematic Target Modeling:** Model round-trip delay $\tau(t)$, range progression $R(t) = R_0 + v\cdot t$, approaching/receding velocity signs, relative radar cross-section (RCS), and Additive White Gaussian Noise (AWGN).
3. **Quadrature Dechirping:** Implement complex mixing $s_{IF}(t) = s_{tx}(t) \cdot s_{rx}^*(t)$ to extract the baseband beat signal.
4. **Range FFT & Calibration:** Apply fast-time Hann windowing and FFT along sample axes to extract beat frequencies and map them to physical distance ($R = \frac{c f_b}{2S}$).
5. **Doppler FFT & Calibration:** Apply slow-time Hann windowing and centered FFT across chirps to extract Doppler shifts and map them to radial velocity ($v = \frac{\lambda f_D}{2}$).
6. **2D Range-Doppler Map (RDM):** Generate calibrated 2D power spectral maps in logarithmic decibels (dB).
7. **Adaptive Detection (CA-CFAR):** Implement 2D Cell-Averaging Constant False Alarm Rate detection to separate target reflections from noise.
8. **Quantitative Verification:** Formulate automated target association via Hungarian matching and compute statistical error metrics (MAE, RMSE, $P_d$).
9. **Interactive UI:** Deliver an accessible Streamlit web application for educational and demonstration purposes.

---

## 3. Mathematical Radar Signal Model

### 3.1 Transmitted FMCW Waveform
The transmitted signal consists of repeated linear chirps. For active duration $t \in [0, T_c)$:
$$f_{tx}(t) = f_c + S \cdot t$$
where:
- $f_c = 77.0\text{ GHz}$ (RF Carrier)
- $B = 150.0\text{ MHz}$ (Sweep Bandwidth)
- $T_c = 25.0\,\mu\text{s}$ (Active Chirp Duration)
- $S = \frac{B}{T_c} = 6.0 \times 10^{12}\text{ Hz/s}$ (Chirp Slope)

The instantaneous phase is:
$$\phi_{tx}(t) = 2\pi \int_0^t f_{tx}(\xi) d\xi = 2\pi \left(f_c t + \frac{1}{2} S t^2\right)$$
The analytic transmitted signal is:
$$s_{tx}(t) = \exp\left(j 2\pi \left(f_c t + \frac{1}{2} S t^2\right)\right)$$

### 3.2 Target Echo Model
Consider a target with initial range $R_0$ and radial velocity $v$.
Under the standard sign convention:
- $v > 0$: Receding target (range increasing with time).
- $v < 0$: Approaching target (range decreasing with time).
- $v = 0$: Stationary target.

Across the Coherent Processing Interval (CPI) of $N_c$ chirps at repetition interval $T_{PRI}$ ($m = 0, 1, \dots, N_c - 1$):
$$R(t_m) = R_0 + v \cdot m T_{PRI}$$
The round-trip propagation delay is:
$$\tau(t_m) = \frac{2 R(t_m)}{c} = \frac{2(R_0 + v \cdot m T_{PRI})}{c}$$

The received signal at the antenna is delayed by $\tau(t_m)$ and attenuated by target amplitude $A$:
$$s_{rx}(t, m) = A \exp\left(j 2\pi \left[ f_c (t - \tau(t_m)) + \frac{1}{2} S (t - \tau(t_m))^2 \right]\right)$$

### 3.3 Dechirping / Quadrature Mixing
The received echo is multiplied by the complex conjugate of the transmitted reference signal:
$$s_{IF}(t, m) = s_{tx}(t) \cdot s_{rx}^*(t, m)$$
Evaluating the instantaneous phase $\phi_{IF} = \phi_{tx} - \phi_{rx}$:
$$\phi_{IF}(t, m) = 2\pi \left[ f_c \tau(t_m) + S \tau(t_m) t - \frac{1}{2} S \tau(t_m)^2 \right]$$

Substituting $\tau(t_m) = \frac{2(R_0 + v m T_{PRI})}{c}$:
$$\phi_{IF}(t, m) = \underbrace{2\pi \left(\frac{2 S R_0}{c}\right) t}_{\text{Fast-time Range Beat}} + \underbrace{2\pi \left(\frac{2 v}{\lambda}\right) m T_{PRI}}_{\text{Slow-time Doppler Phase}} + \underbrace{\frac{4\pi R_0}{\lambda}}_{\text{Constant Phase}} - \underbrace{\pi S \tau^2}_{\text{Residual Video Phase}}$$

This reveals the two fundamental relationships of FMCW radar:
1. **Range Beat Frequency ($f_b$):**
   $$f_b = \frac{2 S R_0}{c} = \frac{2 B R_0}{c T_c} \implies R_0 = \frac{c \cdot f_b}{2 S}$$
2. **Doppler Frequency ($f_D$):**
   $$f_D = \frac{2 v}{\lambda} \implies v = \frac{\lambda \cdot f_D}{2}$$

---

## 4. Digital Signal Processing Pipeline

The 2D data matrix has dimensions $N_{samples} \times N_{chirps} = 250 \times 128$:
- **Fast-time dimension (rows):** Samples within an individual chirp sampled at $F_s = 10\text{ MS/s}$.
- **Slow-time dimension (columns):** Successive chirps spaced by $T_{PRI} = 30\,\mu\text{s}$.

```
                Fast-time (samples n = 0 ... N_s-1)
               +------------------------------------+
       Chirp 0 | s_IF(0, 0)   ...   s_IF(N_s-1, 0)  |  -- Range FFT --> Range Bins
Slow-  Chirp 1 | s_IF(0, 1)   ...   s_IF(N_s-1, 1)  |
time     ...   |     ...      ...        ...        |
     Chirp N-1 | s_IF(0, N-1) ... s_IF(N_s-1, N-1)  |
               +------------------------------------+
                                 |
                            Doppler FFT
                                 v
                            Doppler Bins
```

### 4.1 Stage 1: Fast-Time Windowing & Range FFT
To prevent rectangular window spectral leakage (sidelobes only $-13\text{ dB}$ down), a Hann window is applied:
$$w_{fast}(n) = 0.5 - 0.5 \cos\left(\frac{2\pi n}{N_s - 1}\right), \quad 0 \le n < N_s$$
Fast-time FFT converts each column into range frequency bins. The range axis is calibrated as:
$$R(k) = \frac{c \cdot k \cdot F_s}{2 S \cdot N_{fft, range}}, \quad k = 0, 1, \dots, \frac{N_{fft, range}}{2} - 1$$
The maximum unambiguous range for positive baseband frequencies is:
$$R_{max} = \frac{c F_s}{4 S} = \frac{3\times 10^8 \times 10\times 10^6}{4 \times 6\times 10^{12}} = 125.0\text{ m}$$

### 4.2 Stage 2: Slow-Time Windowing & Doppler FFT
Along the slow-time axis, each range bin contains phase progression proportional to target velocity. A slow-time Hann window is applied, followed by an FFT with `fftshift`:
$$f_D(m) \in \left[-\frac{PRF}{2}, +\frac{PRF}{2}\right), \quad PRF = \frac{1}{T_{PRI}} = 33.33\text{ kHz}$$
The calibrated velocity axis is:
$$v(m) = \frac{\lambda \cdot f_D(m)}{2} \in [-32.44\text{ m/s}, +32.44\text{ m/s}]$$
Velocity resolution is:
$$\Delta v = \frac{\lambda}{2 N_c T_{PRI}} = \frac{0.0038934}{2 \times 128 \times 30\times 10^{-6}} = 0.507\text{ m/s} \quad (1.82\text{ km/h})$$

### 4.3 Stage 3: 2D CA-CFAR Detection
To dynamically detect targets in clutter and noise, 2D CA-CFAR slides a window across the power map $P(r, d) = |S_{RD}(r, d)|^2$:
- Outer Training Window: $2(G_r + T_r) + 1 \times 2(G_d + T_d) + 1$
- Inner Guard Window: $2G_r + 1 \times 2G_d + 1$
- Training cells $N_{train} = (2(G_r + T_r) + 1)(2(G_d + T_d) + 1) - (2G_r + 1)(2G_d + 1)$

Noise floor estimation:
$$P_{noise}(r, d) = \frac{1}{N_{train}} \sum_{(i,j) \in \text{training}} P(i,j)$$
Threshold multiplier for exponential/Rayleigh noise:
$$\alpha = N_{train} \left( P_{fa}^{-1/N_{train}} - 1 \right)$$
Detection condition:
$$P(r, d) > \alpha \cdot P_{noise}(r, d)$$
Detections are clustered using 2D Non-Maximum Suppression (NMS) to eliminate multi-cell target blooming.

---

## 5. Experimental Results & Verification

### 5.1 Automated Unit Test Results
All 12 automated unit tests passed on Python 3.11:
1. `test_waveform_duration_and_bandwidth`: PASSED (slope $= 6.0\times 10^{12}\text{ Hz/s}$, duration $= 25\,\mu\text{s}$)
2. `test_stationary_target_range_estimation`: PASSED (True: $45.2\text{ m}$, Est: $45.24\text{ m}$, Error $< 0.05\text{ m}$)
3. `test_approaching_target_velocity_sign`: PASSED (True: $-20.0\text{ m/s}$, Est: $-20.03\text{ m/s}$, Error $< 0.04\text{ m/s}$)
4. `test_receding_target_velocity_sign`: PASSED (True: $+18.0\text{ m/s}$, Est: $+18.01\text{ m/s}$, Error $< 0.02\text{ m/s}$)
5. `test_multiple_separated_targets_detection`: PASSED ($P_d = 100\%$, 3 targets resolved)
6. `test_estimation_under_moderate_noise`: PASSED ($12\text{ dB}$ SNR, Range MAE $< 0.4\text{ m}$, Velocity MAE $< 0.2\text{ m/s}$)
7. `test_bandwidth_range_resolution_relationship`: PASSED ($\Delta R = c / (2B)$ verified for $100\text{ MHz}$ and $300\text{ MHz}$)
8. `test_invalid_radar_parameters_raise_errors`: PASSED (Parameter validation handles negative values and timing violations)
9. `test_fft_axes_and_units_calibration`: PASSED ($R_{max} = 125.0\text{ m}$, $v_{max} = \pm 32.44\text{ m/s}$)
10. `test_reproducible_random_seed`: PASSED (Bitwise identical outputs with identical seed)
11. `test_ca_cfar_detection_and_false_alarm_rejection`: PASSED (Target detected above noise floor)
12. `test_stationary_target_zero_velocity`: PASSED ($|v_{est}| < 0.1\text{ m/s}$)

### 5.2 Multi-Target Traffic Scenario Benchmark
Under Scenario 2 (Multi-Target Highway Traffic) with 3 vehicles at $15\text{ dB}$ SNR:

| Target Name | True Range (m) | Est. Range (m) | Range Error (m) | True Velocity (m/s) | Est. Velocity (m/s) | Velocity Error (m/s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Receding Lead Car** | 35.00 | 35.12 | 0.12 | +15.00 | +15.02 | 0.02 | Matched |
| **Oncoming Truck** | 70.00 | 69.84 | 0.16 | -24.00 | -23.98 | 0.02 | Matched |
| **Oncoming Bike** | 95.00 | 95.21 | 0.21 | -10.00 | -10.01 | 0.01 | Matched |

- **Detection Rate ($P_d$):** $100.0\%$
- **False Alarm Count:** 0
- **Range MAE:** $0.163\text{ m}$
- **Velocity MAE:** $0.017\text{ m/s}$

---

## 6. Engineering Constraints & Limitations
1. **Simulation Boundary:** This software models the physical signal transformations at complex baseband. It does not control actual RF electronics, high-frequency front-ends, or phased antenna arrays.
2. **Single-Channel (SISO):** The radar simulates a single Tx-Rx channel. Spatial Angle of Arrival (AoA) estimation requires multiple spatial antennas (MIMO beamforming / virtual array processing).
3. **Point-Target Simplification:** Vehicles are modeled as point scatterers with constant RCS (Swerling 0). Real vehicles exhibit complex multi-point scattering and angle-dependent RCS fluctuations.
4. **Resolution vs Zero-Padding:** Zero-padding in the FFT provides visual interpolation but cannot resolve two targets separated by less than $\Delta R = c / (2B)$.

---

## 7. Conclusion & Future Scope
This project successfully demonstrates the foundational principles of 77 GHz FMCW automotive radar signal processing. By developing the complete processing chain—from chirp synthesis and multi-target delay kinematics to 2D Range-Doppler transformation and CA-CFAR detection—we demonstrated how physical millimeter-wave radar achieves high-accuracy distance and velocity estimation.

### Future Work:
1. **MIMO Virtual Array Integration:** Expand to a 3 Tx $\times$ 4 Rx virtual array (12 virtual channels) to estimate target azimuth and elevation angles via Capon / MUSIC beamforming.
2. **Micro-Doppler Analysis:** Simulate rotating wheels and pedestrian limb movements to classify targets via micro-Doppler spectrograms.
3. **Clutter Modeling:** Implement Weibull and K-distribution ground clutter for realistic road surface returns.
4. **Kalman Filtering:** Implement an Extended Kalman Filter (EKF) to track target state vectors $[x, y, v_x, v_y]^T$ over multiple frames.

