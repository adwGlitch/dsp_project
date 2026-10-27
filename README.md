# Moving Object Detection Using FMCW Radar Signal Processing (77 GHz)

A comprehensive, software-only Digital Signal Processing (DSP) simulation and interactive analysis laboratory of an automotive Frequency-Modulated Continuous-Wave (FMCW) radar operating at 77 GHz.

---

## 1. Project Overview

Modern automotive Advanced Driver Assistance Systems (ADAS) and Autonomous Vehicles rely on 77 GHz FMCW millimeter-wave radar for Adaptive Cruise Control (ACC), Autonomous Emergency Braking (AEB), and Blind-Spot Detection (BSD). Unlike optical sensors, FMCW radar provides reliable velocity and distance tracking through rain, fog, dust, and darkness.

This project simulates the complete physical signal generation and digital signal processing pipeline:
1. **Linear FMCW Chirp Waveform Generation** ($77\text{ GHz}$ nominal carrier, $150\text{ MHz}$ sweep bandwidth).
2. **Kinematic Multi-Target Echo Simulation** (propagation delay $\tau$, Doppler phase progression across chirps, complex Gaussian noise).
3. **Quadrature Dechirping / Mixing** to extract the Intermediate Frequency (IF) beat signal.
4. **Range FFT** across fast-time samples with Hann windowing for distance estimation.
5. **Doppler FFT** across slow-time chirps (`fftshift`) for radial velocity estimation.
6. **2D Range-Doppler Map (RDM)** computation in logarithmic dB scale.
7. **2D CA-CFAR (Cell-Averaging Constant False Alarm Rate)** and 2D Peak Non-Maximum Suppression (NMS) target detection.
8. **Automated Target Association & Evaluation** (Hungarian matching, Range/Velocity MAE and RMSE, Detection Rate $P_d$).
9. **Interactive Streamlit Engineering Dashboard** with 6 dedicated analysis pages.

> **Academic Disclaimer:** This is an academic DSP simulation running locally on a personal computer. It models the physical radar equations at complex baseband / intermediate frequency and does not transmit electromagnetic radiation or interface with physical radar hardware.

---

## 2. System Architecture & Directory Layout

```
dsp/
├── app.py                      # 6-page interactive Streamlit dashboard
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Pytest configuration
├── README.md                   # Setup guide and technical documentation
├── radar/                      # Core DSP and radar simulation package
│   ├── __init__.py             # Package API exports
│   ├── config.py               # Radar parameters, physical constants, validation
│   ├── target_model.py         # Point target kinematics and scenario presets
│   ├── waveform.py             # Chirp generation, delay, Doppler, dechirp, AWGN
│   ├── signal_processing.py    # Range FFT, Doppler FFT, 2D Range-Doppler Map
│   ├── detection.py            # 2D Peak NMS and 2D CA-CFAR detectors
│   └── metrics.py              # Target association, MAE, RMSE, P_d evaluation
├── tests/                      # Automated validation suite (12 pytest tests)
│   ├── test_waveform.py        # Tests 1, 8, 10: duration, slope, errors, seeds
│   ├── test_range_estimation.py# Tests 2, 7, 9: stationary distance, axes, resolution
│   ├── test_velocity_estimation.py # Tests 3, 4: approaching/receding signs, zero Doppler
│   └── test_detection.py       # Tests 5, 6: multi-target, noise robustness, CA-CFAR
└── docs/                       # Comprehensive documentation
    ├── project_report.md       # Full academic project report (ECE/DSP)
    └── viva_guide.md           # Oral exam / viva voce Q&A preparation guide
```

---

## 3. Radar Operating Specifications

| Parameter | Symbol | Value | Unit | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Carrier Frequency** | $f_c$ | $77.0$ | $\text{GHz}$ | Millimeter-wave automotive band |
| **Wavelength** | $\lambda = c / f_c$ | $3.893$ | $\text{mm}$ | Physical RF wavelength |
| **Sweep Bandwidth** | $B$ | $150.0$ | $\text{MHz}$ | Linear frequency chirp excursion |
| **Chirp Duration** | $T_c$ | $25.0$ | $\mu\text{s}$ | Active ramp duration |
| **Chirp Repetition Interval** | $T_{PRI}$ | $30.0$ | $\mu\text{s}$ | Pulse Repetition Interval ($T_{PRI} \ge T_c$) |
| **Chirp Slope** | $S = B / T_c$ | $6.0 \times 10^{12}$ | $\text{Hz/s}$ | Frequency modulation rate ($6\text{ MHz/}\mu\text{s}$) |
| **ADC Sampling Frequency** | $F_s$ | $10.0$ | $\text{MS/s}$ | Baseband ADC rate |
| **Samples per Chirp** | $N_s$ | $250$ | samples | Fast-time samples per chirp |
| **Chirps per Frame** | $N_c$ | $128$ | chirps | Slow-time chirps per CPI frame |
| **Range Resolution** | $\Delta R = c / (2B)$ | $1.00$ | $\text{m}$ | Physical distance resolving limit |
| **Max Unambiguous Range** | $R_{max} = \frac{c F_s}{4 S}$ | $125.0$ | $\text{m}$ | Maximum range at $f_b = F_s / 2$ |
| **Velocity Resolution** | $\Delta v = \frac{\lambda}{2 N_c T_{PRI}}$ | $0.507$ | $\text{m/s}$ | Velocity resolving limit ($1.82\text{ km/h}$) |
| **Max Unambiguous Velocity** | $v_{max} = \frac{\lambda}{4 T_{PRI}}$ | $\pm 32.44$ | $\text{m/s}$ | Unambiguous speed limit ($\pm 116.8\text{ km/h}$) |
| **CPI Duration** | $T_{CPI} = N_c T_{PRI}$ | $3.84$ | $\text{ms}$ | Coherent processing interval |

---

## 4. Installation & Quickstart

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your system.

### Step 1: Clone or Navigate to the Workspace
```powershell
cd c:\Users\HP\Desktop\dsp
```

### Step 2: Create and Activate a Virtual Environment
**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run Automated Verification Tests
Run the complete automated test suite verifying all 10 project requirements:
```powershell
pytest -v
```
Expected output:
```
tests/test_detection.py::test_multiple_separated_targets_detection PASSED
tests/test_detection.py::test_estimation_under_moderate_noise PASSED
tests/test_detection.py::test_ca_cfar_detection_and_false_alarm_rejection PASSED
tests/test_range_estimation.py::test_stationary_target_range_estimation PASSED
tests/test_range_estimation.py::test_fft_axes_and_units_calibration PASSED
tests/test_range_estimation.py::test_bandwidth_range_resolution_relationship PASSED
tests/test_velocity_estimation.py::test_approaching_target_velocity_sign PASSED
tests/test_velocity_estimation.py::test_receding_target_velocity_sign PASSED
tests/test_velocity_estimation.py::test_stationary_target_zero_velocity PASSED
tests/test_waveform.py::test_waveform_duration_and_bandwidth PASSED
tests/test_waveform.py::test_invalid_radar_parameters_raise_errors PASSED
tests/test_waveform.py::test_reproducible_random_seed PASSED
============================= 12 passed in 2.77s ==============================
```

### Step 5: Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 5. Dashboard Features & Walkthrough

The web application contains 6 dedicated pages accessible from the sidebar:

1. **Overview & System Specs:** Real-time KPI cards, full DSP architecture diagram, operating parameter specifications.
2. **Scenario Configuration:**
   - 4 Predefined scenarios (Single approaching car, multi-target highway traffic, stationary barrier in noise, close-range resolution test).
   - Interactive target editor (Range $1-200$ m, Velocity $\pm 50$ m/s, Reflection RCS).
   - Real-time radar parameter sliders with physical constraint validation.
   - Detector selection: 2D Peak NMS or 2D CA-CFAR.
3. **Signal Analysis (Time & 1D FFT):**
   - Transmitted FMCW chirp frequency sweep ($f_{tx}(t)$) and time-domain waveform.
   - Mixed IF beat signal in time domain.
   - Slow-time phase accumulation across chirps.
   - 1D Range Profile with ground-truth distance annotations.
   - 1D Doppler Spectrum for the peak range bin.
4. **Range-Doppler Map (2D FFT):**
   - 2D Heatmap with calibrated Range (m) and Radial Velocity (m/s) axes.
   - Overlay markers on detected target peaks with target ID and coordinates.
   - Target parameter table with motion classification (Approaching $\leftarrow$, Receding $\rightarrow$, Stationary $\bullet$).
5. **Performance Evaluation & Errors:**
   - Quantitative evaluation comparing ground-truth against estimates.
   - Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).
   - Probability of Detection ($P_d$) and False Alarm count.
   - Interactive Monte-Carlo SNR sweep ($0\text{ dB}$ to $30\text{ dB}$) plotting error curves.
6. **DSP Mathematics & Theory:**
   - Student-friendly mathematical derivations with KaTeX equations.
   - In-depth explanations of beat frequency, Doppler shift, windowing, and CFAR.
   - Explicit discussion of simulation limitations.

---

## 6. Mathematical Foundations

### 1. FMCW Instantaneous Frequency & Slope
$$f_{tx}(t) = f_c + S \cdot t, \quad S = \frac{B}{T_c}$$

### 2. Beat Frequency Dechirping
For round-trip delay $\tau = \frac{2R}{c}$, the mixer computes $s_{IF}(t) = s_{tx}(t) \cdot s_{rx}^*(t)$:
$$f_b = S \cdot \tau = \frac{2 S R}{c} \implies R = \frac{c \cdot f_b}{2 S}$$

### 3. Doppler Frequency & Radial Velocity
Across chirps $m$ at interval $T_{PRI}$, target movement $\Delta R = v \cdot T_{PRI}$ produces phase shift $\Delta \phi = \frac{4\pi v T_{PRI}}{\lambda}$:
$$f_D = \frac{2 v}{\lambda} \implies v = \frac{\lambda \cdot f_D}{2}$$
- **Velocity Sign Convention:**
  - $v < 0$: Approaching target (distance decreasing).
  - $v > 0$: Receding target (distance increasing).
  - $v = 0$: Stationary target.

### 4. 2D CA-CFAR Detection
The Cell-Averaging Constant False Alarm Rate detector evaluates a Cell Under Test (CUT) against surrounding training cells while excluding guard cells:
$$T_{CUT} = \alpha \cdot P_{noise}, \quad \alpha = N_{train} \left( P_{fa}^{-1/N_{train}} - 1 \right)$$

---

## 7. Testing & Quality Assurance

All 10 project validation requirements are covered by automated unit tests:
- **Test 1:** Chirp duration, bandwidth, and slope verification.
- **Test 2:** Stationary target distance estimation ($< 0.3$ m error).
- **Test 3:** Approaching target velocity sign verification ($v < 0$).
- **Test 4:** Receding target velocity sign verification ($v > 0$).
- **Test 5:** Simultaneous multi-target detection and association ($100\%$ $P_d$).
- **Test 6:** Estimation accuracy maintained under moderate noise ($12\text{ dB}$ SNR).
- **Test 7:** Measured range resolution inverse scaling with bandwidth.
- **Test 8:** Parameter validation error handling ($B \le 0$, $T_{PRI} < T_c$, etc.).
- **Test 9:** FFT axis and unit calibration ($R_{max}, v_{max}$).
- **Test 10:** Deterministic simulation reproducibility with fixed random seed.

