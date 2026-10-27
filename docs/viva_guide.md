# Viva Voce & Oral Examination Preparation Guide

## FMCW Radar Signal Processing & Moving Object Detection (77 GHz)

This guide prepares students for academic reviews, external evaluations, and technical viva voce questions.

---

### Q1: Why is 77 GHz chosen for automotive radar instead of lower frequencies (like 24 GHz)?
**Answer:**
1. **Wavelength ($\lambda = 3.89\text{ mm}$):** The millimeter wavelength allows highly compact antenna arrays that easily fit behind vehicle bumpers.
2. **Regulatory Bandwidth:** The 76–81 GHz band offers up to $4\text{ GHz}$ of sweep bandwidth (compared to only $\approx 200\text{ MHz}$ at 24 GHz). Since range resolution $\Delta R = \frac{c}{2B}$, wider bandwidth allows sub-meter and even centimeter-level range resolution.
3. **Velocity Sensitivity:** Doppler shift $f_D = \frac{2v}{\lambda}$ is inversely proportional to $\lambda$. At 77 GHz, small velocity changes produce larger, easily detectable phase shifts across chirps.

---

### Q2: What is FMCW radar, and why is it preferred over pulsed radar for cars?
**Answer:**
- **FMCW (Frequency-Modulated Continuous-Wave)** continuously transmits a chirp signal whose frequency changes linearly over time.
- **Why not pulsed radar?** Pulsed radar requires massive peak transmit power (kilowatts) to achieve adequate SNR over short pulse durations, plus gigahertz-speed ADCs to sample nanosecond pulses. FMCW transmits continuously with milliwatt power levels ($10-20\text{ dBm}$) and mixes the received echo down to a baseband beat frequency of only a few megahertz, allowing standard low-cost $10-25\text{ MS/s}$ ADCs.

---

### Q3: What is "dechirping" and what is the "beat frequency"?
**Answer:**
- **Dechirping** is the analog mixing of the received echo $s_{rx}(t)$ with the reference transmit chirp $s_{tx}(t)$ in a quadrature mixer:
  $$s_{IF}(t) = s_{tx}(t) \cdot s_{rx}^*(t)$$
- Because the transmit chirp frequency is ramping linearly at slope $S = B / T_c$, a delayed echo arriving after round-trip time $\tau = 2R/c$ differs in instantaneous frequency by:
  $$f_b = S \cdot \tau = \frac{2 S R}{c}$$
- $f_b$ is the **beat frequency**. Since $f_b$ is directly proportional to range $R$, measuring the frequency of the beat signal immediately gives target distance.

---

### Q4: Explain the difference between "Fast-Time" and "Slow-Time".
**Answer:**
- **Fast-Time:** The time scale *within* an individual chirp ($t \in [0, T_c)$, e.g., $25\,\mu\text{s}$), sampled at the ADC rate ($10\text{ MS/s}$). Sampling along fast-time captures the beat frequency $f_b$ to measure target **Range**.
- **Slow-Time:** The time scale *across successive chirps* ($t_m = m \cdot T_{PRI}$, e.g., across 128 chirps over $3.84\text{ ms}$). Sampling along slow-time captures the inter-chirp Doppler phase shift to measure target **Radial Velocity**.

---

### Q5: Why do we need a 2D FFT instead of a single 1D FFT?
**Answer:**
- The first FFT (fast-time) acts on each chirp column to extract range bins.
- However, if multiple targets exist at the *same distance* with *different speeds* (or same speed at different distances), a 1D FFT cannot distinguish them.
- Performing the second FFT (slow-time) across all chirps for each range bin isolates the Doppler frequency.
- The resulting 2D matrix—the **Range-Doppler Map (RDM)**—simultaneously separates targets in both distance and velocity.

---

### Q6: Why is Hann windowing applied before each FFT?
**Answer:**
- A finite duration sample sequence is equivalent to multiplying an infinite signal by a rectangular window.
- The Fourier transform of a rectangular window is a $\text{sinc}$ function with high sidelobes (only $-13\text{ dB}$ below the mainlobe). A strong target's sidelobes can easily mask or create false detections near weaker adjacent targets.
- Applying a **Hann window** smooths the signal boundaries, reducing spectral leakage and suppressing sidelobes to $-31\text{ dB}$, ensuring weak targets are not masked.

---

### Q7: Does zero-padding an FFT improve physical resolution?
**Answer:**
- **NO.** Zero-padding merely provides frequency interpolation on a denser grid, giving smoother peak shapes and aiding sub-bin peak localization.
- Fundamental physical resolution is strictly governed by the signal physics:
  - **Range Resolution:** $\Delta R = \frac{c}{2B}$ (strictly limited by sweep bandwidth $B$).
  - **Velocity Resolution:** $\Delta v = \frac{\lambda}{2 T_{CPI}}$ (strictly limited by total coherent integration time $T_{CPI} = N_c T_{PRI}$).
- Two targets separated by less than $\Delta R$ cannot be resolved as distinct peaks, no matter how much zero-padding is applied.

---

### Q8: What causes Doppler aliasing, and what is $v_{max}$?
**Answer:**
- Doppler frequency across chirps is sampled at the Chirp Repetition Frequency $PRF = \frac{1}{T_{PRI}}$.
- By Nyquist theorem, unambiguous Doppler frequencies must satisfy:
  $$|f_D| \le \frac{PRF}{2} = \frac{1}{2 T_{PRI}}$$
- Since $f_D = \frac{2v}{\lambda}$, the maximum unambiguous velocity is:
  $$v_{max} = \frac{\lambda}{4 T_{PRI}}$$
- If target speed exceeds $v_{max}$, its phase wraps modulo $2\pi$, causing Doppler aliasing (e.g., an approaching vehicle at $-25\text{ m/s}$ could appear as a receding target at $+7.4\text{ m/s}$ if $v_{max} = 16.2\text{ m/s}$). In this project, $T_{PRI} = 30\,\mu\text{s}$ guarantees $v_{max} = \pm 32.44\text{ m/s}$ ($\pm 116.8\text{ km/h}$).

---

### Q9: How does 2D CA-CFAR work, and why do we use guard cells?
**Answer:**
- **CA-CFAR (Cell-Averaging Constant False Alarm Rate):** Dynamically computes a detection threshold by estimating local noise in surrounding training cells:
  $$T_{CUT} = \alpha \cdot P_{noise}, \quad \alpha = N_{train} (P_{fa}^{-1/N_{train}} - 1)$$
- **Guard Cells:** Target reflections in the Cell Under Test (CUT) spill energy into adjacent cells due to FFT mainlobe width and windowing. If adjacent cells were included in noise estimation, the target's own energy would inflate the noise estimate, potentially causing the target to miss detection (target self-masking). Guard cells surround the CUT to isolate it from the training cells.
- **Limitation:** In dense multi-target environments, an interfering target inside the training window inflates the noise estimate, masking weaker nearby targets.

---

### Q10: How does the software simulate a 77 GHz signal without an impossible sampling rate?
**Answer:**
- Directly sampling a 77 GHz RF carrier requires an ADC sampling faster than the Nyquist rate of $154\text{ GHz}$, which is impossible on ordinary computers.
- Instead, the simulation models the system at **complex baseband / intermediate frequency (IF)** after the receiver mixer.
- The physical carrier frequency $f_c = 77\text{ GHz}$ enters the simulation analytically through the phase term $2\pi f_c \tau = \frac{4\pi R}{\lambda}$, perfectly preserving the millimeter-wave Doppler shifts and propagation physics while allowing the baseband signal to be sampled at a practical $10\text{ MS/s}$.

---

### Q11: What are the main limitations of this simulation compared to real automotive hardware?
**Answer:**
1. **Single Antenna (SISO):** Only range and velocity are measured; estimating azimuth angle requires a MIMO virtual antenna array.
2. **Point Targets:** Vehicles are modeled as ideal point reflectors (Swerling 0); real vehicles have distributed scattering centers and angle-dependent RCS.
3. **No Clutter / Multipath:** Road surface clutter, guardrail reflections, and multipath ghost targets are not modeled.
4. **Ideal RF Front-End:** Mixer non-linearities, phase noise, amplifier saturation, and mutual interference from other automotive radars are assumed ideal.

