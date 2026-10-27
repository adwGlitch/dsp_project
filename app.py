"""
app.py
Interactive Engineering Dashboard for FMCW Radar Signal Processing & Moving Object Detection.
Operating Frequency: 77 GHz Automotive FMCW Radar.

State-Management Architecture
-------------------------------
There is ONE authoritative copy of each piece of state:
  st.session_state.cfg        – active RadarConfig (written only by _apply_radar_config)
  st.session_state.targets    – active List[Target]  (written only by _apply_targets)
  st.session_state.det_algo   – str: detection algorithm name
  st.session_state.cpi_res    – RadarCPIResult from the last run
  st.session_state.proc_res   – ProcessingResult  from the last run
  st.session_state.det_res    – DetectionResult   from the last run
  st.session_state.metrics    – EvaluationMetrics from the last run
  st.session_state.sim_stamp  – str: human-readable description of what was last simulated

Widgets use stable keys prefixed with "hw_" (hardware sliders) or "tgt_" (target inputs).
Widget values are read only when the corresponding action button is pressed; they are never
written back into session_state after the widget has been rendered (which would cause
Streamlit's "overwrite after instantiation" warning).
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

from radar.config import RadarConfig
from radar.target_model import Target, get_scenario_presets
from radar.waveform import simulate_radar_cpi
from radar.signal_processing import RangeDopplerProcessor
from radar.detection import detect_peaks_2d, ca_cfar_2d
from radar.metrics import evaluate_detections

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="77 GHz FMCW Radar DSP Lab",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #58a6ff;
        font-family: 'SF Pro Display', -apple-system, sans-serif;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 6px 6px 0 0;
        padding: 8px 18px;
        color: #c9d1d9;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Default radar configuration
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_CFG = dict(
    carrier_frequency=77.0e9,
    bandwidth=150.0e6,
    chirp_duration=25.0e-6,
    chirp_repetition_interval=30.0e-6,
    sampling_rate=10.0e6,
    num_chirps=128,
    snr_db=15.0,
    seed=42,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: validate, build config, run pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _validate_hw_params(
    bw: float, tc: float, pri: float, fs: float, nc: int, snr: Optional[float]
) -> Tuple[bool, str]:
    """
    Return (ok, error_message).
    Checks the most common physically/numerically invalid combinations
    before attempting to construct a RadarConfig.
    """
    if bw <= 0:
        return False, "Bandwidth must be positive."
    if tc <= 0:
        return False, "Chirp duration must be positive."
    if pri < tc:
        return False, (
            f"Chirp Repetition Interval ({pri*1e6:.1f} µs) must be ≥ "
            f"Active Chirp Duration ({tc*1e6:.1f} µs)."
        )
    if fs <= 0:
        return False, "Sampling frequency must be positive."
    ns = int(round(fs * tc))
    if ns < 16:
        return False, (
            f"Too few fast-time samples ({ns}). "
            "Increase sampling rate or chirp duration."
        )
    if nc < 2:
        return False, "Number of chirps must be ≥ 2."
    if snr is not None and snr < 0:
        return False, "SNR must be ≥ 0 dB (or leave noise disabled)."
    return True, ""


def _build_config(
    bw: float, tc: float, pri: float, fs: float,
    nc: int, snr: Optional[float], seed: int,
) -> Tuple[Optional[RadarConfig], str]:
    """
    Attempt to build and validate a RadarConfig.
    Returns (cfg, "") on success or (None, error_string) on failure.
    """
    ok, err = _validate_hw_params(bw, tc, pri, fs, nc, snr)
    if not ok:
        return None, err
    try:
        cfg = RadarConfig(
            carrier_frequency=77.0e9,
            bandwidth=bw,
            chirp_duration=tc,
            chirp_repetition_interval=pri,
            sampling_rate=fs,
            num_chirps=nc,
            snr_db=snr,
            seed=seed,
        )
        return cfg, ""
    except ValueError as exc:
        return None, str(exc)


def _run_pipeline(cfg: RadarConfig, targets: List[Target], det_algo: str) -> str:
    """
    Execute the complete simulation + processing + detection + metrics pipeline.
    Stores all results in session_state.
    Returns a human-readable stamp of what was simulated (for success messages).
    """
    cpi_res = simulate_radar_cpi(cfg, targets, seed=cfg.seed)
    processor = RangeDopplerProcessor(cfg, n_fft_range=512, n_fft_doppler=256)
    proc_res = processor.process(cpi_res.if_signal)

    if det_algo == "2D CA-CFAR":
        det_res = ca_cfar_2d(
            proc_res.range_doppler_map,
            proc_res.range_axis,
            proc_res.velocity_axis,
            train_cells_range=5,
            train_cells_doppler=3,
            guard_cells_range=2,
            guard_cells_doppler=1,
            pfa=1e-4,
        )
    else:
        det_res = detect_peaks_2d(
            proc_res.range_doppler_map_db,
            proc_res.range_axis,
            proc_res.velocity_axis,
            threshold_db=-22.0,
        )

    metrics = evaluate_detections(targets, det_res.detected_targets, cfg)

    # ── Commit to session_state atomically ──────────────────────────────────
    st.session_state.cpi_res  = cpi_res
    st.session_state.proc_res = proc_res
    st.session_state.det_res  = det_res
    st.session_state.metrics  = metrics

    snr_str = f"{cfg.snr_db:.1f} dB" if cfg.snr_db is not None else "noiseless"
    stamp = (
        f"{len(targets)} target(s) | "
        f"B={cfg.bandwidth/1e6:.0f} MHz | "
        f"Tc={cfg.chirp_duration*1e6:.0f} µs | "
        f"SNR={snr_str} | "
        f"seed={cfg.seed} | "
        f"algo={det_algo}"
    )
    st.session_state.sim_stamp = stamp
    return stamp


def _apply_radar_config(
    bw: float, tc: float, pri: float, fs: float,
    nc: int, snr: Optional[float], seed: int, det_algo: str,
) -> Tuple[bool, str]:
    """
    Validate → build config → run pipeline.
    On success: updates st.session_state.cfg and st.session_state.det_algo.
    Returns (success, message).
    """
    cfg, err = _build_config(bw, tc, pri, fs, nc, snr, seed)
    if cfg is None:
        return False, f"⚠️ Invalid parameters: {err}"

    st.session_state.cfg      = cfg
    st.session_state.det_algo = det_algo

    stamp = _run_pipeline(cfg, st.session_state.targets, det_algo)
    return True, f"✅ Simulation complete — {stamp}"


def _apply_targets(targets: List[Target]) -> Tuple[bool, str]:
    """
    Update authoritative target list and re-run with current radar config.
    Returns (success, message).
    """
    if not targets:
        return False, "⚠️ Target list is empty."
    st.session_state.targets = targets
    cfg       = st.session_state.cfg
    det_algo  = st.session_state.det_algo
    stamp = _run_pipeline(cfg, targets, det_algo)
    return True, f"✅ Targets updated — {stamp}"


# ─────────────────────────────────────────────────────────────────────────────
# One-time session_state initialisation
# ─────────────────────────────────────────────────────────────────────────────
if "cfg" not in st.session_state:
    st.session_state.cfg = RadarConfig(**_DEFAULT_CFG)

if "targets" not in st.session_state:
    presets = get_scenario_presets()
    st.session_state.targets = list(presets[1].targets)   # Scenario 2 default

if "det_algo" not in st.session_state:
    st.session_state.det_algo = "Peak Threshold NMS"

if "sim_stamp" not in st.session_state:
    st.session_state.sim_stamp = ""

# Run initial pipeline if no results exist yet
if "proc_res" not in st.session_state:
    _run_pipeline(
        st.session_state.cfg,
        st.session_state.targets,
        st.session_state.det_algo,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("📡 FMCW Radar DSP")
st.sidebar.caption("77 GHz Automotive Radar Simulation")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Overview & System Specs",
        "2. Scenario Configuration",
        "3. Signal Analysis (Time & 1D FFT)",
        "4. Range-Doppler Map (2D FFT)",
        "5. Performance Evaluation & Errors",
        "6. DSP Mathematics & Theory",
    ],
    index=0,
)

_cfg = st.session_state.cfg
st.sidebar.divider()
st.sidebar.markdown(f"**Carrier Frequency:** 77.0 GHz")
st.sidebar.markdown(f"**Wavelength (λ):** {_cfg.wavelength*1e3:.2f} mm")
st.sidebar.markdown(f"**Bandwidth:** {_cfg.bandwidth/1e6:.1f} MHz")
st.sidebar.markdown(f"**Range Res (ΔR):** {_cfg.range_resolution:.2f} m")
st.sidebar.markdown(f"**Velocity Res (Δv):** {_cfg.velocity_resolution:.2f} m/s")
st.sidebar.markdown(f"**Max Range:** {_cfg.max_unambiguous_range:.1f} m")
st.sidebar.markdown(f"**Max Velocity:** ±{_cfg.max_unambiguous_velocity:.1f} m/s")
if st.session_state.sim_stamp:
    st.sidebar.divider()
    st.sidebar.caption(f"Last run: {st.session_state.sim_stamp}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if nav_choice == "1. Overview & System Specs":
    st.title("Moving Object Detection Using FMCW Radar Signal Processing")
    st.markdown("### Digital Signal Processing (DSP) Laboratory & Automotive Simulation")
    st.markdown("""
    This project demonstrates the core DSP algorithms used in
    **77 GHz automotive FMCW radar** for ADAS (Adaptive Cruise Control,
    Autonomous Emergency Braking, Blind Spot Detection).
    """)

    cfg = st.session_state.cfg
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RF Carrier Frequency</div>
            <div class="metric-value">77.0 GHz</div>
            <div style="color:#8b949e;font-size:0.8rem">λ = {cfg.wavelength*1e3:.2f} mm (Millimeter Wave)</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Range Resolution (ΔR)</div>
            <div class="metric-value">{cfg.range_resolution:.2f} m</div>
            <div style="color:#8b949e;font-size:0.8rem">Sweep Bandwidth = {cfg.bandwidth/1e6:.1f} MHz</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Velocity Resolution (Δv)</div>
            <div class="metric-value">{cfg.velocity_resolution:.2f} m/s</div>
            <div style="color:#8b949e;font-size:0.8rem">CPI = {cfg.num_chirps} chirps ({cfg.cpi_duration*1e3:.1f} ms)</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        det_res = st.session_state.det_res
        targets = st.session_state.targets
        metrics = st.session_state.metrics
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Target Detection Status</div>
            <div class="metric-value">{len(det_res.detected_targets)} / {len(targets)}</div>
            <div style="color:#8b949e;font-size:0.8rem">Detection Rate: {metrics.detection_rate*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("DSP Architecture Pipeline")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.code("""\
+---------------------------------------+
| 1. FMCW Waveform Generator            |
|    s_tx(t) = exp(j*pi*S*t^2)          |
+---------------------------------------+
               |
               v
+---------------------------------------+
| 2. Multi-Target Echo Channel Model    |
|    s_rx(t,m) with delay & Doppler     |
+---------------------------------------+
               |
               v
+---------------------------------------+
| 3. Quadrature Mixing / Dechirping     |
|    s_IF(t,m) = s_tx * conj(s_rx)     |
+---------------------------------------+
               |
               v
+---------------------------------------+
| 4. Fast-Time Hann + Range FFT         |
|    Beat freq f_b → Range R            |
+---------------------------------------+
               |
               v
+---------------------------------------+
| 5. Slow-Time Hann + Doppler FFT       |
|    Doppler f_D → Velocity v           |
+---------------------------------------+
               |
               v
+---------------------------------------+
| 6. 2D Range-Doppler Map + CA-CFAR     |
|    Detection, NMS, param extraction   |
+---------------------------------------+
        """, language="text")
    with col_b:
        st.markdown("#### System Parameter Specification")
        st.dataframe(
            pd.DataFrame(list(cfg.summary_dict().items()),
                         columns=["Parameter", "Value"]),
            use_container_width=True,
        )
        st.info(
            "**Note:** A 77 GHz RF carrier cannot be sampled on a PC "
            "(Nyquist > 154 GHz). The simulation models the exact physical "
            "IF/baseband equations, preserving millimetre-wave phase and "
            "Doppler dynamics."
        )

    st.divider()
    st.subheader("🎯 Active Simulated Scenario Targets")
    st.caption("Ground truth targets currently active in the 77 GHz radar DSP pipeline:")
    active_targets = st.session_state.targets
    det_res = st.session_state.det_res
    targets_table = []
    for t in active_targets:
        matched = any(
            abs(d.range_m - t.range_m) <= max(2.0, 2.0 * cfg.range_resolution) and
            abs(d.velocity_mps - t.velocity_mps) <= max(1.0, 2.0 * cfg.velocity_resolution)
            for d in det_res.detected_targets
        )
        targets_table.append({
            "Target Name": t.name,
            "True Range (m)": f"{t.range_m:.2f}",
            "True Velocity (m/s)": f"{t.velocity_mps:+.2f}",
            "Speed (km/h)": f"{t.velocity_mps * 3.6:+.1f}",
            "Motion Direction": "Receding ↗" if t.velocity_mps > 0 else ("Approaching ↙" if t.velocity_mps < 0 else "Stationary •"),
            "Amplitude (RCS)": f"{t.rcs_or_amplitude:.2f}",
            "DSP Detection Status": "Detected ✅" if matched else "Unresolved ⚠️",
        })
    st.dataframe(pd.DataFrame(targets_table), use_container_width=True)
    st.caption(f"Active simulation stamp: `{st.session_state.sim_stamp}`")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — SCENARIO CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
elif nav_choice == "2. Scenario Configuration":
    st.title("Scenario & Radar Parameter Configuration")
    st.markdown(
        "Configure simulated automotive traffic scenarios, noise levels, "
        "and radar hardware settings."
    )
    st.info(
        f"📡 **Active Radar Simulation:** `{st.session_state.sim_stamp}` — "
        f"**{len(st.session_state.targets)} target(s)** currently in simulation pipeline."
    )

    tabs = st.tabs([
        "Predefined Example Scenarios",
        "Custom Target Configuration",
        "Radar Hardware Settings",
    ])

    # ── Tab 0: Preset scenarios ───────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Automotive Scenario Presets")
        presets     = get_scenario_presets()
        preset_names = [p.name for p in presets]

        selected_p_name = st.selectbox(
            "Select Traffic Preset", preset_names,
            index=1, key="preset_selector",
        )
        selected_preset = next(p for p in presets if p.name == selected_p_name)

        st.write(f"**Description:** {selected_preset.description}")
        st.write(f"**Preset SNR:** {selected_preset.snr_db} dB")

        target_df = pd.DataFrame([
            {
                "Target Name": t.name,
                "Range (m)": t.range_m,
                "Velocity (m/s)": t.velocity_mps,
                "Speed (km/h)": round(t.velocity_mps * 3.6, 1),
                "Motion": (
                    "Receding ↗" if t.velocity_mps > 0
                    else ("Approaching ↙" if t.velocity_mps < 0 else "Stationary •")
                ),
                "Amplitude": t.rcs_or_amplitude,
            }
            for t in selected_preset.targets
        ])
        st.dataframe(target_df, use_container_width=True)

        st.caption(
            "ℹ️ Loading a preset updates the **target list** only. "
            "The preset's SNR value is applied to the radar configuration."
        )

        if st.button("Load This Preset & Simulate", type="primary", key="btn_load_preset"):
            # Build a new cfg from current hw settings but with the preset SNR
            curr = st.session_state.cfg
            new_cfg, err = _build_config(
                bw=curr.bandwidth,
                tc=curr.chirp_duration,
                pri=curr.chirp_repetition_interval,
                fs=curr.sampling_rate,
                nc=curr.num_chirps,
                snr=selected_preset.snr_db,
                seed=curr.seed,
            )
            if new_cfg is None:
                st.error(f"Configuration error: {err}")
            else:
                new_targets = list(selected_preset.targets)
                st.session_state.cfg     = new_cfg
                st.session_state.targets = new_targets

                # ── Purge custom-target widget keys so that
                # the Custom tab re-derives its row count and default values
                # from the freshly loaded preset.
                st.session_state["tgt_count"] = len(new_targets)
                for k in list(st.session_state.keys()):
                    if k.startswith(("tgt_name_", "tgt_range_", "tgt_vel_", "tgt_amp_")):
                        del st.session_state[k]

                stamp = _run_pipeline(
                    new_cfg,
                    new_targets,
                    st.session_state.det_algo,
                )
                st.success(f"✅ Loaded **{selected_p_name}** and executed DSP pipeline — {stamp}")
                det_res = st.session_state.det_res
                metrics = st.session_state.metrics
                st.info(
                    f"🎯 **Simulation Ready:** {len(det_res.detected_targets)} of {len(new_targets)} targets detected "
                    f"({metrics.detection_rate*100:.0f}%). "
                    f"Check **Page 3 (Signal Analysis)**, **Page 4 (Range-Doppler Map)**, or **Page 1 (Overview)** to view the updated scenario."
                )

    # ── Tab 1: Custom targets ─────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Configure Target List")
        st.caption(
            "Velocity convention: **>0 = Receding** (distance increasing), "
            "**<0 = Approaching** (distance decreasing)."
        )

        curr_targets = st.session_state.targets

        # Bug A fix: synchronise tgt_count only when the key is absent (i.e.
        # after a preset load purged it). Using setdefault means we only seed
        # the key once; on subsequent renders the widget's own value is kept,
        # so the user can freely change the number of targets.
        st.session_state.setdefault("tgt_count", len(curr_targets))

        num_t = st.number_input(
            "Number of Targets", min_value=1, max_value=8,
            value=len(curr_targets), key="tgt_count",
        )

        edited: List[Target] = []
        for i in range(num_t):
            default = (
                curr_targets[i] if i < len(curr_targets)
                else Target(
                    range_m=30.0 + i * 20.0,
                    velocity_mps=-10.0 + i * 10.0,
                    name=f"Target {i+1}",
                )
            )
            st.markdown(f"**Target #{i+1}**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                # Bug C fix (corrected): seed the key with the authoritative
                # default only if it is ABSENT (setdefault). This ensures that
                # after a preset load (which purged these keys) the new preset
                # values appear, but on subsequent renders user edits are
                # preserved and not overwritten every frame.
                st.session_state.setdefault(f"tgt_name_{i}",  default.name)
                st.session_state.setdefault(f"tgt_range_{i}", float(default.range_m))
                st.session_state.setdefault(f"tgt_vel_{i}",   float(default.velocity_mps))
                st.session_state.setdefault(f"tgt_amp_{i}",   float(default.rcs_or_amplitude))
                name = st.text_input(
                    f"Name #{i+1}", value=default.name, key=f"tgt_name_{i}"
                )
            with c2:
                rng = st.number_input(
                    f"Range [m] #{i+1}", min_value=1.0, max_value=200.0,
                    value=float(default.range_m), step=1.0, key=f"tgt_range_{i}",
                )
            with c3:
                vel = st.number_input(
                    f"Velocity [m/s] #{i+1}", min_value=-50.0, max_value=50.0,
                    value=float(default.velocity_mps), step=1.0, key=f"tgt_vel_{i}",
                )
            with c4:
                amp = st.slider(
                    f"Amplitude #{i+1}", min_value=0.1, max_value=2.0,
                    value=float(default.rcs_or_amplitude), step=0.1,
                    key=f"tgt_amp_{i}",
                )
            edited.append(Target(range_m=rng, velocity_mps=vel,
                                 rcs_or_amplitude=amp, name=name))

        if st.button("Apply Targets & Simulate", type="primary", key="btn_apply_targets"):
            ok, msg = _apply_targets(edited)
            if ok:
                st.success(msg)
                det_res = st.session_state.det_res
                metrics = st.session_state.metrics
                st.info(
                    f"🎯 **DSP Simulation Complete:** Successfully processed {len(edited)} target(s). "
                    f"**{len(det_res.detected_targets)} of {len(edited)}** detected "
                    f"(Detection Rate: {metrics.detection_rate*100:.0f}%).\n\n"
                    f"👉 **Next steps:** Open **Page 4 (Range-Doppler Map)** to view the 2D FFT heatmap, "
                    f"**Page 3 (Signal Analysis)** for 1D spectra, or **Page 1 (Overview)** to view the target specs."
                )
                if det_res.detected_targets:
                    st.write("**Detected Targets Summary:**")
                    det_df = pd.DataFrame([
                        {
                            "Detection ID": f"T{d.id}",
                            "Estimated Range (m)": f"{d.range_m:.2f}",
                            "Estimated Velocity (m/s)": f"{d.velocity_mps:+.2f}",
                            "Power (dB)": f"{d.power_db:.1f}",
                            "Est SNR (dB)": f"{d.snr_estimate_db:.1f}",
                        }
                        for d in det_res.detected_targets
                    ])
                    st.dataframe(det_df, use_container_width=True)
            else:
                st.error(msg)

    # ── Tab 2: Radar hardware settings ────────────────────────────────────────
    with tabs[2]:
        st.subheader("Radar Physical & Processing Configuration")

        # Read current config once; widget default values come from here
        cur = st.session_state.cfg

        c1, c2 = st.columns(2)
        with c1:
            hw_bw = st.slider(
                "Chirp Bandwidth (MHz)",
                min_value=50, max_value=400,
                value=int(cur.bandwidth / 1e6), step=25,
                key="hw_bw",
            ) * 1e6

            hw_tc = st.slider(
                "Active Chirp Duration (µs)",
                min_value=10, max_value=60,
                value=int(cur.chirp_duration * 1e6), step=5,
                key="hw_tc",
            ) * 1e-6

            # Keep PRI ≥ Tc dynamically.
            # Bug B fix: if the cached hw_pri value is below the new minimum
            # (because the user raised Tc), clamp it before the slider renders,
            # otherwise Streamlit raises a ValueError.
            pri_min = int(hw_tc * 1e6)
            cached_pri = int(st.session_state.get("hw_pri", int(cur.chirp_repetition_interval * 1e6)))
            if cached_pri < pri_min:
                st.session_state["hw_pri"] = pri_min
            pri_cur = max(int(cur.chirp_repetition_interval * 1e6), pri_min)
            hw_pri = st.slider(
                "Chirp Repetition Interval (µs)",
                min_value=pri_min, max_value=80,
                value=pri_cur, step=5,
                key="hw_pri",
            ) * 1e-6

            hw_fs = st.slider(
                "ADC Sampling Frequency (MS/s)",
                min_value=5, max_value=25,
                value=int(cur.sampling_rate / 1e6), step=1,
                key="hw_fs",
            ) * 1e6

        with c2:
            hw_nc = st.select_slider(
                "Chirps per CPI Frame",
                options=[32, 64, 128, 256],
                value=cur.num_chirps,
                key="hw_nc",
            )

            hw_noise_en = st.checkbox(
                "Enable AWGN Noise",
                value=(cur.snr_db is not None),
                key="hw_noise_en",
            )
            if hw_noise_en:
                hw_snr: Optional[float] = st.slider(
                    "SNR (dB)", min_value=0.0, max_value=40.0,
                    value=float(cur.snr_db if cur.snr_db is not None else 15.0),
                    step=1.0, key="hw_snr",
                )
            else:
                hw_snr = None

            hw_seed = st.number_input(
                "Random Seed", min_value=1, max_value=999999,
                value=int(cur.seed if cur.seed is not None else 42),
                key="hw_seed",
            )

            hw_algo = st.radio(
                "Detection Algorithm",
                ["Peak Threshold NMS", "2D CA-CFAR"],
                index=0 if st.session_state.det_algo == "Peak Threshold NMS" else 1,
                key="hw_algo",
            )

        # Preview the derived resolutions BEFORE applying
        ok_preview, err_preview = _validate_hw_params(
            hw_bw, hw_tc, hw_pri, hw_fs, hw_nc, hw_snr
        )
        if ok_preview:
            preview_cfg, _ = _build_config(hw_bw, hw_tc, hw_pri, hw_fs, hw_nc, hw_snr, int(hw_seed))
            if preview_cfg:
                st.info(
                    f"**Preview (not yet applied):** "
                    f"ΔR = {preview_cfg.range_resolution:.2f} m | "
                    f"Δv = {preview_cfg.velocity_resolution:.2f} m/s | "
                    f"R_max = {preview_cfg.max_unambiguous_range:.1f} m | "
                    f"v_max = ±{preview_cfg.max_unambiguous_velocity:.1f} m/s | "
                    f"N_samples = {preview_cfg.num_samples_per_chirp}"
                )
        else:
            st.warning(f"⚠️ Current slider values are invalid: {err_preview}")

        st.divider()
        if st.button(
            "💾 Save Radar Settings & Re-simulate",
            type="primary",
            key="btn_save_hw",
        ):
            ok, msg = _apply_radar_config(
                bw=hw_bw, tc=hw_tc, pri=hw_pri, fs=hw_fs,
                nc=hw_nc, snr=hw_snr, seed=int(hw_seed),
                det_algo=hw_algo,
            )
            if ok:
                st.success(msg)
                st.info(
                    f"⚙️ **Updated Radar Parameters:** Bandwidth = {st.session_state.cfg.bandwidth/1e6:.1f} MHz | "
                    f"Range Resolution ΔR = {st.session_state.cfg.range_resolution:.2f} m | "
                    f"Velocity Resolution Δv = {st.session_state.cfg.velocity_resolution:.2f} m/s | "
                    f"Max Range = {st.session_state.cfg.max_unambiguous_range:.1f} m."
                )
            else:
                st.error(msg)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — SIGNAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif nav_choice == "3. Signal Analysis (Time & 1D FFT)":
    st.title("Signal Analysis: Time Domain & 1D Spectrum")
    st.markdown(
        "Plots generated from the **active simulation** — change settings on "
        "Page 2 and re-simulate to update."
    )

    cpi  = st.session_state.cpi_res
    cfg  = st.session_state.cfg
    proc = st.session_state.proc_res

    st.caption(f"Active simulation: {st.session_state.sim_stamp}")

    # ── Plot 1: TX chirp ─────────────────────────────────────────────────────
    st.subheader("1. FMCW Transmitted Waveform & Instantaneous Frequency")
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 4), facecolor="#0e1117")
    for ax in (ax1a, ax1b):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values():
            sp.set_color("#30363d")

    fast_t_us = cpi.fast_time * 1e6
    ax1a.plot(fast_t_us, cfg.chirp_slope * cpi.fast_time / 1e6, color="#58a6ff", lw=2)
    ax1a.set_title("Instantaneous Frequency Sweep", color="#c9d1d9", fontsize=11)
    ax1a.set_xlabel("Fast Time (µs)", color="#8b949e")
    ax1a.set_ylabel("Frequency Offset (MHz)", color="#8b949e")
    ax1a.grid(True, color="#21262d", linestyle="--")

    n_show = min(80, len(cpi.tx_chirp_sample))
    ax1b.plot(fast_t_us[:n_show], np.real(cpi.tx_chirp_sample[:n_show]),
              color="#3fb950", lw=1.5, label="I (in-phase)")
    ax1b.plot(fast_t_us[:n_show], np.imag(cpi.tx_chirp_sample[:n_show]),
              color="#f0883e", lw=1.2, linestyle="--", label="Q (quadrature)")
    ax1b.set_title(f"Tx Analytic Chirp (first {n_show} samples)", color="#c9d1d9", fontsize=11)
    ax1b.set_xlabel("Fast Time (µs)", color="#8b949e")
    ax1b.set_ylabel("Normalised Amplitude", color="#8b949e")
    ax1b.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax1b.grid(True, color="#21262d", linestyle="--")
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    st.markdown(
        f"> **Note:** slope $S = B / T_c = {cfg.chirp_slope/1e12:.2f}\\times 10^{{12}}$ Hz/s, "
        f"B = {cfg.bandwidth/1e6:.0f} MHz, Tc = {cfg.chirp_duration*1e6:.0f} µs."
    )

    # ── Plot 2: IF beat signal ───────────────────────────────────────────────
    st.subheader("2. IF Beat Signal (Time Domain)")
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 4), facecolor="#0e1117")
    for ax in (ax2a, ax2b):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values():
            sp.set_color("#30363d")

    if_1d    = cpi.if_signal[:, 0]
    clean_1d = cpi.if_signal_clean[:, 0]
    ax2a.plot(fast_t_us, np.real(if_1d), color="#58a6ff", lw=1.2, label="Noisy IF (I)")
    if cfg.snr_db is not None:
        ax2a.plot(fast_t_us, np.real(clean_1d), color="#3fb950",
                  lw=1.0, alpha=0.7, label="Clean IF (I)")
    ax2a.set_title("Beat Signal s_IF(t) — Chirp #1", color="#c9d1d9", fontsize=11)
    ax2a.set_xlabel("Fast Time (µs)", color="#8b949e")
    ax2a.set_ylabel("Voltage Amplitude", color="#8b949e")
    ax2a.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax2a.grid(True, color="#21262d", linestyle="--")

    slow_t_ms = cpi.slow_time * 1e3
    sample_idx = min(10, cpi.if_signal.shape[0] - 1)
    ax2b.plot(slow_t_ms, np.real(cpi.if_signal[sample_idx, :]),
              color="#bc8cff", lw=1.5, marker=".", markersize=4)
    ax2b.set_title(f"Slow-Time Phase — Sample #{sample_idx}", color="#c9d1d9", fontsize=11)
    ax2b.set_xlabel("Slow Time (ms)", color="#8b949e")
    ax2b.set_ylabel("Amplitude", color="#8b949e")
    ax2b.grid(True, color="#21262d", linestyle="--")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── Plot 3: 1D Range Profile + Doppler slice ─────────────────────────────
    st.subheader("3. 1D Range Profile & 1D Doppler Slice")
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 4.5), facecolor="#0e1117")
    for ax in (ax3a, ax3b):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values():
            sp.set_color("#30363d")

    ax3a.plot(proc.range_axis, proc.range_profile_1d, color="#58a6ff", lw=1.8)
    ax3a.set_title("1D Range Profile (Hann windowed)", color="#c9d1d9", fontsize=11)
    ax3a.set_xlabel("Distance (m)", color="#8b949e")
    ax3a.set_ylabel("Normalised Magnitude (dB)", color="#8b949e")
    ax3a.set_ylim(-65, 5)
    ax3a.grid(True, color="#21262d", linestyle="--")
    for t in st.session_state.targets:
        ax3a.axvline(t.range_m, color="#f85149", linestyle=":", lw=1.2, alpha=0.8)
        ax3a.text(t.range_m + 1.0, -10, f"{t.name}\n({t.range_m}m)",
                  color="#f85149", fontsize=8)

    ax3b.plot(proc.velocity_axis, proc.doppler_spectrum_1d, color="#3fb950", lw=1.8)
    ax3b.set_title("1D Doppler Spectrum (peak range bin)", color="#c9d1d9", fontsize=11)
    ax3b.set_xlabel("Radial Velocity (m/s)", color="#8b949e")
    ax3b.set_ylabel("Normalised Magnitude (dB)", color="#8b949e")
    ax3b.set_ylim(-65, 5)
    ax3b.grid(True, color="#21262d", linestyle="--")
    for t in st.session_state.targets:
        ax3b.axvline(t.velocity_mps, color="#f0883e", linestyle=":", lw=1.2, alpha=0.8)
        ax3b.text(t.velocity_mps + 0.5, -12, f"{t.velocity_mps} m/s",
                  color="#f0883e", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — RANGE-DOPPLER MAP
# ─────────────────────────────────────────────────────────────────────────────
elif nav_choice == "4. Range-Doppler Map (2D FFT)":
    st.title("2D Range-Doppler Map & Target Detection")
    st.markdown("Two-dimensional spectral map from the **active simulation**.")
    st.caption(f"Active simulation: {st.session_state.sim_stamp}")

    proc    = st.session_state.proc_res
    det_res = st.session_state.det_res
    cfg     = st.session_state.cfg

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Range-Doppler Power Heatmap (dB)")
        fig, ax = plt.subplots(figsize=(9, 6.5), facecolor="#0e1117")
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values():
            sp.set_color("#30363d")

        extent = [
            proc.velocity_axis[0], proc.velocity_axis[-1],
            proc.range_axis[0],    proc.range_axis[-1],
        ]
        im = ax.imshow(
            proc.range_doppler_map_db,
            aspect="auto", origin="lower", extent=extent,
            cmap="inferno", vmin=-50.0, vmax=0.0,
        )
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Power Relative to Peak (dB)", color="#8b949e")
        cbar.ax.yaxis.set_tick_params(color="#8b949e")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#8b949e")

        for det in det_res.detected_targets:
            ax.scatter(det.velocity_mps, det.range_m,
                       s=120, facecolors="none", edgecolors="#00e5ff",
                       linewidths=2.0, marker="o")
            ax.text(det.velocity_mps + 0.8, det.range_m + 1.2,
                    f"T{det.id}: {det.range_m:.1f}m, {det.velocity_mps:.1f}m/s",
                    color="#00e5ff", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#161b22",
                              edgecolor="#00e5ff", alpha=0.8))

        ax.set_title(f"2D Range-Doppler Map ({det_res.algorithm})",
                     color="#c9d1d9", fontsize=12)
        ax.set_xlabel("Radial Velocity (m/s)  [<0 Approaching | >0 Receding]",
                      color="#8b949e")
        ax.set_ylabel("Range Distance (m)", color="#8b949e")
        ax.set_xlim(proc.velocity_axis[0], proc.velocity_axis[-1])
        ax.set_ylim(0, min(150, proc.range_axis[-1]))
        ax.axvline(0, color="#8b949e", linestyle="--", alpha=0.4)
        ax.grid(True, color="#30363d", linestyle=":", alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.subheader("Target Detection Table")
        st.caption(f"Detected {len(det_res.detected_targets)} targets "
                   f"using {det_res.algorithm}.")
        if det_res.detected_targets:
            rows = []
            for det in det_res.detected_targets:
                motion = (
                    "Receding ↗"   if det.velocity_mps > 0.5
                    else "Approaching ↙" if det.velocity_mps < -0.5
                    else "Stationary •"
                )
                rows.append({
                    "ID": f"T{det.id}",
                    "Range (m)": det.range_m,
                    "Vel (m/s)": det.velocity_mps,
                    "Speed (km/h)": round(det.velocity_mps * 3.6, 1),
                    "Motion": motion,
                    "Power (dB)": det.power_db,
                    "Est SNR (dB)": det.snr_estimate_db,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.warning("No targets exceeded the detection threshold.")

        st.divider()
        st.markdown("#### Radar Operating Limits")
        st.write(f"- **ΔR:** `{cfg.range_resolution:.2f} m`")
        st.write(f"- **Δv:** `{cfg.velocity_resolution:.2f} m/s`")
        st.write(f"- **R_max:** `{cfg.max_unambiguous_range:.1f} m`")
        st.write(f"- **v_max:** `±{cfg.max_unambiguous_velocity:.1f} m/s` "
                 f"({cfg.max_unambiguous_velocity*3.6:.0f} km/h)")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — PERFORMANCE EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
elif nav_choice == "5. Performance Evaluation & Errors":
    st.title("Performance Evaluation & Ground-Truth Verification")
    st.markdown("Quantitative comparison between simulated ground truth and DSP estimates.")
    st.caption(f"Active simulation: {st.session_state.sim_stamp}")

    metrics = st.session_state.metrics
    cfg     = st.session_state.cfg

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Detection Rate (P_d)",
                  f"{metrics.detection_rate*100:.1f}%")
    with k2:
        st.metric("False Alarm Count", f"{metrics.false_alarm_count}")
    with k3:
        st.metric("Range MAE", f"{metrics.range_mae_m:.3f} m")
    with k4:
        st.metric("Velocity MAE", f"{metrics.velocity_mae_mps:.3f} m/s")

    st.divider()
    st.subheader("Target Association & Ground Truth Matching")
    st.caption("Hungarian assignment in normalised resolution-cell units (ΔR, Δv).")
    st.dataframe(metrics.to_dataframe(), use_container_width=True)

    if metrics.matched_pairs:
        st.subheader("Estimation Error Analysis")
        fig_err, (axe1, axe2) = plt.subplots(1, 2, figsize=(14, 4), facecolor="#0e1117")
        for ax in (axe1, axe2):
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e")
            for sp in ax.spines.values():
                sp.set_color("#30363d")

        r_trues = [p.target.range_m      for p in metrics.matched_pairs]
        r_errs  = [p.range_error_m       for p in metrics.matched_pairs]
        v_trues = [p.target.velocity_mps for p in metrics.matched_pairs]
        v_errs  = [p.velocity_error_mps  for p in metrics.matched_pairs]

        axe1.scatter(r_trues, r_errs, s=80, color="#58a6ff", zorder=3)
        axe1.axhline(cfg.range_resolution, color="#f85149", linestyle="--",
                     label=f"ΔR = {cfg.range_resolution:.2f} m")
        axe1.set_title("Range Error vs Target Distance", color="#c9d1d9")
        axe1.set_xlabel("True Range (m)", color="#8b949e")
        axe1.set_ylabel("Absolute Error (m)", color="#8b949e")
        axe1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
        axe1.grid(True, color="#21262d", linestyle="--")

        axe2.scatter(v_trues, v_errs, s=80, color="#3fb950", zorder=3)
        axe2.axhline(cfg.velocity_resolution, color="#f0883e", linestyle="--",
                     label=f"Δv = {cfg.velocity_resolution:.2f} m/s")
        axe2.set_title("Velocity Error vs Target Velocity", color="#c9d1d9")
        axe2.set_xlabel("True Velocity (m/s)", color="#8b949e")
        axe2.set_ylabel("Absolute Error (m/s)", color="#8b949e")
        axe2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
        axe2.grid(True, color="#21262d", linestyle="--")

        plt.tight_layout()
        st.pyplot(fig_err)
        plt.close(fig_err)

    # ── SNR sweep ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Noise Sensitivity Sweep (SNR vs Performance)")
    st.markdown(
        "Runs the simulation at 7 SNR levels using the **current radar config "
        "and target list** to show how noise impacts estimation accuracy."
    )
    if st.button("Execute SNR Sweep (0–30 dB)", key="btn_snr_sweep"):
        with st.spinner("Running 7 SNR levels…"):
            snr_levels  = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
            sweep_r_mae = []
            sweep_v_mae = []
            sweep_pd    = []

            for snr in snr_levels:
                sweep_cfg, err = _build_config(
                    bw=cfg.bandwidth,
                    tc=cfg.chirp_duration,
                    pri=cfg.chirp_repetition_interval,
                    fs=cfg.sampling_rate,
                    nc=cfg.num_chirps,
                    snr=snr,
                    seed=42,
                )
                if sweep_cfg is None:
                    st.warning(f"Skipping SNR={snr}: {err}")
                    sweep_r_mae.append(np.nan)
                    sweep_v_mae.append(np.nan)
                    sweep_pd.append(np.nan)
                    continue

                cpi_s = simulate_radar_cpi(sweep_cfg, st.session_state.targets, seed=42)
                p_s   = RangeDopplerProcessor(sweep_cfg, n_fft_range=512, n_fft_doppler=256)
                res_s = p_s.process(cpi_s.if_signal)
                det_s = detect_peaks_2d(
                    res_s.range_doppler_map_db,
                    res_s.range_axis, res_s.velocity_axis,
                    threshold_db=-20.0,
                )
                m_s = evaluate_detections(
                    st.session_state.targets, det_s.detected_targets, sweep_cfg
                )
                sweep_r_mae.append(m_s.range_mae_m)
                sweep_v_mae.append(m_s.velocity_mae_mps)
                sweep_pd.append(m_s.detection_rate * 100.0)

        fig_sw, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(14, 4), facecolor="#0e1117")
        for ax in (ax_s1, ax_s2):
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e")
            for sp in ax.spines.values():
                sp.set_color("#30363d")

        ax_s1.plot(snr_levels, sweep_r_mae, marker="o", color="#58a6ff",
                   lw=2, label="Range MAE (m)")
        ax_s1.plot(snr_levels, sweep_v_mae, marker="s", color="#3fb950",
                   lw=2, label="Velocity MAE (m/s)")
        ax_s1.set_title("Estimation Error vs SNR", color="#c9d1d9")
        ax_s1.set_xlabel("SNR (dB)", color="#8b949e")
        ax_s1.set_ylabel("Mean Absolute Error", color="#8b949e")
        ax_s1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
        ax_s1.grid(True, color="#21262d", linestyle="--")

        ax_s2.plot(snr_levels, sweep_pd, marker="^", color="#e3b341", lw=2)
        ax_s2.set_title("Detection Rate vs SNR", color="#c9d1d9")
        ax_s2.set_xlabel("SNR (dB)", color="#8b949e")
        ax_s2.set_ylabel("Detection Rate (%)", color="#8b949e")
        ax_s2.set_ylim(-5, 105)
        ax_s2.grid(True, color="#21262d", linestyle="--")

        plt.tight_layout()
        st.pyplot(fig_sw)
        plt.close(fig_sw)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — DSP THEORY
# ─────────────────────────────────────────────────────────────────────────────
elif nav_choice == "6. DSP Mathematics & Theory":
    st.title("Digital Signal Processing (DSP) Mathematical Formulations")
    st.markdown("Theoretical foundations of FMCW radar for academic review.")

    cfg = st.session_state.cfg   # show live values for active config

    st.markdown(f"""
    ### 1. FMCW Transmitted Waveform
    $$f_{{tx}}(t) = f_c + S \\cdot t, \\quad S = \\frac{{B}}{{T_c}}$$
    Active values: $B = {cfg.bandwidth/1e6:.0f}\\text{{ MHz}},\\;
    T_c = {cfg.chirp_duration*1e6:.0f}\\,\\mu\\text{{s}},\\;
    S = {cfg.chirp_slope/1e12:.2f}\\times 10^{{12}}\\text{{ Hz/s}}$

    ---

    ### 2. Target Echo & Round-Trip Delay
    At chirp $m$:
    $$R(t_m) = R_0 + v\\,m T_{{PRI}}, \\qquad \\tau(t_m) = \\frac{{2R(t_m)}}{{c}}$$

    ---

    ### 3. Beat Frequency & Range Equation
    $$f_b = S\\,\\tau = \\frac{{2SR}}{{c}} \\implies R = \\frac{{c\\,f_b}}{{2S}}$$

    ---

    ### 4. Doppler Phase & Velocity Equation
    $$f_D = \\frac{{2v}}{{\\lambda}} \\implies v = \\frac{{\\lambda\\,f_D}}{{2}}$$
    Sign convention: $v < 0$ approaching, $v > 0$ receding.

    Active: $\\lambda = {cfg.wavelength*1e3:.3f}\\text{{ mm}}$,
    $v_{{max}} = \\pm{cfg.max_unambiguous_velocity:.2f}\\text{{ m/s}}$

    ---

    ### 5. Resolution Limits (Physical — cannot be improved by zero-padding)
    
    $$\Delta R = \frac{c}{2B} = {cfg.range_resolution:.3f}\text{ m}$$
    
    $$\Delta v = \frac{\lambda}{2 N_c T_{{PRI}}} = {cfg.velocity_resolution:.3f}\text{ m/s}$$

    ---

    ### 6. 2D CA-CFAR Detection Threshold
    
    $$T_{{CUT}} = \alpha \cdot P_{{noise}}$$
    
    $$\alpha = N_{{train}}\!\left(P_{{fa}}^{{-1/N_{{train}}}} - 1\right)$$

    ---

    ### 7. Engineering Limitations
    1. **SISO only** — Single-Input Single-Output: no azimuth or elevation angle-of-arrival (AoA) estimation without a MIMO antenna array.
    2. **Stop-and-Go approximation** — Target displacement within active chirp $T_c = {cfg.chirp_duration*1e6:.0f}\,\mu\text{s}$ is $< 1\text{ mm}$ at automotive speeds; Doppler velocity is accurately captured in slow-time chirp-to-chirp phase shift.
    3. **Point target model** — Real vehicles exhibit distributed, angle-dependent Radar Cross Section (RCS) fluctuations (e.g., Swerling 1/3 models).
    4. **No clutter / multipath** — Ground clutter, road surface reflections, guard-rails, and atmospheric attenuation are not modelled in this simulation.
    """)
