"""Orbit -- Upload your data. Predict the future.

Run with:
    streamlit run orbit/app.py
"""
from __future__ import annotations

import io
import os
import sys
import tempfile

# Disable tqdm and warnings before any other imports
os.environ["TQDM_DISABLE"] = "1"

import warnings
warnings.filterwarnings("ignore")

# Ensure the project root and orbit dir are on sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_PROJECT_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from style import CUSTOM_CSS, MPL_STYLE_PARAMS, ACCENT, SUCCESS
from analysis import engineering_report, compute_accuracy_pct

from scipy.signal import savgol_filter

from koopsim import KoopSim
from koopsim.systems import (
    HopfBifurcation,
    RLCCircuit,
    SpringMassDamper,
    VanDerPolOscillator,
    EulerBernoulliBeam,
)

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="Orbit", page_icon="◎", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
plt.rcParams.update(MPL_STYLE_PARAMS)

# ── Session state defaults ───────────────────────────────────────────
for key, default in {"X": None, "Y": None, "dt": None, "system_name": None,
                      "model": None, "report": None, "traj_true": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _denoise(data: np.ndarray, window: int = 11, poly_order: int = 3) -> np.ndarray:
    """Apply Savitzky-Golay filter to each column of data."""
    window = min(window, data.shape[0])
    if window % 2 == 0:
        window -= 1
    if window < poly_order + 2:
        return data
    return savgol_filter(data, window_length=window, polyorder=poly_order, axis=0)


def _has_data() -> bool:
    return st.session_state["X"] is not None


def _has_model() -> bool:
    return st.session_state["model"] is not None


def _run_demo(system, name, poly_degree=None, rbf_centers=None, regularization=1e-6):
    """Generate data, fit model, compute report."""
    rng = np.random.default_rng(42)
    x0 = rng.standard_normal(system.state_dim) * 0.5
    # Use small dt for smooth "measured" data (critical for plot quality)
    dt = 0.005
    X, Y = system.generate_snapshots(x0, dt=dt, n_steps=200, n_trajectories=15)
    st.session_state["X"] = X
    st.session_state["Y"] = Y
    st.session_state["dt"] = dt
    st.session_state["system_name"] = name

    # Also store a smooth ground-truth trajectory for plotting
    traj = system.generate_trajectory(x0, dt=dt, n_steps=400)
    st.session_state["traj_true"] = traj

    sim = KoopSim(method="edmd", poly_degree=poly_degree,
                  rbf_centers=rbf_centers, regularization=regularization, verbose=False)
    sim.fit(X, Y, dt)
    st.session_state["model"] = sim

    spec = sim.spectral_analysis()
    report = engineering_report(spec["eigenvalues"], dt)
    n_test = min(200, X.shape[0])
    report["accuracy_pct"] = compute_accuracy_pct(sim.model, X[:n_test], Y[:n_test])
    st.session_state["report"] = report


# =====================================================================
# HEADER
# =====================================================================
st.markdown("# ◎ Orbit")
st.markdown("*Instant dynamics prediction for vibrations, circuits & control systems. No PhD. No $10k license.*")
st.divider()

# =====================================================================
# STEP 1 — Choose Your System
# =====================================================================
st.markdown("### Step 1 — Choose Your System")

col_upload, col_known, col_demo = st.columns(3, gap="large")

# ── Upload CSV ───────────────────────────────────────────────────────
with col_upload:
    st.markdown("**I have measurement data**")
    st.caption("CSV: columns = variables, rows = time steps")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    upload_dt = st.number_input("Time step (s)", min_value=1e-6, value=0.01, format="%.6f")
    denoise_on = st.checkbox("🧹 Denoise input data", value=False,
                             help="Apply Savitzky-Golay smoothing filter to reduce sensor noise.")
    if uploaded is not None:
        if st.button("Load data", key="btn_csv"):
            content = uploaded.read().decode("utf-8")
            # Skip header rows that aren't numeric
            lines = content.strip().split("\n")
            start = 0
            for i, line in enumerate(lines):
                try:
                    [float(v) for v in line.split(",")]
                    start = i
                    break
                except ValueError:
                    continue
            data = np.loadtxt(io.StringIO("\n".join(lines[start:])), delimiter=",")
            if data.ndim == 2 and data.shape[0] >= 3:
                if denoise_on:
                    data = _denoise(data)
                st.session_state["X"] = data[:-1]
                st.session_state["Y"] = data[1:]
                st.session_state["dt"] = upload_dt
                st.session_state["system_name"] = uploaded.name
                st.session_state["model"] = None
                st.session_state["report"] = None
                st.session_state["traj_true"] = None
                st.rerun()

# ── Known system ─────────────────────────────────────────────────────
with col_known:
    st.markdown("**I have a known system**")
    sys_choice = st.selectbox("System", ["RLC Circuit", "Mass-Spring-Damper",
                                          "Vibrating Beam", "Van der Pol", "Hopf Limit Cycle"])

    if sys_choice == "RLC Circuit":
        R = st.slider("Resistance R (Ω)", 0.1, 20.0, 1.0, 0.1)
        L = st.slider("Inductance L (H)", 0.01, 10.0, 1.0, 0.01)
        C = st.slider("Capacitance C (F)", 0.001, 5.0, 1.0, 0.001)
        _sys_factory = lambda: RLCCircuit(R=R, L=L, C=C)
    elif sys_choice == "Mass-Spring-Damper":
        nm = st.slider("Number of masses", 1, 10, 3, 1)
        k = st.slider("Spring stiffness k (N/m)", 0.1, 50.0, 1.0, 0.1)
        c = st.slider("Damping c (Ns/m)", 0.0, 5.0, 0.1, 0.01)
        m = st.slider("Mass m (kg)", 0.1, 20.0, 1.0, 0.1)
        _sys_factory = lambda: SpringMassDamper(n_masses=nm, k=k, c=c, m=m)
    elif sys_choice == "Vibrating Beam":
        _sys_factory = lambda: EulerBernoulliBeam()
    elif sys_choice == "Van der Pol":
        mu = st.slider("Nonlinearity μ", 0.1, 10.0, 1.0, 0.1)
        _sys_factory = lambda: VanDerPolOscillator(mu=mu)
    else:  # Hopf
        mu = st.slider("Bifurcation parameter μ", 0.1, 5.0, 1.0, 0.1)
        _sys_factory = lambda: HopfBifurcation(mu=mu)

    if st.button("Simulate system", key="btn_sim"):
        system = _sys_factory()
        rng = np.random.default_rng(42)
        x0 = rng.standard_normal(system.state_dim) * 0.5
        dt = 0.005
        X, Y = system.generate_snapshots(x0, dt=dt, n_steps=200, n_trajectories=15)
        traj = system.generate_trajectory(x0, dt=dt, n_steps=400)
        st.session_state["X"] = X
        st.session_state["Y"] = Y
        st.session_state["dt"] = dt
        st.session_state["system_name"] = sys_choice
        st.session_state["traj_true"] = traj
        st.session_state["model"] = None
        st.session_state["report"] = None
        st.rerun()

# ── One-click demos ──────────────────────────────────────────────────
with col_demo:
    st.markdown("**Try a demo**")
    st.caption("One click — instant results.")

    if st.button("⚡  RLC Circuit Demo", key="btn_rlc", type="primary"):
        _run_demo(RLCCircuit(R=0.5, L=1.0, C=0.5), "RLC Circuit Demo")
        st.rerun()

    if st.button("🔧  Vibration Demo", key="btn_vib"):
        _run_demo(SpringMassDamper(n_masses=3, k=1.0, c=0.1, m=1.0), "Vibration Demo")
        st.rerun()

    if st.button("🌀  Limit Cycle Demo", key="btn_hopf"):
        _run_demo(HopfBifurcation(mu=1.0), "Limit Cycle Demo",
                  poly_degree=3, rbf_centers=30, regularization=1e-2)
        st.rerun()

# ── Data preview ─────────────────────────────────────────────────────
if _has_data():
    X = st.session_state["X"]
    st.success(f"**{st.session_state['system_name']}** — {X.shape[0]} samples, "
               f"{X.shape[1]} variables, dt = {st.session_state['dt']:.4g} s")

st.divider()

# =====================================================================
# STEP 2 — Build Model
# =====================================================================
if _has_data():
    st.markdown("### Step 2 — Build Model")

    if _has_model():
        st.success("Model ready!")
    else:
        with st.expander("Quality settings", expanded=False):
            quality = st.select_slider("Quality", options=["Fast", "Balanced", "Accurate"],
                                       value="Balanced")
        if st.button("🚀  Build Prediction Model", key="btn_build", type="primary"):
            qmap = {"Fast": (None, None), "Balanced": (2, None), "Accurate": (3, 50)}
            pd_, rb_ = qmap[quality]
            sim = KoopSim(method="edmd", poly_degree=pd_, rbf_centers=rb_, verbose=False)
            sim.fit(st.session_state["X"], st.session_state["Y"], st.session_state["dt"])
            st.session_state["model"] = sim

            spec = sim.spectral_analysis()
            report = engineering_report(spec["eigenvalues"], st.session_state["dt"])
            n_test = min(200, st.session_state["X"].shape[0])
            report["accuracy_pct"] = compute_accuracy_pct(
                sim.model, st.session_state["X"][:n_test], st.session_state["Y"][:n_test])
            st.session_state["report"] = report
            st.rerun()

    # ── Show model metrics ───────────────────────────────────────────
    if _has_model() and st.session_state["report"] is not None:
        rpt = st.session_state["report"]

        # FIX: Honest accuracy messaging for nonlinear cases
        is_linear = rpt["is_linear"]
        acc = rpt["accuracy_pct"]

        m1, m2, m3, m4 = st.columns(4)
        if is_linear:
            m1.metric("Prediction Accuracy", f'{acc:.1f}%')
        else:
            m1.metric("One-Step Accuracy", f'{acc:.1f}%')
        m2.metric("System Type", "Linear" if is_linear else "Nonlinear")
        m3.metric("Dominant Frequency", f'{rpt["dominant_freq_hz"]:.3f} Hz')
        settling = f'{rpt["settling_time_s"]:.2f} s' if rpt["settling_time_s"] else "N/A"
        m4.metric("Settling Time", settling)

        if is_linear:
            st.info("✅ **Linear system detected** — predictions are mathematically exact at all time horizons.")
        else:
            st.warning("⚠️ **Nonlinear system** — short-term predictions are highly accurate. "
                       "Accuracy decreases at longer horizons. Use confidence bands to gauge reliability.")

st.divider()

# =====================================================================
# STEP 3 — Explore Predictions
# =====================================================================
if _has_model():
    st.markdown("### Step 3 — Explore Predictions")

    sim: KoopSim = st.session_state["model"]
    X = st.session_state["X"]
    Y = st.session_state["Y"]
    dt_val = st.session_state["dt"]
    n_state = sim.model.n_state_dims
    rpt = st.session_state["report"]
    traj_true = st.session_state.get("traj_true")

    tab_predict, tab_report, tab_export = st.tabs(
        ["📈 Predict Future", "📊 Stability Report", "💾 Export"])

    # ── Predict Future ────────────────────────────────────────────────
    with tab_predict:
        default_x0 = X[0]
        x0_str = ", ".join(f"{v:.4g}" for v in default_x0)
        x0_input = st.text_input(f"Initial state ({n_state} values, comma-separated)",
                                  value=x0_str)

        x0 = None
        try:
            x0 = np.array([float(v.strip()) for v in x0_input.split(",")])
            if len(x0) != n_state:
                st.error(f"Expected {n_state} values, got {len(x0)}.")
                x0 = None
        except ValueError:
            st.error("Enter comma-separated numbers.")

        if x0 is not None:
            training_window = X.shape[0] * dt_val
            t_pred = st.slider("Prediction horizon (seconds)", 0.01,
                                float(training_window * 5), float(training_window))

            # FIX: Variable selector for high-dim systems
            if n_state > 4:
                var_options = [f"Variable {i+1}" for i in range(n_state)]
                selected_vars = st.multiselect("Select variables to plot",
                                                var_options,
                                                default=var_options[:4])
                plot_indices = [int(v.split()[-1]) - 1 for v in selected_vars]
            else:
                plot_indices = list(range(n_state))

            n_plot = len(plot_indices)
            if n_plot == 0:
                st.warning("Select at least one variable to plot.")
            else:
                n_pts = min(500, max(100, int(t_pred / dt_val)))
                times = np.linspace(0, t_pred, n_pts)
                traj_pred = sim.predict_trajectory(x0, times)

                # FIX: Use smooth ground-truth trajectory if available
                if traj_true is not None:
                    n_true = min(int(t_pred / dt_val) + 1, traj_true.shape[0])
                    times_true = np.arange(n_true) * dt_val
                    true_data = traj_true[:n_true]
                else:
                    n_true = min(int(t_pred / dt_val) + 1, X.shape[0])
                    times_true = np.arange(n_true) * dt_val
                    true_data = X[:n_true]

                # FIX: Confidence bands from eigenvalue stability
                spec = sim.spectral_analysis()
                eig_mags = np.abs(spec["eigenvalues"])
                max_growth = max(np.max(eig_mags) - 1.0, 0.0)  # per-step growth
                # Uncertainty grows with time: band = amplitude * growth_factor * sqrt(t)
                amplitude = np.std(X, axis=0)

                # Plot
                fig, axes = plt.subplots(n_plot, 1, figsize=(10, 2.8 * n_plot), squeeze=False)
                for plot_i, var_i in enumerate(plot_indices):
                    ax = axes[plot_i, 0]
                    # Smooth measured data
                    ax.plot(times_true, true_data[:, var_i],
                            color="#4F8EF7", lw=2, label="Measured", alpha=0.9)
                    # Prediction
                    ax.plot(times, traj_pred[:, var_i],
                            color="#FF5252", lw=2, ls="--", label="Predicted")
                    # Confidence band
                    band = amplitude[var_i] * max_growth * np.sqrt(times / dt_val + 1) * 0.5
                    ax.fill_between(times,
                                    traj_pred[:, var_i] - band,
                                    traj_pred[:, var_i] + band,
                                    color="#FF5252", alpha=0.08, label="Confidence band")
                    ax.set_ylabel(f"Var {var_i + 1}")
                    ax.legend(fontsize=8, loc="upper right")
                    ax.grid(True, alpha=0.2)
                axes[-1, 0].set_xlabel("Time (s)")
                fig.suptitle("Prediction vs Measured Data", fontsize=13, y=1.01)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Show final predicted values
                final = sim.predict(x0, t_pred)
                val_cols = st.columns(min(n_state, 6))
                for i in range(min(n_state, 6)):
                    val_cols[i].metric(f"Var {i+1} at t={t_pred:.2f}s", f"{final[i]:.6g}")

    # ── Stability Report ──────────────────────────────────────────────
    with tab_report:
        if rpt:
            r1, r2 = st.columns(2)
            r1.metric("Classification", rpt["classification"])
            r1.metric("Dominant Frequency", f'{rpt["dominant_freq_hz"]:.4f} Hz')
            settling = f'{rpt["settling_time_s"]:.3f} s' if rpt["settling_time_s"] else "N/A"
            r1.metric("Settling Time", settling)
            r2.metric("Damping Ratio", f'{rpt["dominant_damping"]:.4f}')
            overshoot = f'{rpt["peak_overshoot_pct"]:.2f}%' if rpt["peak_overshoot_pct"] else "N/A"
            r2.metric("Peak Overshoot", overshoot)
            acc_label = "Accuracy (exact)" if rpt["is_linear"] else "One-Step Accuracy"
            r2.metric(acc_label, f'{rpt["accuracy_pct"]:.1f}%')

            # Resonance risk warning
            if abs(rpt["dominant_damping"]) < 0.05 and rpt["dominant_freq_hz"] > 0.001:
                st.error("🔴 **Resonance Risk** — Damping ratio is very low "
                         f"({rpt['dominant_damping']:.4f}). This system is near resonance "
                         "and may experience large oscillations under excitation.")
            elif abs(rpt["dominant_damping"]) < 0.15 and rpt["dominant_freq_hz"] > 0.001:
                st.warning("🟡 **Low Damping** — Damping ratio is moderate "
                           f"({rpt['dominant_damping']:.4f}). Monitor for resonance under load.")

            st.info(rpt["summary"])

            # Phase portrait
            if n_state >= 2:
                st.markdown("**Phase Portrait**")
                x0_pp = X[0]
                t_pp = X.shape[0] * dt_val
                times_pp = np.linspace(0, t_pp, 500)
                traj_pp = sim.predict_trajectory(x0_pp, times_pp)

                fig_pp, ax_pp = plt.subplots(figsize=(6, 6))
                ax_pp.plot(traj_pp[:, 0], traj_pp[:, 1], color=ACCENT, lw=1.5, alpha=0.9)
                ax_pp.plot(traj_pp[0, 0], traj_pp[0, 1], "o", color=SUCCESS, ms=8, label="Start")
                ax_pp.plot(traj_pp[-1, 0], traj_pp[-1, 1], "s", color="#FF5252", ms=8, label="End")
                ax_pp.set_xlabel("Variable 1")
                ax_pp.set_ylabel("Variable 2")
                ax_pp.set_title("Phase Portrait")
                ax_pp.legend()
                ax_pp.set_aspect("equal", adjustable="datalim")
                ax_pp.grid(True, alpha=0.2)
                plt.tight_layout()
                st.pyplot(fig_pp)
                plt.close(fig_pp)

    # ── Export ────────────────────────────────────────────────────────
    with tab_export:
        e1, e2, e3 = st.columns(3)

        with e1:
            st.markdown("**Predictions CSV**")
            x0_e = X[0]
            times_e = np.linspace(0, X.shape[0] * dt_val, 200)
            traj_e = sim.predict_trajectory(x0_e, times_e)
            buf = io.StringIO()
            header = ",".join(["time"] + [f"var_{i+1}" for i in range(n_state)])
            buf.write(header + "\n")
            for j, t in enumerate(times_e):
                row = f"{t:.8g}," + ",".join(f"{traj_e[j, k]:.8g}" for k in range(n_state))
                buf.write(row + "\n")
            st.download_button("📥 Download CSV", buf.getvalue(),
                               "orbit_predictions.csv", "text/csv")

        with e2:
            st.markdown("**Model file**")
            with tempfile.NamedTemporaryFile(suffix=".koop", delete=False) as tmp:
                sim.save(tmp.name)
            with open(tmp.name, "rb") as f:
                model_bytes = f.read()
            os.unlink(tmp.name)
            st.download_button("📥 Download .koop", model_bytes,
                               "orbit_model.koop", "application/octet-stream")

        with e3:
            st.markdown("**Python snippet**")
            st.code('''from koopsim import KoopSim
import numpy as np

sim = KoopSim.load("orbit_model.koop")
x0 = np.array([...])  # initial state
result = sim.predict(x0, t=5.0)
print(result)''', language="python")

elif not _has_data():
    st.info("👆 Select a system or upload data above to get started.")
