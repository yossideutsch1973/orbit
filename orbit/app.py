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
                      "model": None, "report": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _has_data() -> bool:
    return st.session_state["X"] is not None


def _has_model() -> bool:
    return st.session_state["model"] is not None


def _run_demo(system, name, poly_degree=None, rbf_centers=None):
    """Generate data, fit model, compute report. Called from button callbacks."""
    rng = np.random.default_rng(42)
    x0 = rng.standard_normal(system.state_dim) * 0.5
    dt = 0.01
    X, Y = system.generate_snapshots(x0, dt=dt, n_steps=100, n_trajectories=10)
    st.session_state["X"] = X
    st.session_state["Y"] = Y
    st.session_state["dt"] = dt
    st.session_state["system_name"] = name

    sim = KoopSim(method="edmd", poly_degree=poly_degree,
                  rbf_centers=rbf_centers, verbose=False)
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
st.markdown("*Upload your data. Predict the future.*")
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
    if uploaded is not None:
        if st.button("Load data", key="btn_csv"):
            content = uploaded.read().decode("utf-8")
            data = np.loadtxt(io.StringIO(content), delimiter=",")
            if data.ndim == 2 and data.shape[0] >= 3:
                st.session_state["X"] = data[:-1]
                st.session_state["Y"] = data[1:]
                st.session_state["dt"] = upload_dt
                st.session_state["system_name"] = uploaded.name
                st.session_state["model"] = None
                st.session_state["report"] = None
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
        X, Y = system.generate_snapshots(x0, dt=0.01, n_steps=100, n_trajectories=10)
        st.session_state["X"] = X
        st.session_state["Y"] = Y
        st.session_state["dt"] = 0.01
        st.session_state["system_name"] = sys_choice
        st.session_state["model"] = None
        st.session_state["report"] = None
        st.rerun()

# ── One-click demos ──────────────────────────────────────────────────
with col_demo:
    st.markdown("**Try a demo**")
    st.caption("One click — instant results.")

    if st.button("⚡  RLC Circuit Demo", key="btn_rlc"):
        _run_demo(RLCCircuit(R=1.0, L=1.0, C=1.0), "RLC Circuit Demo")
        st.rerun()

    if st.button("🔧  Vibration Demo", key="btn_vib"):
        _run_demo(SpringMassDamper(n_masses=3, k=1.0, c=0.1, m=1.0), "Vibration Demo")
        st.rerun()

    if st.button("🌀  Limit Cycle Demo", key="btn_hopf"):
        _run_demo(HopfBifurcation(mu=1.0), "Limit Cycle Demo", poly_degree=3, rbf_centers=30)
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
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prediction Accuracy", f'{rpt["accuracy_pct"]:.1f}%')
        m2.metric("System Type", "Linear" if rpt["is_linear"] else "Nonlinear")
        m3.metric("Dominant Frequency", f'{rpt["dominant_freq_hz"]:.3f} Hz')
        settling = f'{rpt["settling_time_s"]:.2f} s' if rpt["settling_time_s"] else "N/A"
        m4.metric("Settling Time", settling)

        if rpt["is_linear"]:
            st.info("✅ Linear system detected — predictions are exact.")
        else:
            st.info("ℹ️ Nonlinear system — predictions reliable for short-to-medium horizons.")

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
                                float(training_window * 10), float(training_window))

            n_pts = min(500, max(100, int(t_pred / dt_val)))
            times = np.linspace(0, t_pred, n_pts)
            traj_pred = sim.predict_trajectory(x0, times)

            # Ground truth from training data
            n_true = min(int(t_pred / dt_val) + 1, X.shape[0])
            times_true = np.arange(n_true) * dt_val

            # Plot
            fig, axes = plt.subplots(n_state, 1, figsize=(10, 3 * n_state), squeeze=False)
            for i in range(n_state):
                ax = axes[i, 0]
                ax.plot(times_true, X[:n_true, i], color="#4F8EF7", lw=2, label="Measured")
                ax.plot(times, traj_pred[:, i], color="#FF5252", lw=2, ls="--", label="Predicted")
                ax.set_ylabel(f"Variable {i + 1}")
                ax.legend(fontsize=9, loc="upper right")
                ax.grid(True, alpha=0.3)
            axes[-1, 0].set_xlabel("Time (s)")
            fig.suptitle("Prediction vs Measured Data", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Show final predicted values
            final = sim.predict(x0, t_pred)
            cols = st.columns(n_state)
            for i in range(n_state):
                cols[i].metric(f"Var {i+1} at t={t_pred:.2f}s", f"{final[i]:.6g}")

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
            r2.metric("Accuracy", f'{rpt["accuracy_pct"]:.1f}%')

            st.info(rpt["summary"])

            # Phase portrait
            if n_state >= 2:
                x0_pp = X[0]
                t_pp = X.shape[0] * dt_val
                times_pp = np.linspace(0, t_pp, 500)
                traj_pp = sim.predict_trajectory(x0_pp, times_pp)

                fig_pp, ax_pp = plt.subplots(figsize=(6, 6))
                ax_pp.plot(traj_pp[:, 0], traj_pp[:, 1], color=ACCENT, lw=1.5)
                ax_pp.plot(traj_pp[0, 0], traj_pp[0, 1], "o", color=SUCCESS, ms=8, label="Start")
                ax_pp.plot(traj_pp[-1, 0], traj_pp[-1, 1], "s", color="#FF5252", ms=8, label="End")
                ax_pp.set_xlabel("Variable 1")
                ax_pp.set_ylabel("Variable 2")
                ax_pp.set_title("Phase Portrait")
                ax_pp.legend()
                ax_pp.set_aspect("equal", adjustable="datalim")
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
            st.download_button("Download CSV", buf.getvalue(),
                               "orbit_predictions.csv", "text/csv")

        with e2:
            st.markdown("**Model file**")
            with tempfile.NamedTemporaryFile(suffix=".koop", delete=False) as tmp:
                sim.save(tmp.name)
            with open(tmp.name, "rb") as f:
                model_bytes = f.read()
            os.unlink(tmp.name)
            st.download_button("Download .koop", model_bytes,
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
