"""KoopSim Streamlit Dashboard.

Run with: streamlit run koopsim/dashboard.py
"""
from __future__ import annotations

import logging
import tempfile

import numpy as np
import streamlit as st

logger = logging.getLogger("koopsim")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="KoopSim Dashboard",
    page_icon="~",
    layout="wide",
)

st.title("KoopSim -- Koopman Operator Simulation Toolkit")

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "X": None,
    "Y": None,
    "dt": None,
    "sim": None,
    "data_info": None,
    "train_log": None,
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# System registry  (lazy -- avoids importing koopsim at module level)
# ---------------------------------------------------------------------------

SYSTEM_REGISTRY: dict[str, dict] = {
    "hopf": {
        "label": "Hopf Bifurcation",
        "class": "HopfBifurcation",
        "module": "koopsim.systems.fluid_particles",
    },
    "double-gyre": {
        "label": "Double Gyre",
        "class": "DoubleGyre",
        "module": "koopsim.systems.fluid_grid",
    },
    "spring-mass": {
        "label": "Spring-Mass-Damper",
        "class": "SpringMassDamper",
        "module": "koopsim.systems.mechanical",
    },
    "rlc": {
        "label": "RLC Circuit",
        "class": "RLCCircuit",
        "module": "koopsim.systems.circuit",
    },
    "vanderpol": {
        "label": "Van der Pol Oscillator",
        "class": "VanDerPolOscillator",
        "module": "koopsim.systems.mechanical",
    },
    "beam": {
        "label": "Euler-Bernoulli Beam",
        "class": "EulerBernoulliBeam",
        "module": "koopsim.systems.mechanical",
    },
    "point-vortex": {
        "label": "Point Vortex System",
        "class": "PointVortexSystem",
        "module": "koopsim.systems.fluid_particles",
    },
}


def _get_system_instance(key: str):
    """Lazily import and instantiate a built-in system."""
    import importlib

    info = SYSTEM_REGISTRY[key]
    mod = importlib.import_module(info["module"])
    cls = getattr(mod, info["class"])
    return cls()


# ===================================================================
# Tab definitions
# ===================================================================

tab_data, tab_train, tab_predict, tab_analysis = st.tabs(
    ["Data", "Train", "Predict", "Analysis"]
)

# -------------------------------------------------------------------
# Tab 1: Data
# -------------------------------------------------------------------

with tab_data:
    st.header("Data")
    data_source = st.radio(
        "Data source",
        ["Built-in system", "Upload file"],
        horizontal=True,
    )

    if data_source == "Upload file":
        uploaded = st.file_uploader(
            "Upload snapshot data (CSV or NPY)",
            type=["csv", "npy"],
        )
        upload_dt = st.number_input(
            "Time step (dt) between snapshots",
            min_value=1e-6,
            value=0.01,
            format="%.6f",
            key="upload_dt",
        )

        if uploaded is not None and st.button("Load uploaded data", key="btn_load_upload"):
            try:
                with st.spinner("Loading file..."):
                    if uploaded.name.endswith(".csv"):
                        import io
                        content = uploaded.read().decode("utf-8")
                        data = np.loadtxt(io.StringIO(content), delimiter=",")
                    else:
                        import io as _io
                        data = np.load(_io.BytesIO(uploaded.read()))

                    if data.ndim != 2 or data.shape[0] < 2:
                        st.error(
                            "Data must be a 2-D array with at least 2 rows "
                            "(consecutive snapshots)."
                        )
                    else:
                        X = data[:-1]
                        Y = data[1:]
                        st.session_state["X"] = X
                        st.session_state["Y"] = Y
                        st.session_state["dt"] = upload_dt
                        st.session_state["data_info"] = {
                            "source": f"Uploaded: {uploaded.name}",
                            "shape_X": X.shape,
                        }
                        st.success(
                            f"Loaded {X.shape[0]} snapshot pairs with "
                            f"{X.shape[1]} state dimensions."
                        )
            except Exception as exc:
                st.error(f"Failed to load file: {exc}")

    else:
        # Built-in system selection
        sys_key = st.selectbox(
            "System",
            list(SYSTEM_REGISTRY.keys()),
            format_func=lambda k: SYSTEM_REGISTRY[k]["label"],
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            gen_dt = st.slider(
                "dt", min_value=0.001, max_value=0.1, value=0.01, step=0.001,
                key="gen_dt",
            )
        with col2:
            gen_n_steps = st.slider(
                "n_steps", min_value=50, max_value=500, value=100, step=10,
                key="gen_n_steps",
            )
        with col3:
            gen_n_traj = st.slider(
                "n_trajectories", min_value=1, max_value=50, value=10, step=1,
                key="gen_n_traj",
            )
        with col4:
            gen_noise = st.slider(
                "noise_std", min_value=0.0, max_value=0.1, value=0.0,
                step=0.001, key="gen_noise",
            )

        if st.button("Generate snapshots", key="btn_generate"):
            try:
                with st.spinner("Generating trajectories..."):
                    system = _get_system_instance(sys_key)
                    rng = np.random.default_rng(42)
                    x0 = rng.standard_normal(system.state_dim) * 0.5
                    X, Y = system.generate_snapshots(
                        x0,
                        dt=gen_dt,
                        n_steps=gen_n_steps,
                        n_trajectories=gen_n_traj,
                        noise_std=gen_noise,
                    )
                    st.session_state["X"] = X
                    st.session_state["Y"] = Y
                    st.session_state["dt"] = gen_dt
                    st.session_state["data_info"] = {
                        "source": SYSTEM_REGISTRY[sys_key]["label"],
                        "shape_X": X.shape,
                    }
                    st.success(
                        f"Generated {X.shape[0]} snapshot pairs "
                        f"({X.shape[1]} state dims) from "
                        f"{SYSTEM_REGISTRY[sys_key]['label']}."
                    )
            except Exception as exc:
                st.error(f"Failed to generate data: {exc}")

    # Preview section
    st.subheader("Data preview")
    if st.session_state["X"] is not None:
        X = st.session_state["X"]
        Y = st.session_state["Y"]
        info = st.session_state["data_info"]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Source:** {info['source']}")
            st.markdown(f"**X shape:** `{X.shape}`  |  **Y shape:** `{Y.shape}`")
            st.markdown(f"**dt:** `{st.session_state['dt']}`")
        with col_b:
            st.markdown("**Basic statistics (X):**")
            stats = {
                "min": np.min(X, axis=0),
                "max": np.max(X, axis=0),
                "mean": np.mean(X, axis=0),
                "std": np.std(X, axis=0),
            }
            import pandas as pd

            dim_labels = [f"dim_{i}" for i in range(X.shape[1])]
            stats_df = pd.DataFrame(stats, index=dim_labels).T
            st.dataframe(stats_df, use_container_width=True)

        with st.expander("First 10 rows of X"):
            st.dataframe(
                pd.DataFrame(X[:10], columns=dim_labels),
                use_container_width=True,
            )
    else:
        st.info("No data loaded yet. Generate from a built-in system or upload a file above.")

# -------------------------------------------------------------------
# Tab 2: Train
# -------------------------------------------------------------------

with tab_train:
    st.header("Train Koopman Model")

    if st.session_state["X"] is None:
        st.warning("Load or generate data in the **Data** tab first.")
    else:
        method = st.selectbox("Method", ["edmd", "neural"], key="train_method")

        if method == "edmd":
            st.subheader("EDMD parameters")
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                use_poly = st.checkbox("Polynomial dictionary", value=False, key="use_poly")
                poly_degree = st.slider(
                    "poly_degree", min_value=2, max_value=5, value=3,
                    key="poly_degree", disabled=not use_poly,
                )
                use_rbf = st.checkbox("RBF dictionary", value=False, key="use_rbf")
                rbf_centers = st.slider(
                    "rbf_centers", min_value=10, max_value=100, value=50,
                    key="rbf_centers", disabled=not use_rbf,
                )
            with col_e2:
                reg = st.slider(
                    "regularization (log10 scale)",
                    min_value=-8.0,
                    max_value=-2.0,
                    value=-6.0,
                    step=0.5,
                    key="edmd_reg",
                )
                regularization = 10.0 ** reg
                st.text(f"regularization = {regularization:.2e}")

                use_svd = st.checkbox("SVD rank truncation", value=False, key="use_svd")
                n_koopman_est = st.session_state["X"].shape[1]
                svd_rank = st.slider(
                    "svd_rank",
                    min_value=1,
                    max_value=max(n_koopman_est * 3, 10),
                    value=min(n_koopman_est, 10),
                    key="svd_rank",
                    disabled=not use_svd,
                )

        else:
            st.subheader("Neural Koopman parameters")
            col_n1, col_n2 = st.columns(2)
            with col_n1:
                latent_dim = st.slider(
                    "latent_dim", min_value=4, max_value=64, value=16,
                    key="latent_dim",
                )
                max_epochs = st.slider(
                    "max_epochs", min_value=10, max_value=500, value=100,
                    key="max_epochs",
                )
            with col_n2:
                lr_log = st.slider(
                    "learning rate (log10 scale)",
                    min_value=-5.0,
                    max_value=-2.0,
                    value=-3.0,
                    step=0.5,
                    key="neural_lr",
                )
                lr = 10.0 ** lr_log
                st.text(f"lr = {lr:.2e}")

        if st.button("Train model", key="btn_train"):
            try:
                with st.spinner("Training..."):
                    from koopsim.koopsim import KoopSim as _KoopSim

                    kwargs: dict = {"method": method, "verbose": False}

                    if method == "edmd":
                        kwargs["poly_degree"] = poly_degree if use_poly else None
                        kwargs["rbf_centers"] = rbf_centers if use_rbf else None
                        kwargs["regularization"] = regularization
                        kwargs["svd_rank"] = svd_rank if use_svd else None
                    else:
                        kwargs["latent_dim"] = latent_dim
                        kwargs["max_epochs"] = max_epochs
                        kwargs["lr"] = lr

                    sim = _KoopSim(**kwargs)
                    sim.fit(
                        st.session_state["X"],
                        st.session_state["Y"],
                        st.session_state["dt"],
                    )
                    st.session_state["sim"] = sim

                    # Collect training metrics
                    model = sim.model
                    K = model.get_koopman_matrix()
                    eigvals = np.linalg.eigvals(K)
                    cond = float(np.linalg.cond(K))
                    st.session_state["train_log"] = {
                        "method": method,
                        "K_shape": K.shape,
                        "condition_number": cond,
                        "eigenvalue_magnitudes": np.abs(eigvals),
                    }
                st.success("Model trained successfully!")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

        # Show training metrics
        if st.session_state["train_log"] is not None:
            st.subheader("Training metrics")
            tlog = st.session_state["train_log"]
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Method", tlog["method"].upper())
            with col_m2:
                st.metric(
                    "Koopman matrix shape",
                    f"{tlog['K_shape'][0]} x {tlog['K_shape'][1]}",
                )
            with col_m3:
                st.metric("Condition number", f"{tlog['condition_number']:.4g}")

            mags = tlog["eigenvalue_magnitudes"]
            st.markdown("**Eigenvalue summary:**")
            summary_cols = st.columns(4)
            with summary_cols[0]:
                st.metric("Count", len(mags))
            with summary_cols[1]:
                st.metric("Max |lambda|", f"{np.max(mags):.6f}")
            with summary_cols[2]:
                st.metric("Min |lambda|", f"{np.min(mags):.6f}")
            with summary_cols[3]:
                n_inside = int(np.sum(mags <= 1.0))
                st.metric("Inside unit circle", f"{n_inside}/{len(mags)}")

# -------------------------------------------------------------------
# Tab 3: Predict
# -------------------------------------------------------------------

with tab_predict:
    st.header("Predict")

    if st.session_state["sim"] is None:
        st.warning("Train a model in the **Train** tab first.")
    else:
        sim = st.session_state["sim"]
        n_state = sim.model.n_state_dims
        dt = st.session_state["dt"]

        # Default initial state from training data
        default_x0 = st.session_state["X"][0] if st.session_state["X"] is not None else np.zeros(n_state)
        default_str = ", ".join(f"{v:.6g}" for v in default_x0)

        st.subheader("Initial state")
        x0_input = st.text_input(
            f"Initial state ({n_state} values, comma-separated)",
            value=default_str,
            key="x0_input",
        )

        try:
            x0 = np.array([float(v.strip()) for v in x0_input.split(",")])
            if len(x0) != n_state:
                st.error(
                    f"Expected {n_state} values, got {len(x0)}. "
                    f"Please enter exactly {n_state} comma-separated numbers."
                )
                x0 = None
        except ValueError:
            st.error("Could not parse initial state. Enter comma-separated numbers.")
            x0 = None

        if x0 is not None:
            # --- Single time prediction ---
            st.subheader("Single-time prediction")
            t_single = st.number_input(
                "Prediction time t",
                min_value=0.0,
                value=float(dt * 10),
                format="%.6f",
                key="t_single",
            )
            if st.button("Predict at t", key="btn_predict_single"):
                try:
                    result = sim.predict(x0, t_single)
                    st.success(f"State at t = {t_single}:")
                    st.code(
                        "  ".join(f"{v:.8g}" for v in result),
                        language=None,
                    )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

            # --- Trajectory prediction ---
            st.subheader("Trajectory prediction")
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                t_start = st.number_input(
                    "t_start", min_value=0.0, value=0.0,
                    format="%.4f", key="t_start",
                )
            with col_t2:
                t_end = st.number_input(
                    "t_end", min_value=0.0, value=float(dt * 100),
                    format="%.4f", key="t_end",
                )
            with col_t3:
                n_points = st.slider(
                    "n_points", min_value=10, max_value=1000, value=200,
                    key="n_points",
                )

            if st.button("Predict trajectory", key="btn_predict_traj"):
                if t_end <= t_start:
                    st.error("t_end must be greater than t_start.")
                else:
                    try:
                        with st.spinner("Computing trajectory..."):
                            times = np.linspace(t_start, t_end, n_points)
                            traj = sim.predict_trajectory(x0, times)

                        st.success(
                            f"Trajectory computed: {traj.shape[0]} time points, "
                            f"{traj.shape[1]} state dims."
                        )

                        # Plot with visualization utilities
                        from koopsim.utils.visualization import (
                            plot_phase_portrait,
                            plot_trajectory_comparison,
                        )

                        # Time-series plot (predicted only)
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

                        fig_ts, axes = plt.subplots(
                            n_state, 1,
                            figsize=(10, max(3 * n_state, 4)),
                            squeeze=False,
                        )
                        for i in range(n_state):
                            ax = axes[i, 0]
                            ax.plot(times, traj[:, i], linewidth=1.5)
                            ax.set_ylabel(f"dim {i}")
                            ax.grid(True, alpha=0.3)
                        axes[-1, 0].set_xlabel("Time")
                        fig_ts.suptitle("Predicted Trajectory")
                        plt.tight_layout()
                        st.pyplot(fig_ts)
                        plt.close(fig_ts)

                        # Phase portrait (if 2+ dims)
                        if n_state >= 2:
                            fig_pp = plot_phase_portrait(
                                traj, dims=(0, 1), backend="matplotlib",
                            )
                            st.pyplot(fig_pp)
                            plt.close(fig_pp)

                    except Exception as exc:
                        st.error(f"Trajectory prediction failed: {exc}")

# -------------------------------------------------------------------
# Tab 4: Analysis
# -------------------------------------------------------------------

with tab_analysis:
    st.header("Analysis")

    if st.session_state["sim"] is None:
        st.warning("Train a model in the **Train** tab first.")
    else:
        sim = st.session_state["sim"]

        # --- Eigenspectrum ---
        st.subheader("Eigenspectrum")
        try:
            from koopsim.utils.visualization import plot_eigenspectrum

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig_eigen = plot_eigenspectrum(sim.model, backend="matplotlib")
            st.pyplot(fig_eigen)
            plt.close(fig_eigen)
        except Exception as exc:
            st.error(f"Failed to plot eigenspectrum: {exc}")

        # --- Spectral analysis table ---
        st.subheader("Spectral analysis")
        try:
            spec = sim.spectral_analysis()
            import pandas as pd

            n_modes = len(spec["eigenvalues"])
            # Sort by dominant mode order
            order = spec["dominant_mode_indices"]
            rows = []
            for idx in order:
                ev = spec["eigenvalues"][idx]
                rows.append({
                    "Mode": int(idx),
                    "Eigenvalue": f"{ev.real:.6f} + {ev.imag:.6f}j",
                    "|lambda|": f"{abs(ev):.6f}",
                    "Frequency (rad/s)": f"{spec['frequencies'][idx]:.6f}",
                    "Growth rate": f"{spec['growth_rates'][idx]:.6f}",
                })
            spec_df = pd.DataFrame(rows)
            st.dataframe(spec_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Spectral analysis failed: {exc}")

        # --- Uncertainty quantification ---
        st.subheader("Uncertainty quantification")
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            uq_n_samples = st.slider(
                "n_samples (MC)", min_value=10, max_value=500, value=100,
                key="uq_n_samples",
            )
        with col_u2:
            uq_noise_log = st.slider(
                "noise_scale (log10)",
                min_value=-4.0, max_value=-1.0, value=-2.0, step=0.5,
                key="uq_noise_scale",
            )
            uq_noise_scale = 10.0 ** uq_noise_log
            st.text(f"noise_scale = {uq_noise_scale:.2e}")

        uq_t = st.number_input(
            "Prediction time for UQ",
            min_value=0.0,
            value=float(st.session_state["dt"] * 50) if st.session_state["dt"] else 0.5,
            format="%.4f",
            key="uq_t",
        )

        if st.button("Run uncertainty quantification", key="btn_uq"):
            try:
                with st.spinner("Running Monte Carlo UQ..."):
                    X = st.session_state["X"]
                    x0 = X[0] if X is not None else np.zeros(sim.model.n_state_dims)
                    uq_result = sim.predict_with_uncertainty(
                        x0, uq_t,
                        n_samples=uq_n_samples,
                        noise_scale=uq_noise_scale,
                    )

                st.markdown(f"**Mean prediction:** `{uq_result['mean']}`")
                st.markdown(f"**Std deviation:** `{uq_result['std']}`")

                # Plot uncertainty band over a range of times
                from koopsim.utils.visualization import plot_uncertainty_band

                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                dt_val = st.session_state["dt"]
                n_uq_times = 50
                uq_times = np.linspace(0.0, uq_t, n_uq_times)

                # Collect mean and std at each time point
                n_dims = sim.model.n_state_dims
                means = np.zeros((n_uq_times, n_dims))
                stds = np.zeros((n_uq_times, n_dims))

                for i, t_val in enumerate(uq_times):
                    res = sim.predict_with_uncertainty(
                        x0, t_val,
                        n_samples=max(uq_n_samples // 5, 10),
                        noise_scale=uq_noise_scale,
                    )
                    means[i] = res["mean"]
                    stds[i] = res["std"]

                fig_uq = plot_uncertainty_band(
                    uq_times, means, stds, backend="matplotlib",
                )
                st.pyplot(fig_uq)
                plt.close(fig_uq)

            except Exception as exc:
                st.error(f"Uncertainty quantification failed: {exc}")

        # --- Multi-step prediction error ---
        st.subheader("Prediction error (multi-step)")
        if st.session_state["X"] is not None:
            err_n_steps = st.slider(
                "Number of prediction steps",
                min_value=5, max_value=200, value=50,
                key="err_n_steps",
            )
            if st.button("Compute multi-step error", key="btn_error"):
                try:
                    with st.spinner("Computing multi-step error..."):
                        from koopsim.core.validation import ModelValidator
                        from koopsim.utils.visualization import plot_prediction_error

                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

                        X = st.session_state["X"]
                        Y = st.session_state["Y"]
                        dt_val = st.session_state["dt"]

                        # Build a short ground-truth trajectory from the data.
                        # Use the first trajectory's consecutive snapshots.
                        max_len = min(err_n_steps + 1, X.shape[0])
                        trajectory = np.vstack([X[:max_len], Y[max_len - 1 : max_len]])
                        # Trim to actual available length
                        actual_steps = trajectory.shape[0] - 1
                        if actual_steps < 1:
                            st.error("Not enough data for multi-step error.")
                        else:
                            errors = ModelValidator.multi_step_error(
                                sim.model, trajectory, dt_val, actual_steps,
                            )
                            steps = np.arange(1, actual_steps + 1)

                            fig_err = plot_prediction_error(
                                steps, errors, backend="matplotlib",
                            )
                            st.pyplot(fig_err)
                            plt.close(fig_err)

                            st.markdown(
                                f"**Final RMSE (step {actual_steps}):** "
                                f"`{errors[-1]:.6g}`  |  "
                                f"**Mean RMSE:** `{np.mean(errors):.6g}`"
                            )
                except Exception as exc:
                    st.error(f"Error computation failed: {exc}")
        else:
            st.info("Load data in the **Data** tab to compute prediction error.")

        # --- Save / Export model ---
        st.subheader("Save / Export model")
        export_path = st.text_input(
            "File path (.koop)",
            value="model.koop",
            key="export_path",
        )
        if st.button("Save model", key="btn_save"):
            try:
                with st.spinner("Saving model..."):
                    sim.save(export_path)
                st.success(f"Model saved to `{export_path}`.")
            except Exception as exc:
                st.error(f"Failed to save model: {exc}")
