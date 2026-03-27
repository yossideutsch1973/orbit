"""Engineering analysis helpers for Orbit.

Translates raw eigenvalue data from the model into engineer-friendly
quantities: natural frequencies (Hz), damping ratios, settling times,
system classification, etc.

No references to 'Koopman', 'EDMD', or 'lifting' leak out of this module.
"""

from __future__ import annotations

import numpy as np


def engineering_report(eigenvalues: np.ndarray, dt: float) -> dict:
    """Build a full engineering dynamics report from discrete eigenvalues.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Complex eigenvalues of the discrete-time system matrix.
    dt : float
        Sampling period used during model fitting.

    Returns
    -------
    dict
        Keys:
        - classification (str)
        - natural_frequencies_hz (np.ndarray)
        - damping_ratios (np.ndarray)
        - settling_time_s (float or None)
        - peak_overshoot_pct (float or None)
        - dominant_freq_hz (float)
        - dominant_damping (float)
        - is_linear (bool)
        - summary (str)
    """
    eigenvalues = np.asarray(eigenvalues, dtype=complex)

    magnitudes = np.abs(eigenvalues)
    angles = np.angle(eigenvalues)

    # ---- continuous-time equivalents ----
    log_mag = np.log(np.clip(magnitudes, 1e-300, None))
    sigma = log_mag / dt  # real part of continuous eigenvalue
    omega_d = angles / dt  # damped frequency (rad/s)
    omega_n = np.sqrt(sigma**2 + omega_d**2)  # natural frequency (rad/s)

    # Damping ratio (guard division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta = np.where(omega_n > 1e-12, -sigma / omega_n, 0.0)

    freq_hz = np.abs(omega_d) / (2.0 * np.pi)

    # ---- dominant mode (largest |angle| that is non-trivially oscillatory) ----
    osc_mask = np.abs(angles) > 1e-6
    if np.any(osc_mask):
        osc_indices = np.where(osc_mask)[0]
        # pick the one with largest magnitude among oscillatory modes
        dom_idx = osc_indices[np.argmax(magnitudes[osc_indices])]
    else:
        dom_idx = np.argmax(magnitudes)

    dominant_freq_hz = float(freq_hz[dom_idx])
    dominant_damping = float(zeta[dom_idx])

    # ---- settling time (from slowest decaying stable mode) ----
    stable_mask = magnitudes < 1.0 - 1e-10
    if np.any(stable_mask):
        # slowest decay -> smallest |sigma| among stable modes
        abs_sigma_stable = np.abs(sigma[stable_mask])
        slowest = (
            np.min(abs_sigma_stable[abs_sigma_stable > 1e-12])
            if np.any(abs_sigma_stable > 1e-12)
            else None
        )
        settling_time = 4.0 / slowest if slowest is not None else None
    else:
        settling_time = None

    # ---- peak overshoot ----
    if dominant_damping > 0 and dominant_damping < 1.0:
        peak_overshoot = 100.0 * np.exp(
            -np.pi * dominant_damping / np.sqrt(1.0 - dominant_damping**2)
        )
    else:
        peak_overshoot = None

    # ---- classification ----
    near_unit = np.abs(magnitudes - 1.0) < 0.02
    is_linear = bool(np.all(near_unit | (magnitudes < 1.0)))

    all_stable = bool(np.all(magnitudes < 1.0 + 1e-6))
    has_limit_cycle = bool(np.any(near_unit & osc_mask))
    has_oscillation = bool(np.any(osc_mask & (magnitudes > 0.01)))

    if not all_stable:
        classification = "Unstable"
    elif has_limit_cycle:
        classification = "Limit Cycle"
    elif has_oscillation and np.all(magnitudes < 1.0 - 1e-6):
        classification = "Oscillatory (Damped)"
    elif has_oscillation:
        classification = "Oscillatory"
    else:
        classification = "Stable"

    # ---- human-readable summary ----
    summary = _build_summary(classification, dominant_freq_hz, dominant_damping, settling_time)

    return {
        "classification": classification,
        "natural_frequencies_hz": freq_hz,
        "damping_ratios": zeta,
        "settling_time_s": settling_time,
        "peak_overshoot_pct": peak_overshoot,
        "dominant_freq_hz": dominant_freq_hz,
        "dominant_damping": dominant_damping,
        "is_linear": is_linear,
        "summary": summary,
    }


def _build_summary(
    classification: str,
    freq_hz: float,
    damping: float,
    settling_time: float | None,
) -> str:
    parts: list[str] = []

    if classification == "Limit Cycle":
        period = 1.0 / freq_hz if freq_hz > 1e-6 else float("inf")
        parts.append(
            f"This system converges to a stable periodic orbit "
            f"with frequency {freq_hz:.3f} Hz (period {period:.3f} s)."
        )
    elif "Oscillatory" in classification:
        parts.append(f"This system oscillates at {freq_hz:.3f} Hz.")
        if settling_time is not None:
            parts.append(
                f"It settles to equilibrium in approximately {settling_time:.2f} seconds."
            )
    elif classification == "Unstable":
        parts.append("This system is unstable -- predictions diverge from equilibrium over time.")
    else:
        parts.append("This system is stable and converges to equilibrium.")
        if settling_time is not None:
            parts.append(f"Settling time: approximately {settling_time:.2f} seconds.")

    if 0 < damping < 1:
        parts.append(f"Damping ratio: {damping:.3f}.")

    return " ".join(parts)


def compute_accuracy_pct(
    model,
    X: np.ndarray,
    Y: np.ndarray,
) -> float:
    """Compute a prediction accuracy percentage on held-out data.

    Returns a value between 0 and 100 representing how well one-step
    predictions match ground truth (100 - relative error * 100).
    """
    from koopsim.core.validation import ModelValidator

    rel_error = ModelValidator.prediction_error(model, X, Y, metric="relative")
    accuracy = max(0.0, (1.0 - rel_error) * 100.0)
    return float(accuracy)
