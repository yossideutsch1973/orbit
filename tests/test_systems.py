"""Tests for Phase 6 — Domain Systems (data generators)."""

from __future__ import annotations

import numpy as np
import pytest

from koopsim.core.edmd import EDMD
from koopsim.systems.base import DynamicalSystem
from koopsim.systems.chaotic import LorenzAttractor, LotkaVolterra
from koopsim.systems.circuit import RLCCircuit
from koopsim.systems.fluid_grid import DoubleGyre
from koopsim.systems.fluid_particles import HopfBifurcation, PointVortexSystem
from koopsim.systems.mechanical import (
    EulerBernoulliBeam,
    SpringMassDamper,
    VanDerPolOscillator,
)
from koopsim.utils.dictionary import IdentityDictionary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALL_SYSTEMS: list[tuple[DynamicalSystem, np.ndarray]] = [
    (HopfBifurcation(mu=1.0), np.array([0.5, 0.5])),
    (PointVortexSystem(n_vortices=3), np.array([0.0, 0.0, 1.0, 0.0, 0.5, 0.866])),
    (DoubleGyre(), np.array([0.5, 0.5])),
    (SpringMassDamper(n_masses=3), np.array([0.1, 0.0, -0.1, 0.0, 0.0, 0.0])),
    (EulerBernoulliBeam(n_elements=3), np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])),
    (VanDerPolOscillator(mu=1.0), np.array([1.0, 0.0])),
    (RLCCircuit(R=1.0, L=1.0, C=1.0), np.array([1.0, 0.0])),
    (LorenzAttractor(), np.array([1.0, 1.0, 1.0])),
    (LotkaVolterra(), np.array([2.0, 1.0])),
]


@pytest.fixture(params=ALL_SYSTEMS, ids=lambda s: s[0].name)
def system_and_ic(request):
    """Parametrised fixture yielding (system, x0) tuples."""
    return request.param


# ---------------------------------------------------------------------------
# 1. Shape tests — generate_trajectory returns correct shape
# ---------------------------------------------------------------------------


class TestTrajectoryShape:
    """Each system's generate_trajectory should return the correct shape."""

    def test_trajectory_shape(self, system_and_ic):
        system, x0 = system_and_ic
        dt = 0.01
        n_steps = 50
        traj = system.generate_trajectory(x0, dt, n_steps)
        assert traj.shape == (n_steps + 1, system.state_dim)


# ---------------------------------------------------------------------------
# 2. Snapshot pair tests — generate_snapshots returns correct X, Y shapes
# ---------------------------------------------------------------------------


class TestSnapshotPairShape:
    """generate_snapshots should return X, Y with the correct shapes."""

    def test_single_trajectory(self, system_and_ic):
        system, x0 = system_and_ic
        dt = 0.01
        n_steps = 30
        X, Y = system.generate_snapshots(x0, dt, n_steps, n_trajectories=1)
        assert X.shape == (n_steps, system.state_dim)
        assert Y.shape == (n_steps, system.state_dim)

    def test_multiple_trajectories(self, system_and_ic):
        system, x0 = system_and_ic
        dt = 0.01
        n_steps = 20
        n_traj = 3
        rng = np.random.default_rng(0)
        X, Y = system.generate_snapshots(
            x0, dt, n_steps, n_trajectories=n_traj, rng=rng
        )
        assert X.shape == (n_steps * n_traj, system.state_dim)
        assert Y.shape == (n_steps * n_traj, system.state_dim)

    def test_list_of_ics(self, system_and_ic):
        system, x0 = system_and_ic
        ics = [x0, x0 * 0.9, x0 * 1.1]
        dt = 0.01
        n_steps = 10
        X, Y = system.generate_snapshots(ics, dt, n_steps)
        assert X.shape == (n_steps * 3, system.state_dim)
        assert Y.shape == (n_steps * 3, system.state_dim)


# ---------------------------------------------------------------------------
# 3. RLC analytical — underdamped comparison
# ---------------------------------------------------------------------------


class TestRLCAnalytical:
    """RLC circuit should match its analytical solution in the underdamped case."""

    def test_underdamped_solution(self):
        R, L, C = 1.0, 1.0, 0.5
        rlc = RLCCircuit(R=R, L=L, C=C)

        q0, i0 = 1.0, 0.0
        x0 = np.array([q0, i0])
        dt = 0.005
        n_steps = 400

        traj = rlc.generate_trajectory(x0, dt, n_steps)

        # Analytical: characteristic equation s^2 + (R/L)s + 1/(LC) = 0
        alpha = R / (2.0 * L)
        omega_0_sq = 1.0 / (L * C)
        discriminant = alpha ** 2 - omega_0_sq

        assert discriminant < 0, "Expected underdamped case"

        omega_d = np.sqrt(-discriminant)
        times = np.arange(n_steps + 1) * dt

        # q(t) = exp(-alpha*t) * (q0*cos(omega_d*t) + ((i0 + alpha*q0)/omega_d)*sin(omega_d*t))
        A_coeff = q0
        B_coeff = (i0 + alpha * q0) / omega_d
        q_analytical = np.exp(-alpha * times) * (
            A_coeff * np.cos(omega_d * times) + B_coeff * np.sin(omega_d * times)
        )

        np.testing.assert_allclose(traj[:, 0], q_analytical, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# 4. Hopf limit cycle — radius converges to sqrt(mu) for mu > 0
# ---------------------------------------------------------------------------


class TestHopfLimitCycle:
    """For mu > 0, the Hopf bifurcation should approach a limit cycle."""

    def test_limit_cycle_radius(self):
        mu = 2.0
        hopf = HopfBifurcation(mu=mu)
        x0 = np.array([0.1, 0.1])
        dt = 0.01
        n_steps = 5000

        traj = hopf.generate_trajectory(x0, dt, n_steps)

        # Last portion of trajectory should be on the limit cycle
        tail = traj[-500:]
        radii = np.sqrt(tail[:, 0] ** 2 + tail[:, 1] ** 2)
        expected_radius = np.sqrt(mu)

        np.testing.assert_allclose(
            radii.mean(), expected_radius, atol=0.05,
            err_msg=f"Mean radius {radii.mean():.4f} does not match sqrt(mu)={expected_radius:.4f}",
        )
        # Radius should not vary much on the limit cycle
        assert radii.std() < 0.05, f"Radius std too large: {radii.std():.4f}"


# ---------------------------------------------------------------------------
# 5. Spring-mass energy conservation (undamped case, c=0)
# ---------------------------------------------------------------------------


class TestSpringMassEnergy:
    """Total energy should be approximately conserved for undamped case."""

    def test_energy_conservation_undamped(self):
        n = 3
        k = 2.0
        m = 1.0
        smd = SpringMassDamper(n_masses=n, k=k, c=0.0, m=m)

        x0 = np.array([1.0, 0.0, -0.5, 0.0, 0.0, 0.0])
        dt = 0.01
        n_steps = 2000

        traj = smd.generate_trajectory(x0, dt, n_steps)

        # Compute kinetic + potential energy at each time step
        energies = np.zeros(n_steps + 1)
        K_mat = smd._K_mat
        for idx in range(n_steps + 1):
            q = traj[idx, :n]
            v = traj[idx, n:]
            KE = 0.5 * m * np.sum(v ** 2)
            PE = 0.5 * q @ K_mat @ q
            energies[idx] = KE + PE

        # Energy should be approximately constant
        relative_variation = (energies.max() - energies.min()) / energies[0]
        assert relative_variation < 1e-5, (
            f"Energy not conserved: relative variation = {relative_variation:.2e}"
        )


# ---------------------------------------------------------------------------
# 6. End-to-end smoke test: generate_snapshots -> EDMD -> predict -> finite error
# ---------------------------------------------------------------------------


class TestEndToEndSmoke:
    """For each system, generate data, fit EDMD, predict, verify finite error."""

    def test_smoke(self, system_and_ic):
        system, x0 = system_and_ic
        dt = 0.01
        n_steps = 100
        rng = np.random.default_rng(42)

        X, Y = system.generate_snapshots(
            x0, dt, n_steps, n_trajectories=3, rng=rng
        )

        model = EDMD(dictionary=IdentityDictionary(), regularization=1e-6)
        model.fit(X, Y, dt)

        K = model.get_koopman_matrix()

        # One-step prediction
        Psi_X = model.lift(X[:10])
        Psi_Y_pred = Psi_X @ K
        Y_pred = model.unlift(Psi_Y_pred)

        error = np.linalg.norm(Y_pred - Y[:10])
        assert np.isfinite(error), "Prediction error is not finite"
        # The error should be at least in a reasonable range (not NaN/Inf)
        assert error < 1e6, f"Prediction error unreasonably large: {error:.2e}"


# ---------------------------------------------------------------------------
# 7. VanDerPol trajectory — generates without errors, state remains bounded
# ---------------------------------------------------------------------------


class TestVanDerPolTrajectory:
    """Van der Pol should generate cleanly and stay bounded."""

    def test_trajectory_bounded(self):
        vdp = VanDerPolOscillator(mu=1.0)
        x0 = np.array([0.5, 0.0])
        dt = 0.01
        n_steps = 3000

        traj = vdp.generate_trajectory(x0, dt, n_steps)
        assert traj.shape == (n_steps + 1, 2)
        assert np.all(np.isfinite(traj)), "Van der Pol trajectory contains non-finite values"
        # The Van der Pol limit cycle amplitude is roughly 2 for mu=1
        assert np.max(np.abs(traj)) < 10.0, "Van der Pol trajectory blew up"


# ---------------------------------------------------------------------------
# 8. PointVortex — trajectory and state_dim
# ---------------------------------------------------------------------------


class TestPointVortex:
    """PointVortexSystem should generate a trajectory with correct state_dim."""

    def test_state_dim(self):
        pv = PointVortexSystem(n_vortices=4)
        assert pv.state_dim == 8

    def test_trajectory(self):
        pv = PointVortexSystem(n_vortices=3)
        x0 = np.array([0.0, 0.0, 1.0, 0.0, 0.5, 0.866])
        dt = 0.01
        n_steps = 50

        traj = pv.generate_trajectory(x0, dt, n_steps)
        assert traj.shape == (n_steps + 1, 6)
        assert np.all(np.isfinite(traj)), "Point vortex trajectory has non-finite values"

    def test_custom_strengths(self):
        strengths = np.array([1.0, -1.0, 2.0])
        pv = PointVortexSystem(n_vortices=3, strengths=strengths)
        assert pv.state_dim == 6

    def test_invalid_strengths_length(self):
        with pytest.raises(ValueError, match="must match"):
            PointVortexSystem(n_vortices=3, strengths=np.array([1.0, -1.0]))


# ---------------------------------------------------------------------------
# 9. Lorenz attractor — bounded chaotic trajectory
# ---------------------------------------------------------------------------


class TestLorenzAttractor:
    """Lorenz attractor should generate bounded chaotic trajectories."""

    def test_trajectory_bounded(self):
        lorenz = LorenzAttractor()
        x0 = np.array([1.0, 1.0, 1.0])
        dt = 0.01
        n_steps = 5000

        traj = lorenz.generate_trajectory(x0, dt, n_steps)
        assert traj.shape == (n_steps + 1, 3)
        assert np.all(np.isfinite(traj)), "Lorenz trajectory contains non-finite values"
        # Lorenz attractor stays within a bounded region
        assert np.max(np.abs(traj)) < 100.0, "Lorenz trajectory blew up"

    def test_state_dim(self):
        lorenz = LorenzAttractor()
        assert lorenz.state_dim == 3

    def test_custom_parameters(self):
        lorenz = LorenzAttractor(sigma=10.0, rho=15.0, beta=2.0)
        x0 = np.array([1.0, 0.0, 0.0])
        traj = lorenz.generate_trajectory(x0, dt=0.01, n_steps=100)
        assert np.all(np.isfinite(traj))

    def test_sensitive_dependence(self):
        """Two nearby initial conditions should diverge (hallmark of chaos)."""
        lorenz = LorenzAttractor()
        x0a = np.array([1.0, 1.0, 1.0])
        x0b = np.array([1.0, 1.0, 1.0 + 1e-10])
        dt = 0.01
        n_steps = 5000

        traj_a = lorenz.generate_trajectory(x0a, dt, n_steps)
        traj_b = lorenz.generate_trajectory(x0b, dt, n_steps)

        # Initially nearly identical
        assert np.linalg.norm(traj_a[0] - traj_b[0]) < 1e-9
        # Should diverge significantly by the end
        final_diff = np.linalg.norm(traj_a[-1] - traj_b[-1])
        assert final_diff > 1.0, (
            f"Trajectories did not diverge enough: diff = {final_diff:.2e}"
        )


# ---------------------------------------------------------------------------
# 10. Lotka-Volterra — conservation and positivity
# ---------------------------------------------------------------------------


class TestLotkaVolterra:
    """Lotka-Volterra should produce positive, periodic-like trajectories."""

    def test_trajectory_positive(self):
        """Populations should remain positive for reasonable initial conditions."""
        lv = LotkaVolterra()
        x0 = np.array([2.0, 1.0])
        dt = 0.01
        n_steps = 5000

        traj = lv.generate_trajectory(x0, dt, n_steps)
        assert traj.shape == (n_steps + 1, 2)
        assert np.all(np.isfinite(traj)), "Lotka-Volterra trajectory has non-finite values"
        assert np.all(traj > 0), "Populations went negative"

    def test_state_dim(self):
        lv = LotkaVolterra()
        assert lv.state_dim == 2

    def test_conserved_quantity(self):
        """The Lotka-Volterra system conserves H = delta*x - gamma*ln(x) + beta*y - alpha*ln(y)."""
        alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 0.25
        lv = LotkaVolterra(alpha=alpha, beta=beta, gamma=gamma, delta=delta)
        x0 = np.array([2.0, 1.0])
        dt = 0.005
        n_steps = 10000

        traj = lv.generate_trajectory(x0, dt, n_steps)

        def hamiltonian(state):
            x, y = state
            return delta * x - gamma * np.log(x) + beta * y - alpha * np.log(y)

        H = np.array([hamiltonian(traj[i]) for i in range(len(traj))])
        relative_variation = (H.max() - H.min()) / abs(H[0])
        assert relative_variation < 1e-4, (
            f"Conserved quantity not conserved: relative variation = {relative_variation:.2e}"
        )
