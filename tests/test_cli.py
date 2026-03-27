"""Tests for the KoopSim CLI (Phase 10)."""

from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner

from koopsim.cli import main


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# 1. Help
# ---------------------------------------------------------------------------


class TestHelp:
    def test_help_exits_zero(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "KoopSim" in result.output

    def test_train_help(self, runner):
        result = runner.invoke(main, ["train", "--help"])
        assert result.exit_code == 0
        assert "--data" in result.output

    def test_generate_help(self, runner):
        result = runner.invoke(main, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--system" in result.output

    def test_predict_help(self, runner):
        result = runner.invoke(main, ["predict", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output


# ---------------------------------------------------------------------------
# 2. Generate
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_generate_hopf_h5(self, runner, tmp_path):
        out = str(tmp_path / "data.h5")
        result = runner.invoke(
            main,
            [
                "generate",
                "--system",
                "hopf",
                "-o",
                out,
                "--dt",
                "0.01",
                "--n-steps",
                "50",
                "--n-trajectories",
                "5",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Generated" in result.output
        assert "snapshot pairs" in result.output

        import h5py

        with h5py.File(out, "r") as f:
            assert "X" in f
            assert "Y" in f
            assert f["X"].shape[0] == f["Y"].shape[0]
            assert f["X"].shape[1] == 2  # Hopf state_dim = 2

    def test_generate_spring_mass(self, runner, tmp_path):
        out = str(tmp_path / "data.h5")
        result = runner.invoke(
            main,
            [
                "generate",
                "--system",
                "spring-mass",
                "-o",
                out,
                "--dt",
                "0.01",
                "--n-steps",
                "30",
                "--n-trajectories",
                "3",
            ],
        )
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------


class TestTrain:
    def test_train_edmd_from_h5(self, runner, tmp_path):
        data_path = str(tmp_path / "data.h5")
        model_path = str(tmp_path / "model.koop")

        # Generate data first
        result = runner.invoke(
            main,
            [
                "generate",
                "--system",
                "hopf",
                "-o",
                data_path,
                "--dt",
                "0.01",
                "--n-steps",
                "50",
                "--n-trajectories",
                "5",
            ],
        )
        assert result.exit_code == 0, result.output

        # Train
        result = runner.invoke(
            main,
            ["train", "--data", data_path, "--method", "edmd", "--dt", "0.01", "-o", model_path],
        )
        assert result.exit_code == 0, result.output
        assert "Model saved" in result.output

        from pathlib import Path

        assert Path(model_path).exists()


# ---------------------------------------------------------------------------
# 4. Predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_outputs_state(self, runner, tmp_path):
        data_path = str(tmp_path / "data.h5")
        model_path = str(tmp_path / "model.koop")

        # Generate + Train
        runner.invoke(
            main,
            [
                "generate",
                "--system",
                "hopf",
                "-o",
                data_path,
                "--dt",
                "0.01",
                "--n-steps",
                "50",
                "--n-trajectories",
                "5",
            ],
        )
        runner.invoke(
            main,
            ["train", "--data", data_path, "--method", "edmd", "--dt", "0.01", "-o", model_path],
        )

        # Predict
        result = runner.invoke(
            main,
            ["predict", "--model", model_path, "--initial-state", "1.0,0.0", "--time", "0.1"],
        )
        assert result.exit_code == 0, result.output
        assert "State at t=0.1" in result.output


# ---------------------------------------------------------------------------
# 5. Info
# ---------------------------------------------------------------------------


class TestInfo:
    def test_info_displays_metadata(self, runner, tmp_path):
        data_path = str(tmp_path / "data.h5")
        model_path = str(tmp_path / "model.koop")

        # Generate + Train
        runner.invoke(
            main,
            [
                "generate",
                "--system",
                "hopf",
                "-o",
                data_path,
                "--dt",
                "0.01",
                "--n-steps",
                "50",
                "--n-trajectories",
                "5",
            ],
        )
        runner.invoke(
            main,
            ["train", "--data", data_path, "--method", "edmd", "--dt", "0.01", "-o", model_path],
        )

        # Info
        result = runner.invoke(main, ["info", "--model", model_path])
        assert result.exit_code == 0, result.output
        assert "model_type" in result.output
        assert "n_state_dims" in result.output
        assert "n_eigenvalues" in result.output
        assert "max |lambda|" in result.output


# ---------------------------------------------------------------------------
# 6. Full pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_generate_train_predict(self, runner, tmp_path):
        data_path = str(tmp_path / "data.h5")
        model_path = str(tmp_path / "model.koop")

        # Generate
        result = runner.invoke(
            main,
            [
                "generate",
                "--system",
                "rlc",
                "-o",
                data_path,
                "--dt",
                "0.01",
                "--n-steps",
                "100",
                "--n-trajectories",
                "5",
            ],
        )
        assert result.exit_code == 0, result.output

        # Train
        result = runner.invoke(
            main,
            ["train", "--data", data_path, "--method", "edmd", "--dt", "0.01", "-o", model_path],
        )
        assert result.exit_code == 0, result.output

        # Predict
        result = runner.invoke(
            main,
            ["predict", "--model", model_path, "--initial-state", "0.5,0.0", "--time", "0.5"],
        )
        assert result.exit_code == 0, result.output
        assert "State at t=0.5" in result.output

        # Info
        result = runner.invoke(main, ["info", "--model", model_path])
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# 7. CSV format
# ---------------------------------------------------------------------------


class TestCSVFormat:
    def test_generate_csv_then_train(self, runner, tmp_path):
        csv_path = str(tmp_path / "data.csv")
        model_path = str(tmp_path / "model.koop")

        # Generate as CSV
        result = runner.invoke(
            main,
            [
                "generate",
                "--system",
                "hopf",
                "-o",
                csv_path,
                "--dt",
                "0.01",
                "--n-steps",
                "50",
                "--n-trajectories",
                "5",
            ],
        )
        assert result.exit_code == 0, result.output

        # Verify CSV file is readable
        from pathlib import Path

        assert Path(csv_path).exists()
        data = np.loadtxt(csv_path, delimiter=",")
        assert data.ndim == 2
        assert data.shape[1] == 2  # Hopf is 2D

        # Train from CSV
        result = runner.invoke(
            main,
            ["train", "--data", csv_path, "--method", "edmd", "--dt", "0.01", "-o", model_path],
        )
        assert result.exit_code == 0, result.output
        assert "Model saved" in result.output


# ---------------------------------------------------------------------------
# 8. NPY format
# ---------------------------------------------------------------------------


class TestNPYFormat:
    def test_generate_npy_then_train(self, runner, tmp_path):
        npy_path = str(tmp_path / "data.npy")
        model_path = str(tmp_path / "model.koop")

        # Generate as NPY
        result = runner.invoke(
            main,
            [
                "generate",
                "--system",
                "hopf",
                "-o",
                npy_path,
                "--dt",
                "0.01",
                "--n-steps",
                "50",
                "--n-trajectories",
                "5",
            ],
        )
        assert result.exit_code == 0, result.output

        # Verify NPY file is loadable
        from pathlib import Path

        assert Path(npy_path).exists()
        loaded = np.load(npy_path, allow_pickle=True).item()
        assert "X" in loaded
        assert "Y" in loaded

        # Train from NPY
        result = runner.invoke(
            main,
            ["train", "--data", npy_path, "--method", "edmd", "--dt", "0.01", "-o", model_path],
        )
        assert result.exit_code == 0, result.output
        assert "Model saved" in result.output


# ---------------------------------------------------------------------------
# 9. Validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_validate_reports_metric(self, runner, tmp_path):
        train_path = str(tmp_path / "train.h5")
        test_path = str(tmp_path / "test.h5")
        model_path = str(tmp_path / "model.koop")

        # Generate training data
        result = runner.invoke(
            main,
            [
                "generate",
                "--system",
                "hopf",
                "-o",
                train_path,
                "--dt",
                "0.01",
                "--n-steps",
                "50",
                "--n-trajectories",
                "5",
            ],
        )
        assert result.exit_code == 0, result.output

        # Generate separate test data
        result = runner.invoke(
            main,
            [
                "generate",
                "--system",
                "hopf",
                "-o",
                test_path,
                "--dt",
                "0.01",
                "--n-steps",
                "30",
                "--n-trajectories",
                "3",
            ],
        )
        assert result.exit_code == 0, result.output

        # Train model
        result = runner.invoke(
            main,
            ["train", "--data", train_path, "--method", "edmd", "--dt", "0.01", "-o", model_path],
        )
        assert result.exit_code == 0, result.output

        # Validate
        result = runner.invoke(
            main,
            ["validate", "--model", model_path, "--test-data", test_path, "--dt", "0.01"],
        )
        assert result.exit_code == 0, result.output
        assert "Validation rmse:" in result.output

    def test_validate_mae_metric(self, runner, tmp_path):
        data_path = str(tmp_path / "data.h5")
        model_path = str(tmp_path / "model.koop")

        # Generate + Train
        runner.invoke(
            main,
            [
                "generate",
                "--system",
                "hopf",
                "-o",
                data_path,
                "--dt",
                "0.01",
                "--n-steps",
                "50",
                "--n-trajectories",
                "5",
            ],
        )
        runner.invoke(
            main,
            ["train", "--data", data_path, "--method", "edmd", "--dt", "0.01", "-o", model_path],
        )

        # Validate with MAE
        result = runner.invoke(
            main,
            [
                "validate",
                "--model",
                model_path,
                "--test-data",
                data_path,
                "--dt",
                "0.01",
                "--metric",
                "mae",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Validation mae:" in result.output
