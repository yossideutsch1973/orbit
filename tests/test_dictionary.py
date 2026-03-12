"""Tests for observable dictionary functions."""

from __future__ import annotations

import numpy as np
import pytest

from koopsim.core.exceptions import DimensionMismatchError, NotFittedError
from koopsim.utils.dictionary import (
    CompositeDictionary,
    IdentityDictionary,
    PolynomialDictionary,
    RBFDictionary,
)


# ---------------------------------------------------------------------------
# IdentityDictionary
# ---------------------------------------------------------------------------


class TestIdentityDictionary:
    """Tests for IdentityDictionary."""

    def test_shape(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((50, 4))
        d = IdentityDictionary().fit(X)
        out = d.transform(X)
        assert out.shape == X.shape

    def test_passthrough(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((30, 3))
        d = IdentityDictionary().fit(X)
        np.testing.assert_array_equal(d.transform(X), X)

    def test_n_output_features(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((10, 7))
        d = IdentityDictionary().fit(X)
        assert d.n_output_features == 7

    def test_not_fitted_transform_raises(self) -> None:
        d = IdentityDictionary()
        with pytest.raises(NotFittedError):
            d.transform(np.zeros((5, 3)))

    def test_not_fitted_n_output_features_raises(self) -> None:
        d = IdentityDictionary()
        with pytest.raises(NotFittedError):
            _ = d.n_output_features

    def test_dimension_mismatch(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((20, 3))
        d = IdentityDictionary().fit(X)
        with pytest.raises(DimensionMismatchError):
            d.transform(rng.standard_normal((10, 5)))

    def test_fit_transform(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((20, 4))
        d = IdentityDictionary()
        out = d.fit_transform(X)
        np.testing.assert_array_equal(out, X)


# ---------------------------------------------------------------------------
# PolynomialDictionary
# ---------------------------------------------------------------------------


class TestPolynomialDictionary:
    """Tests for PolynomialDictionary."""

    def test_excludes_constant_and_linear(self, rng: np.random.Generator) -> None:
        """Output should only contain degree >= 2 terms."""
        X = rng.standard_normal((40, 3))
        d = PolynomialDictionary(degree=2).fit(X)
        out = d.transform(X)
        # For 3 features, degree-2 terms: x1^2, x1*x2, x1*x3, x2^2, x2*x3, x3^2 = 6
        assert out.shape == (40, 6)

    def test_correct_values_degree2(self) -> None:
        """Verify polynomial values for a known simple input."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        d = PolynomialDictionary(degree=2).fit(X)
        out = d.transform(X)
        # Degree-2 terms for 2 features: x1^2, x1*x2, x2^2
        expected = np.array([
            [1.0, 2.0, 4.0],    # 1^2, 1*2, 2^2
            [9.0, 12.0, 16.0],  # 3^2, 3*4, 4^2
        ])
        np.testing.assert_allclose(out, expected)

    def test_correct_values_degree3(self) -> None:
        """Verify polynomial values for degree 3 with known input."""
        X = np.array([[1.0, 2.0]])
        d = PolynomialDictionary(degree=3).fit(X)
        out = d.transform(X)
        # Degree >=2 terms for 2 features up to degree 3:
        # degree 2: x1^2(1), x1*x2(2), x2^2(4)
        # degree 3: x1^3(1), x1^2*x2(2), x1*x2^2(4), x2^3(8)
        expected = np.array([[1.0, 2.0, 4.0, 1.0, 2.0, 4.0, 8.0]])
        np.testing.assert_allclose(out, expected)

    def test_n_output_features(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((30, 4))
        d = PolynomialDictionary(degree=3).fit(X)
        out = d.transform(X)
        assert out.shape[1] == d.n_output_features

    def test_degree_must_be_at_least_2(self) -> None:
        with pytest.raises(ValueError, match="degree must be >= 2"):
            PolynomialDictionary(degree=1)

    def test_not_fitted_raises(self) -> None:
        d = PolynomialDictionary(degree=2)
        with pytest.raises(NotFittedError):
            d.transform(np.zeros((5, 3)))


# ---------------------------------------------------------------------------
# RBFDictionary
# ---------------------------------------------------------------------------


class TestRBFDictionary:
    """Tests for RBFDictionary."""

    def test_shape(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((100, 3))
        d = RBFDictionary(n_centers=20).fit(X)
        out = d.transform(X)
        assert out.shape == (100, 20)

    def test_no_inf_nan(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((200, 5))
        d = RBFDictionary(n_centers=30).fit(X)
        out = d.transform(X)
        assert np.all(np.isfinite(out))

    def test_gamma_auto_heuristic(self, rng: np.random.Generator) -> None:
        """Auto gamma should produce a positive finite value."""
        X = rng.standard_normal((100, 4))
        d = RBFDictionary(n_centers=10, gamma="auto").fit(X)
        assert d._gamma is not None
        assert d._gamma > 0
        assert np.isfinite(d._gamma)

    def test_gamma_manual(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((50, 2))
        d = RBFDictionary(n_centers=5, gamma=0.5).fit(X)
        assert d._gamma == 0.5

    def test_n_output_features(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((80, 3))
        d = RBFDictionary(n_centers=15).fit(X)
        assert d.n_output_features == 15

    def test_output_in_zero_one(self, rng: np.random.Generator) -> None:
        """Gaussian RBF outputs should be in (0, 1]."""
        X = rng.standard_normal((100, 3))
        d = RBFDictionary(n_centers=20).fit(X)
        out = d.transform(X)
        assert np.all(out > 0)
        assert np.all(out <= 1.0 + 1e-10)

    def test_not_fitted_raises(self) -> None:
        d = RBFDictionary(n_centers=10)
        with pytest.raises(NotFittedError):
            d.transform(np.zeros((5, 3)))

    def test_n_centers_capped_by_samples(self, rng: np.random.Generator) -> None:
        """If n_centers > n_samples, should still work (capped)."""
        X = rng.standard_normal((5, 2))
        d = RBFDictionary(n_centers=100).fit(X)
        assert d.n_output_features <= 5

    def test_unsupported_kernel(self) -> None:
        with pytest.raises(ValueError, match="Unsupported kernel"):
            RBFDictionary(kernel="laplacian")


# ---------------------------------------------------------------------------
# CompositeDictionary
# ---------------------------------------------------------------------------


class TestCompositeDictionary:
    """Tests for CompositeDictionary."""

    def test_hstack_shape(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((50, 3))
        poly = PolynomialDictionary(degree=2)
        rbf = RBFDictionary(n_centers=10)
        comp = CompositeDictionary([poly, rbf]).fit(X)
        out = comp.transform(X)
        # identity(3) + poly_deg2(6) + rbf(10) = 19
        assert out.shape == (50, 19)

    def test_first_columns_are_identity(self, rng: np.random.Generator) -> None:
        """First n_state columns must be raw state."""
        X = rng.standard_normal((40, 4))
        poly = PolynomialDictionary(degree=2)
        comp = CompositeDictionary([poly]).fit(X)
        out = comp.transform(X)
        np.testing.assert_array_equal(out[:, :4], X)

    def test_n_output_features_sum(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((60, 3))
        poly = PolynomialDictionary(degree=2)
        rbf = RBFDictionary(n_centers=8)
        comp = CompositeDictionary([poly, rbf]).fit(X)
        # identity + poly + rbf
        identity_n = 3
        poly_n = poly.n_output_features
        rbf_n = rbf.n_output_features
        assert comp.n_output_features == identity_n + poly_n + rbf_n

    def test_identity_not_duplicated(self, rng: np.random.Generator) -> None:
        """Passing IdentityDictionary should not create duplicate identity columns."""
        X = rng.standard_normal((30, 2))
        comp = CompositeDictionary([IdentityDictionary()]).fit(X)
        out = comp.transform(X)
        # Should be just identity (2 features), not doubled
        assert out.shape == (30, 2)
        np.testing.assert_array_equal(out, X)

    def test_n_state_property(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((20, 5))
        comp = CompositeDictionary([PolynomialDictionary(degree=2)]).fit(X)
        assert comp.n_state == 5

    def test_not_fitted_raises(self) -> None:
        comp = CompositeDictionary([PolynomialDictionary(degree=2)])
        with pytest.raises(NotFittedError):
            comp.transform(np.zeros((5, 3)))

    def test_empty_extra_dictionaries(self, rng: np.random.Generator) -> None:
        """Composite with no extra dictionaries is just identity."""
        X = rng.standard_normal((20, 3))
        comp = CompositeDictionary([]).fit(X)
        out = comp.transform(X)
        assert out.shape == (20, 3)
        np.testing.assert_array_equal(out, X)


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Tests for numerical stability on large random data."""

    def test_large_data_no_inf_nan(self, rng: np.random.Generator) -> None:
        """All dictionary types should produce finite output on large data."""
        X = rng.standard_normal((1000, 10))

        identity = IdentityDictionary().fit(X)
        assert np.all(np.isfinite(identity.transform(X)))

        poly = PolynomialDictionary(degree=3).fit(X)
        assert np.all(np.isfinite(poly.transform(X)))

        rbf = RBFDictionary(n_centers=50).fit(X)
        assert np.all(np.isfinite(rbf.transform(X)))

        comp = CompositeDictionary([
            PolynomialDictionary(degree=2),
            RBFDictionary(n_centers=30),
        ]).fit(X)
        assert np.all(np.isfinite(comp.transform(X)))

    def test_large_values_rbf_stable(self, rng: np.random.Generator) -> None:
        """RBF should not produce inf/nan even with large input values."""
        X = rng.standard_normal((100, 5)) * 1000.0
        d = RBFDictionary(n_centers=20).fit(X)
        out = d.transform(X)
        assert np.all(np.isfinite(out))
