"""
Tests for hypotestx.math.distributions — Normal, StudentT, ChiSquare, F.
"""

import math

import pytest

from hypotestx.math.distributions import ChiSquare, F, Normal, StudentT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def approx(a, b, tol=1e-4):
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# Normal distribution
# ---------------------------------------------------------------------------


class TestNormal:
    def test_pdf_at_mean(self):
        n = Normal(0, 1)
        assert approx(n.pdf(0), 1 / math.sqrt(2 * math.pi))

    def test_pdf_symmetric(self):
        n = Normal(0, 1)
        assert approx(n.pdf(1), n.pdf(-1))

    def test_pdf_nonzero_mean(self):
        n = Normal(5, 2)
        assert n.pdf(5) > n.pdf(3)

    def test_cdf_at_mean(self):
        n = Normal(0, 1)
        assert approx(n.cdf(0), 0.5)

    def test_cdf_upper_tail(self):
        n = Normal(0, 1)
        assert approx(n.cdf(1.96), 0.975, tol=5e-3)

    def test_cdf_lower_tail(self):
        n = Normal(0, 1)
        assert approx(n.cdf(-1.96), 0.025, tol=5e-3)

    def test_cdf_monotone(self):
        n = Normal(0, 1)
        vals = [n.cdf(x) for x in [-3, -1, 0, 1, 3]]
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_ppf_roundtrip(self):
        n = Normal(0, 1)
        for p in [0.025, 0.1, 0.5, 0.9, 0.975]:
            assert approx(n.cdf(n.ppf(p)), p, tol=1e-5)

    def test_ppf_at_half(self):
        n = Normal(0, 1)
        assert approx(n.ppf(0.5), 0.0)

    def test_ppf_1_96(self):
        n = Normal(0, 1)
        assert approx(n.ppf(0.975), 1.96, tol=1e-2)

    def test_different_params(self):
        n = Normal(10, 3)
        assert approx(n.cdf(10), 0.5)
        assert approx(n.ppf(0.5), 10.0)


# ---------------------------------------------------------------------------
# StudentT distribution
# ---------------------------------------------------------------------------


class TestStudentT:
    def test_pdf_symmetric(self):
        t = StudentT(5)
        assert approx(t.pdf(1), t.pdf(-1))

    def test_pdf_peak_at_zero(self):
        t = StudentT(5)
        assert t.pdf(0) > t.pdf(1)
        assert t.pdf(0) > t.pdf(-1)

    def test_cdf_at_zero(self):
        t = StudentT(10)
        assert approx(t.cdf(0), 0.5)

    def test_cdf_monotone(self):
        t = StudentT(10)
        vals = [t.cdf(x) for x in [-3, -1, 0, 1, 3]]
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_ppf_roundtrip(self):
        t = StudentT(10)
        for p in [0.025, 0.1, 0.5, 0.9, 0.975]:
            assert approx(t.cdf(t.ppf(p)), p, tol=1e-4)

    def test_large_df_approaches_normal(self):
        """t(1000) should be very close to N(0,1) at p=0.975."""
        t = StudentT(1000)
        n = Normal(0, 1)
        assert approx(t.ppf(0.975), n.ppf(0.975), tol=0.01)

    def test_known_critical_value_df10(self):
        t = StudentT(10)
        # Two-tailed 5% critical value for df=10 is ~2.228
        assert approx(t.ppf(0.975), 2.228, tol=0.01)


# ---------------------------------------------------------------------------
# ChiSquare distribution
# ---------------------------------------------------------------------------


class TestChiSquare:
    def test_pdf_positive_only(self):
        c = ChiSquare(3)
        assert c.pdf(0) == 0.0 or math.isnan(c.pdf(0)) or c.pdf(1e-10) > 0

    def test_cdf_nonnegative(self):
        c = ChiSquare(5)
        assert c.cdf(0) == pytest.approx(0.0, abs=1e-6)

    def test_cdf_approaches_1(self):
        c = ChiSquare(5)
        assert c.cdf(100) > 0.9999

    def test_cdf_monotone(self):
        c = ChiSquare(5)
        vals = [c.cdf(x) for x in [0.5, 1, 2, 5, 10, 20]]
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_critical_value_df1(self):
        # chi2(1, p=0.95) ~ 3.841
        c = ChiSquare(1)
        assert approx(c.ppf(0.95), 3.841, tol=0.05)

    def test_ppf_roundtrip(self):
        c = ChiSquare(5)
        for p in [0.1, 0.5, 0.9, 0.95]:
            assert approx(c.cdf(c.ppf(p)), p, tol=1e-4)


# ---------------------------------------------------------------------------
# F distribution
# ---------------------------------------------------------------------------


class TestFDist:
    def test_pdf_positive(self):
        f = F(3, 10)
        assert f.pdf(1) > 0

    def test_cdf_at_0(self):
        f = F(3, 10)
        assert f.cdf(0) == pytest.approx(0.0, abs=1e-6)

    def test_cdf_approaches_1(self):
        f = F(3, 10)
        assert f.cdf(1000) > 0.9999

    def test_cdf_monotone(self):
        f = F(3, 10)
        vals = [f.cdf(x) for x in [0.1, 0.5, 1, 2, 5, 10]]
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_ppf_roundtrip(self):
        f = F(3, 10)
        for p in [0.1, 0.5, 0.9, 0.95]:
            assert approx(f.cdf(f.ppf(p)), p, tol=1e-4)

    def test_known_critical_value(self):
        # F(1,1, p=0.95) ~ 161.4 — skip, focus on more stable values
        f = F(2, 20)
        # F(2,20, p=0.95) ~ 3.49
        assert approx(f.ppf(0.95), 3.49, tol=0.1)
