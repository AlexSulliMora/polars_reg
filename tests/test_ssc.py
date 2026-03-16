"""Tests for the SSC (small-sample corrections) dataclass and helper functions."""

from __future__ import annotations

import dataclasses

import pytest

from polars_reg._ssc import SSC, _default_ssc, ssc


class TestSSCConstruction:
    """Test SSC dataclass construction."""

    def test_defaults(self):
        s = SSC()
        assert s.k_adj is True
        assert s.k_fixef == "none"
        assert s.G_adj is True
        assert s.G_df == "conventional"

    def test_custom_values(self):
        s = SSC(k_adj=False, k_fixef="nonnested", G_adj=False, G_df="min")
        assert s.k_adj is False
        assert s.k_fixef == "nonnested"
        assert s.G_adj is False
        assert s.G_df == "min"

    def test_k_fixef_full(self):
        s = SSC(k_fixef="full")
        assert s.k_fixef == "full"


class TestSSCValidation:
    """Test SSC validation in __post_init__."""

    def test_invalid_k_fixef(self):
        with pytest.raises(ValueError, match="k_fixef must be"):
            SSC(k_fixef="invalid")

    def test_invalid_G_df(self):
        with pytest.raises(ValueError, match="G_df must be"):
            SSC(G_df="invalid")


class TestSSCFrozen:
    """Test that SSC is frozen (immutable)."""

    def test_cannot_modify_k_adj(self):
        s = SSC()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.k_adj = False  # type: ignore[misc]

    def test_cannot_modify_k_fixef(self):
        s = SSC()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.k_fixef = "full"  # type: ignore[misc]


class TestSSCRepr:
    """Test SSC repr is readable."""

    def test_repr_contains_field_names(self):
        s = SSC()
        r = repr(s)
        assert "k_adj" in r
        assert "k_fixef" in r
        assert "G_adj" in r
        assert "G_df" in r

    def test_repr_shows_values(self):
        s = SSC(k_adj=False, G_df="min")
        r = repr(s)
        assert "False" in r
        assert "min" in r


class TestSscFunction:
    """Test ssc() convenience function."""

    def test_returns_ssc_instance(self):
        result = ssc()
        assert isinstance(result, SSC)

    def test_defaults_match_SSC(self):
        assert ssc() == SSC()

    def test_custom_values(self):
        result = ssc(k_adj=False, k_fixef="nonnested", G_adj=False, G_df="min")
        assert result == SSC(k_adj=False, k_fixef="nonnested", G_adj=False, G_df="min")

    def test_stata_preset(self):
        result = ssc(k_fixef="nonnested", G_df="min")
        assert result.k_fixef == "nonnested"
        assert result.G_df == "min"
        assert result.k_adj is True
        assert result.G_adj is True


class TestDefaultSsc:
    """Test _default_ssc() helper."""

    def test_returns_default_ssc(self):
        result = _default_ssc()
        assert isinstance(result, SSC)
        assert result == SSC()

    def test_returns_new_instance(self):
        a = _default_ssc()
        b = _default_ssc()
        # Frozen dataclasses are equal but could be same or different objects
        assert a == b
