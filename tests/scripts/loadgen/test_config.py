"""Tests for scripts/loadgen/config.py."""

from __future__ import annotations

import pytest

from scripts.loadgen.config import (
    DEFAULT_SPAWN_RATE,
    DEFAULT_USERS,
    MAX_ERROR_RATE,
    P95_CEILINGS_MS,
    SMOKE_UPSTREAM_REQUEST_CAP,
    LoadProfile,
    load_config,
)


class TestLoadConfig:
    def test_defaults_to_volume(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOAD_PROFILE", raising=False)
        assert load_config() is LoadProfile.VOLUME

    def test_volume_profile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOAD_PROFILE", "volume")
        assert load_config() is LoadProfile.VOLUME

    def test_smoke_profile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOAD_PROFILE", "smoke")
        assert load_config() is LoadProfile.SMOKE

    def test_profile_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOAD_PROFILE", "SMOKE")
        assert load_config() is LoadProfile.SMOKE

    def test_unknown_profile_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOAD_PROFILE", "gazebo")
        with pytest.raises(ValueError, match="LOAD_PROFILE"):
            load_config()


class TestProfileConstants:
    def test_smoke_has_no_ramp(self) -> None:
        assert DEFAULT_SPAWN_RATE[LoadProfile.SMOKE] == 1

    def test_smoke_is_one_upstream_user_plus_liveness(self) -> None:
        assert DEFAULT_USERS[LoadProfile.SMOKE] == 2

    def test_volume_has_sustained_pool(self) -> None:
        assert DEFAULT_USERS[LoadProfile.VOLUME] > 1

    def test_p95_ceilings_smoke_looser_than_volume(self) -> None:
        for endpoint in P95_CEILINGS_MS[LoadProfile.VOLUME]:
            assert (
                P95_CEILINGS_MS[LoadProfile.SMOKE][endpoint]
                >= P95_CEILINGS_MS[LoadProfile.VOLUME][endpoint]
            )

    def test_all_endpoints_have_ceilings_in_both_profiles(self) -> None:
        assert set(P95_CEILINGS_MS[LoadProfile.VOLUME]) == {"/health", "/query", "/dive"}
        assert set(P95_CEILINGS_MS[LoadProfile.SMOKE]) == {"/health", "/query", "/dive"}

    def test_smoke_cap_is_bounded(self) -> None:
        assert SMOKE_UPSTREAM_REQUEST_CAP <= 50

    def test_error_rate_ceiling_is_one_percent(self) -> None:
        assert MAX_ERROR_RATE == 0.01
