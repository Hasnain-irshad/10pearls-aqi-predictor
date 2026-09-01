"""Tests for the retrying feature-store read.

The free-tier Arrow Flight service intermittently drops a large read with
``FlightUnavailableError: Socket closed`` (gRPC status 14, UNAVAILABLE), which
failed 5 of 10 nightly training runs. These tests pin the retry behaviour using
a fake feature view, so no Hopsworks connection is needed.
"""
import pandas as pd
import pytest

from aqi.data import hopsworks_store as hs


class FlightUnavailable(RuntimeError):
    """Stands in for pyarrow._flight.FlightUnavailableError."""


class FakeFeatureView:
    """Fails `fail_times` times, then returns a frame."""

    def __init__(self, fail_times=0, rows=3):
        self.fail_times = fail_times
        self.calls = 0
        self._rows = rows

    def training_data(self, description=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise FlightUnavailable("Flight returned unavailable error: Socket closed")
        df = pd.DataFrame({
            "city": ["Lahore"] * self._rows,
            "datetime": ["2026-08-31 03:00:00", "2026-08-31 01:00:00", "2026-08-31 02:00:00"][: self._rows],
            "aqi": [150, 140, 145][: self._rows],
        })
        return df, None


@pytest.fixture
def fake_fv(monkeypatch):
    """Install a fake feature view and make sleeping instant."""
    slept = []
    monkeypatch.setattr(hs.time, "sleep", lambda s: slept.append(s))

    def install(fv):
        monkeypatch.setattr(hs, "get_feature_view", lambda project: fv)
        return slept

    return install


def test_succeeds_first_time_without_sleeping(fake_fv):
    fv = FakeFeatureView(fail_times=0)
    slept = fake_fv(fv)
    df = hs.read_features(project=None)

    assert fv.calls == 1
    assert slept == []
    assert len(df) == 3


def test_retries_a_transient_flight_failure_and_recovers(fake_fv):
    """The real failure mode: two bad attempts, then the read goes through."""
    fv = FakeFeatureView(fail_times=2)
    slept = fake_fv(fv)
    df = hs.read_features(project=None, attempts=4, backoff=30)

    assert fv.calls == 3                 # failed twice, succeeded on the third
    assert slept == [30, 60]             # backoff grows with the attempt number
    assert len(df) == 3


def test_gives_up_after_the_configured_attempts_and_reraises(fake_fv):
    fv = FakeFeatureView(fail_times=99)
    slept = fake_fv(fv)

    with pytest.raises(FlightUnavailable, match="Socket closed"):
        hs.read_features(project=None, attempts=3, backoff=5)

    assert fv.calls == 3                 # exactly the budget, no more
    assert slept == [5, 10]              # no sleep after the final attempt


def test_result_is_normalised(fake_fv):
    """Event time becomes real datetimes and rows come back in time order."""
    fv = FakeFeatureView(fail_times=1)
    fake_fv(fv)
    df = hs.read_features(project=None, attempts=3, backoff=0)

    assert pd.api.types.is_datetime64_any_dtype(df["datetime"])
    assert df["datetime"].is_monotonic_increasing
    assert list(df["aqi"]) == [140, 145, 150]


def test_attempts_and_backoff_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("AQI_FS_READ_ATTEMPTS", "7")
    monkeypatch.setenv("AQI_FS_READ_BACKOFF", "15")
    import importlib

    reloaded = importlib.reload(hs)
    try:
        assert reloaded.READ_ATTEMPTS == 7
        assert reloaded.READ_BACKOFF == 15.0
    finally:
        monkeypatch.delenv("AQI_FS_READ_ATTEMPTS")
        monkeypatch.delenv("AQI_FS_READ_BACKOFF")
        importlib.reload(hs)
