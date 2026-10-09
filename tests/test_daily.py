"""Daily refresh only asks for seasons that overlap the last successful run."""

from datetime import date, datetime, timezone

import pytest

from etl.daily import plan_refresh, seasons_between


def test_april_refresh_stays_in_one_season() -> None:
    seasons, date_from = plan_refresh(datetime(2026, 4, 15, 8, 0), date(2026, 4, 16))
    assert seasons == ["2025-26"]
    assert date_from == "04/15/2026"


def test_june_through_october_includes_the_new_season() -> None:
    seasons, date_from = plan_refresh(datetime(2026, 6, 20, 8, 0), date(2026, 10, 8))
    assert seasons == ["2025-26", "2026-27"]
    assert date_from == "06/20/2026"
    assert "1996-97" not in seasons


def test_utc_timestamp_ahead_of_the_local_date_still_refreshes() -> None:
    seasons, date_from = plan_refresh(
        datetime(2026, 10, 9, 2, 0, tzinfo=timezone.utc),
        date(2026, 10, 8),
    )
    assert seasons == ["2026-27"]
    assert date_from == "10/09/2026"


def test_reversed_range_is_rejected() -> None:
    with pytest.raises(ValueError):
        seasons_between(date(2026, 10, 8), date(2026, 6, 20))
