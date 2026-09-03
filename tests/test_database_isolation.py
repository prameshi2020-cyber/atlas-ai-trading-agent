from pathlib import Path

from app.database import connection


def test_tests_use_isolated_database():
    production_database = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "atlas.db"
    )

    assert connection.DATABASE_PATH != production_database
    assert connection.DATABASE_PATH.exists()
    assert connection.DATABASE_PATH != production_database