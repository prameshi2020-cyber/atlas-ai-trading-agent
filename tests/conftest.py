import hashlib
from pathlib import Path

import pytest

from app.database.models import (
    initialize_database,
    initialize_candle_database,
    initialize_position_database,
)
from app.trading.journal import initialize_trade_journal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DATABASE_PATH = PROJECT_ROOT / "data" / "atlas.db"


def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


@pytest.fixture(scope="session", autouse=True)
def protect_production_database():
    """
    Ensure pytest never modifies the production SQLite database.

    The production database is snapshotted before the test session
    and verified unchanged after the entire test suite finishes.
    """
    before_hash = _file_hash(PRODUCTION_DATABASE_PATH)

    yield

    after_hash = _file_hash(PRODUCTION_DATABASE_PATH)

    assert after_hash == before_hash, (
        "PRODUCTION DATABASE WAS MODIFIED DURING TESTS!\n"
        f"Production database: {PRODUCTION_DATABASE_PATH}\n"
        "Tests must use the isolated pytest database."
    )


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    """
    Give every test its own isolated SQLite database.

    Pytest must never read from or write to the production
    data/atlas.db database.
    """
    database_path = tmp_path / "test_atlas.db"

    monkeypatch.setattr(
        "app.database.connection.DATABASE_PATH",
        database_path,
    )

    monkeypatch.setattr(
        "app.database.connection.DATA_DIR",
        tmp_path,
    )

    # Create the complete test database schema.
    initialize_database()
    initialize_candle_database()
    initialize_position_database()
    initialize_trade_journal()

    yield database_path