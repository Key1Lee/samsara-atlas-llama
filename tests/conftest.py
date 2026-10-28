import os
import tempfile
import atexit

import pytest

_test_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_test_db.close()
os.environ["DATABASE_URL"] = f"sqlite:///{_test_db.name}"
os.environ["PROVIDER_TYPE"] = "stub"
atexit.register(lambda: os.path.exists(_test_db.name) and os.remove(_test_db.name))
os.environ["DOCS_DIR"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "docs")


@pytest.fixture
def db_session():
    # Import models before init_db() so SQLAlchemy's declarative metadata
    # actually knows about every table before create_all() runs - otherwise
    # whichever test happens to touch the DB first (before anything else in
    # the process has imported app.storage.models) creates an incomplete
    # schema.
    from app.storage import models  # noqa: F401
    from app.storage.database import init_db, SessionLocal
    init_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def deterministic_stub(monkeypatch):
    from app.providers.stub import StubProvider
    monkeypatch.setattr("app.engine.steps.get_provider", lambda: StubProvider(timeout_chance=0))


@pytest.fixture(autouse=True)
def isolated_daily_quota(db_session, monkeypatch):
    # Each test starts with its own quota budget, independent of test order.
    from app.storage.models import EmployeeDailyRequests
    # Ordinary synthetic regression tests explicitly opt into the demo profile.
    monkeypatch.setenv("ATLAS_ENV", "development")
    monkeypatch.delenv("ATLAS_EMPLOYEE_TOKEN_HASHES", raising=False)
    db_session.query(EmployeeDailyRequests).delete()
    db_session.commit()
