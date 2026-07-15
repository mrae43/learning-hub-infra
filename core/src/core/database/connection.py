"""SQLAlchemy engine and session factory."""

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from core.config.settings import settings

engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Yield a transactional database session.

    Commits on clean exit and rolls back on any exception.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
