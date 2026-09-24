import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_db_dir = tempfile.mkdtemp(prefix="davinci_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(_db_dir, 'test.db')}"

import pytest

from app.core.database import Base, engine, SessionLocal, SurgerySession


@pytest.fixture()
def db():
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def surgery_session(db):
    session = SurgerySession(
        session_id="SURG-TEST-0001",
        patient_id="P-001",
        patient_name="测试患者",
        surgery_type="测试手术",
        primary_surgeon="主刀",
        remote_expert="专家",
        operating_room="OR-1",
        status="active",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session
