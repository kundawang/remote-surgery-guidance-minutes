import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

# whisper 依赖 torch，测试环境用 stub 代替（后台任务中转写方法会被 mock）
sys.modules.setdefault("whisper", types.ModuleType("whisper"))

TEST_DB_PATH = Path(tempfile.mkdtemp(prefix="surgery_test_")) / "test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.api import surgery, summary, transcript  # noqa: E402
from app.core.database import Base, engine  # noqa: E402

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.include_router(surgery.router, prefix="/api/surgery")
app.include_router(transcript.router, prefix="/api/transcript")
app.include_router(summary.router, prefix="/api/summary")


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def session_id(client):
    response = client.post("/api/surgery", json={
        "patient_id": "P-001",
        "patient_name": "测试患者",
        "surgery_type": "腹腔镜胆囊切除术",
        "primary_surgeon": "张医生",
        "remote_expert": "李专家",
        "operating_room": "OR-1",
    })
    assert response.status_code == 200
    return response.json()["session_id"]
