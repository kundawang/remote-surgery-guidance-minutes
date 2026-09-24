from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.noise_map import router as noise_map_router
from app.core.database import (
    Base,
    NoiseDevice,
    NoiseDistrict,
    NoiseRecording,
    NoiseReportPoint,
    get_db,
)


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client(db_session):
    app = FastAPI()
    app.include_router(noise_map_router, prefix="/noise-map")

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def _add_district(db, name):
    district = NoiseDistrict(name=name)
    db.add(district)
    db.flush()
    return district


def _add_device(db, district, name, lat, lng):
    device = NoiseDevice(district_id=district.id, name=name, latitude=lat, longitude=lng)
    db.add(device)
    db.flush()
    return device


def _add_recording(db, device, level, recorded_at=None):
    rec = NoiseRecording(
        device_id=device.id,
        noise_level=level,
        recorded_at=recorded_at or datetime(2026, 9, 20, 12, 0, 0),
    )
    db.add(rec)
    db.flush()
    return rec


def _stats_by_name(payload):
    return {item["district_name"]: item for item in payload["district_stats"]}


def test_multi_district_multi_device_stats(client, db_session):
    district_a = _add_district(db_session, "A区")
    district_b = _add_district(db_session, "B区")
    dev_a1 = _add_device(db_session, district_a, "a1", 0.0, 0.0)
    dev_a2 = _add_device(db_session, district_a, "a2", 0.0, 1.0)
    dev_b1 = _add_device(db_session, district_b, "b1", 10.0, 10.0)

    # 投诉里的原始场景：A 区只有 80 和 60 两条有效录音，平均必须是 70 而不是 75
    _add_recording(db_session, dev_a1, 80.0)
    _add_recording(db_session, dev_a2, 60.0)
    _add_recording(db_session, dev_b1, 40.0)
    _add_recording(db_session, dev_b1, 50.0)
    db_session.commit()

    resp = client.get("/noise-map/district-stats")
    assert resp.status_code == 200
    stats = _stats_by_name(resp.json())

    assert stats["A区"]["device_count"] == 2
    assert stats["A区"]["recordings_count"] == 2
    assert stats["A区"]["avg_noise_level"] == 70.0
    assert stats["A区"]["max_noise_level"] == 80.0

    assert stats["B区"]["device_count"] == 1
    assert stats["B区"]["recordings_count"] == 2
    assert stats["B区"]["avg_noise_level"] == 45.0
    assert stats["B区"]["max_noise_level"] == 50.0


def test_none_and_zero_noise_levels(client, db_session):
    district_a = _add_district(db_session, "A区")
    district_b = _add_district(db_session, "B区")
    dev_a = _add_device(db_session, district_a, "a1", 0.0, 0.0)
    dev_b = _add_device(db_session, district_b, "b1", 10.0, 10.0)

    # None 视为无数据被剔除；0.0 是合法低噪声值，必须参与平均
    _add_recording(db_session, dev_a, 80.0)
    _add_recording(db_session, dev_a, None)
    _add_recording(db_session, dev_a, 0.0)
    # B 区全部是无效数据：不能返回假均值
    _add_recording(db_session, dev_b, None)
    db_session.commit()

    stats = _stats_by_name(client.get("/noise-map/district-stats").json())

    assert stats["A区"]["recordings_count"] == 2  # 分母只算有效值，不含 None
    assert stats["A区"]["avg_noise_level"] == 40.0  # (80 + 0) / 2，0 必须参与
    assert stats["A区"]["max_noise_level"] == 80.0

    assert stats["B区"]["recordings_count"] == 0
    assert stats["B区"]["avg_noise_level"] is None  # 无有效数据 -> null，不是假均值
    assert stats["B区"]["max_noise_level"] is None


def test_avg_precision_is_consistent(client, db_session):
    district_a = _add_district(db_session, "A区")
    dev_a = _add_device(db_session, district_a, "a1", 0.0, 0.0)
    # (80 + 61 + 60) / 3 = 67.0；(10 + 11 + 12) / 3 = 11.0；用一个除不尽的例子验证保留 2 位
    _add_recording(db_session, dev_a, 10.0)
    _add_recording(db_session, dev_a, 11.0)
    _add_recording(db_session, dev_a, 13.0)  # 34 / 3 = 11.333... -> 11.33
    db_session.commit()

    stats = _stats_by_name(client.get("/noise-map/district-stats").json())
    assert stats["A区"]["avg_noise_level"] == 11.33


def test_report_points_attributed_to_own_district(client, db_session):
    district_a = _add_district(db_session, "A区")
    district_b = _add_district(db_session, "B区")
    _add_device(db_session, district_a, "a1", 0.0, 0.0)
    _add_device(db_session, district_b, "b1", 10.0, 10.0)

    db_session.add(NoiseReportPoint(latitude=0.1, longitude=0.1))   # 离 A 区设备近
    db_session.add(NoiseReportPoint(latitude=9.9, longitude=10.2))  # 离 B 区设备近
    db_session.add(NoiseReportPoint(latitude=10.1, longitude=9.8))  # 离 B 区设备近
    db_session.commit()

    stats = _stats_by_name(client.get("/noise-map/district-stats").json())
    assert stats["A区"]["report_points"] == 1
    assert stats["B区"]["report_points"] == 2  # B 区能拿到自己的数量，不再是 0


def test_report_point_attribution_is_deterministic(client, db_session):
    district_a = _add_district(db_session, "A区")
    district_b = _add_district(db_session, "B区")
    # 举报点到两台设备距离完全相同，按设备 id 较小者归属，结果必须可重复
    _add_device(db_session, district_a, "a1", 1.0, 0.0)
    _add_device(db_session, district_b, "b1", -1.0, 0.0)
    db_session.add(NoiseReportPoint(latitude=0.0, longitude=0.0))
    db_session.commit()

    first = _stats_by_name(client.get("/noise-map/district-stats").json())
    second = _stats_by_name(client.get("/noise-map/district-stats").json())
    assert first == second
    assert first["A区"]["report_points"] == 1
    assert first["B区"]["report_points"] == 0


def test_time_range_filter(client, db_session):
    district_a = _add_district(db_session, "A区")
    dev_a = _add_device(db_session, district_a, "a1", 0.0, 0.0)
    base = datetime(2026, 9, 20, 12, 0, 0)
    _add_recording(db_session, dev_a, 80.0, recorded_at=base)
    _add_recording(db_session, dev_a, 40.0, recorded_at=base - timedelta(days=10))
    db_session.commit()

    stats = _stats_by_name(client.get(
        "/noise-map/district-stats",
        params={"start_time": (base - timedelta(hours=1)).isoformat()},
    ).json())
    assert stats["A区"]["recordings_count"] == 1
    assert stats["A区"]["avg_noise_level"] == 80.0

    stats = _stats_by_name(client.get(
        "/noise-map/district-stats",
        params={"end_time": (base - timedelta(days=1)).isoformat()},
    ).json())
    assert stats["A区"]["recordings_count"] == 1
    assert stats["A区"]["avg_noise_level"] == 40.0


def test_device_filter(client, db_session):
    district_a = _add_district(db_session, "A区")
    dev_a1 = _add_device(db_session, district_a, "a1", 0.0, 0.0)
    dev_a2 = _add_device(db_session, district_a, "a2", 0.0, 1.0)
    _add_recording(db_session, dev_a1, 80.0)
    _add_recording(db_session, dev_a2, 60.0)
    db_session.commit()

    stats = _stats_by_name(client.get(
        "/noise-map/district-stats", params={"device_id": dev_a1.id}
    ).json())
    assert stats["A区"]["device_count"] == 1
    assert stats["A区"]["recordings_count"] == 1
    assert stats["A区"]["avg_noise_level"] == 80.0


def test_heatmap_only_returns_valid_levels(client, db_session):
    district_a = _add_district(db_session, "A区")
    dev_a = _add_device(db_session, district_a, "a1", 1.0, 2.0)
    _add_recording(db_session, dev_a, 55.0)
    _add_recording(db_session, dev_a, None)
    _add_recording(db_session, dev_a, 0.0)
    db_session.commit()

    resp = client.get("/noise-map/heatmap")
    assert resp.status_code == 200
    points = resp.json()["heatmap"]
    assert sorted(p["noise_level"] for p in points) == [0.0, 55.0]
    assert all(p["district_id"] == district_a.id for p in points)
