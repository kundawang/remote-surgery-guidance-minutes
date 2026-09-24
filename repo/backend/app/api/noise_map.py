from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..core.database import (
    NoiseDevice,
    NoiseDistrict,
    NoiseRecording,
    NoiseReportPoint,
    get_db,
)

router = APIRouter()

# avg_noise_level 统一保留的小数位数，所有区域均值走同一精度
AVG_NOISE_DECIMALS = 2


def _valid_levels(recordings):
    # 有效噪声值 = noise_level 非 None。
    # None 视为无数据被剔除；0.0 是合法低噪声值，必须保留，
    # 因此不能用 `if rec.noise_level` 这种真值判断（会把 0 误杀）。
    return [rec.noise_level for rec in recordings if rec.noise_level is not None]


def _average(levels):
    # 分母只用参与计算的有效数据量（len(levels)），绝不用总记录数；
    # 没有有效值时返回 None（JSON null），明确表达"无数据"，不伪造均值。
    if not levels:
        return None
    return round(sum(levels) / len(levels), AVG_NOISE_DECIMALS)


def _attribute_district(point, devices):
    # 举报点归属规则（对同一份数据结果确定）：
    # 按经纬度欧氏距离归到"最近设备"所在的区域；
    # 距离相同（含同区域多台设备）时取设备 id 较小者，保证结果可重复。
    # 没有任何设备时举报点无法归属，不计入任何区域。
    if not devices:
        return None
    nearest = min(
        devices,
        key=lambda d: (
            (d.latitude - point.latitude) ** 2 + (d.longitude - point.longitude) ** 2,
            d.id,
        ),
    )
    return nearest.district_id


@router.get("/district-stats")
def get_district_stats(
    start_time: Optional[datetime] = Query(default=None),
    end_time: Optional[datetime] = Query(default=None),
    device_id: Optional[int] = Query(default=None),
    db: Session = Depends(get_db),
):
    device_query = db.query(NoiseDevice)
    if device_id is not None:
        device_query = device_query.filter(NoiseDevice.id == device_id)
    devices = device_query.order_by(NoiseDevice.id).all()
    device_ids = [d.id for d in devices]

    recording_query = db.query(NoiseRecording).filter(
        NoiseRecording.device_id.in_(device_ids or [-1])
    )
    if start_time is not None:
        recording_query = recording_query.filter(NoiseRecording.recorded_at >= start_time)
    if end_time is not None:
        recording_query = recording_query.filter(NoiseRecording.recorded_at <= end_time)
    recordings = recording_query.all()

    report_query = db.query(NoiseReportPoint)
    if start_time is not None:
        report_query = report_query.filter(NoiseReportPoint.created_at >= start_time)
    if end_time is not None:
        report_query = report_query.filter(NoiseReportPoint.created_at <= end_time)
    report_points = report_query.all()

    districts = db.query(NoiseDistrict).order_by(NoiseDistrict.id).all()
    stats_by_district = {
        district.id: {
            "district_id": district.id,
            "district_name": district.name,
            "device_count": 0,
            "recordings_count": 0,
            "avg_noise_level": None,
            "max_noise_level": None,
            "report_points": 0,
        }
        for district in districts
    }

    for device in devices:
        if device.district_id in stats_by_district:
            stats_by_district[device.district_id]["device_count"] += 1

    device_district = {d.id: d.district_id for d in devices}
    recordings_by_district = {district.id: [] for district in districts}
    for rec in recordings:
        district_id = device_district.get(rec.device_id)
        if district_id in recordings_by_district:
            recordings_by_district[district_id].append(rec)

    for district_id, district_recordings in recordings_by_district.items():
        levels = _valid_levels(district_recordings)
        stats = stats_by_district[district_id]
        # recordings_count 与平均/最大值的计算口径一致：只数有效值，
        # 保证分母 = 参与计算的数据量，而不是总记录数
        stats["recordings_count"] = len(levels)
        stats["avg_noise_level"] = _average(levels)
        stats["max_noise_level"] = max(levels) if levels else None

    for point in report_points:
        district_id = _attribute_district(point, devices)
        if district_id in stats_by_district:
            stats_by_district[district_id]["report_points"] += 1

    return {"district_stats": list(stats_by_district.values())}


@router.get("/heatmap")
def get_heatmap(
    start_time: Optional[datetime] = Query(default=None),
    end_time: Optional[datetime] = Query(default=None),
    device_id: Optional[int] = Query(default=None),
    db: Session = Depends(get_db),
):
    query = (
        db.query(NoiseRecording, NoiseDevice)
        .join(NoiseDevice, NoiseRecording.device_id == NoiseDevice.id)
        .filter(NoiseRecording.noise_level.isnot(None))
    )
    if device_id is not None:
        query = query.filter(NoiseRecording.device_id == device_id)
    if start_time is not None:
        query = query.filter(NoiseRecording.recorded_at >= start_time)
    if end_time is not None:
        query = query.filter(NoiseRecording.recorded_at <= end_time)

    points = [
        {
            "latitude": device.latitude,
            "longitude": device.longitude,
            "noise_level": rec.noise_level,
            "device_id": device.id,
            "district_id": device.district_id,
            "recorded_at": rec.recorded_at,
        }
        for rec, device in query.all()
    ]
    return {"heatmap": points}
