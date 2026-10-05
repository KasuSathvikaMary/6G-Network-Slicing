from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from ..schemas import DashboardResponse, AlertResponse

router = APIRouter()

@router.get("/dashboard", response_model=DashboardResponse)
def get_dashboard():
    return DashboardResponse(
        overview={"active_slices": 5, "total_throughput": 45.0},
        slices=[{"id": "slice-1", "health": "good"}],
        system_health="healthy"
    )

@router.get("/alerts", response_model=AlertResponse)
def get_alerts():
    return AlertResponse(
        alerts=[
            {"id": "alert-1", "severity": "warning", "message": "High CPU usage"}
        ]
    )

@router.get("/metrics")
def get_system_metrics():
    return {
        "cpu_usage": 65.0,
        "memory_usage": 72.0,
        "network_io": 1050.0
    }

@router.get("/traces/{trace_id}")
def get_trace(trace_id: str):
    return {
        "trace_id": trace_id,
        "spans": [
            {"id": "span-1", "name": "request_handler", "duration_ms": 15},
            {"id": "span-2", "name": "db_query", "duration_ms": 5}
        ]
    }
