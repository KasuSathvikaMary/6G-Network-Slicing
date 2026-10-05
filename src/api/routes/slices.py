from fastapi import APIRouter, HTTPException
import uuid
from typing import List
from ..schemas import SliceCreateRequest, SliceResponse, SliceListResponse, MetricsResponse, ExplanationResponse

router = APIRouter()

# Mock data store
_slices = {}

@router.post("", response_model=SliceResponse)
def create_slice(request: SliceCreateRequest):
    slice_id = str(uuid.uuid4())
    new_slice = SliceResponse(
        id=slice_id,
        name=request.name,
        type=request.type,
        status="active",
        allocated_resources={"bandwidth": 100.0, "compute": 10.0}
    )
    _slices[slice_id] = new_slice
    return new_slice

@router.get("", response_model=SliceListResponse)
def list_slices():
    return SliceListResponse(slices=list(_slices.values()))

@router.get("/{slice_id}", response_model=SliceResponse)
def get_slice(slice_id: str):
    if slice_id not in _slices:
        raise HTTPException(status_code=404, detail="Slice not found")
    return _slices[slice_id]

@router.put("/{slice_id}", response_model=SliceResponse)
def update_slice(slice_id: str, request: SliceCreateRequest):
    if slice_id not in _slices:
        raise HTTPException(status_code=404, detail="Slice not found")
    
    updated_slice = SliceResponse(
        id=slice_id,
        name=request.name,
        type=request.type,
        status="active",
        allocated_resources=_slices[slice_id].allocated_resources
    )
    _slices[slice_id] = updated_slice
    return updated_slice

@router.delete("/{slice_id}")
def delete_slice(slice_id: str):
    if slice_id not in _slices:
        raise HTTPException(status_code=404, detail="Slice not found")
    del _slices[slice_id]
    return {"message": "Slice deleted successfully"}

@router.post("/{slice_id}/optimize")
def optimize_slice(slice_id: str):
    if slice_id not in _slices:
        raise HTTPException(status_code=404, detail="Slice not found")
    return {"message": "Optimization triggered successfully"}

@router.get("/{slice_id}/metrics", response_model=MetricsResponse)
def get_slice_metrics(slice_id: str):
    if slice_id not in _slices:
        raise HTTPException(status_code=404, detail="Slice not found")
    return MetricsResponse(
        slice_id=slice_id,
        metrics={"throughput": 95.5, "latency": 12.3}
    )

@router.get("/{slice_id}/explain", response_model=ExplanationResponse)
def explain_slice_decision(slice_id: str):
    if slice_id not in _slices:
        raise HTTPException(status_code=404, detail="Slice not found")
    return ExplanationResponse(
        decision="Increase bandwidth by 10%",
        confidence=0.89,
        factors={"high_traffic": 0.7, "sla_risk": 0.3},
        nl_explanation="Bandwidth was increased to prevent SLA violation due to high traffic."
    )
