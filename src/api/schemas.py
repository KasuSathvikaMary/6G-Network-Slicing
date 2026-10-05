from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class SliceCreateRequest(BaseModel):
    name: str = Field(..., example="eMBB-Slice-1")
    type: str = Field(..., example="eMBB")
    requirements: Dict[str, float] = Field(..., example={"bandwidth": 1000.0, "latency": 10.0})
    priority: int = Field(1, ge=1, le=10)

class SliceResponse(BaseModel):
    id: str
    name: str
    type: str
    status: str
    allocated_resources: Dict[str, float]
    
class SliceListResponse(BaseModel):
    slices: List[SliceResponse]

class ResourceAllocationRequest(BaseModel):
    slice_id: str
    resources: Dict[str, float]

class ResourceAllocationResponse(BaseModel):
    success: bool
    message: str
    allocated: Dict[str, float]

class MetricsResponse(BaseModel):
    slice_id: str
    metrics: Dict[str, Any]

class DashboardResponse(BaseModel):
    overview: Dict[str, Any]
    slices: List[Dict[str, Any]]
    system_health: str

class ModelInfoResponse(BaseModel):
    id: str
    version: str
    status: str
    metrics: Dict[str, float]

class TrainingStatusResponse(BaseModel):
    status: str
    progress: float
    current_loss: float

class ExplanationResponse(BaseModel):
    decision: str
    confidence: float
    factors: Dict[str, float]
    nl_explanation: str

class AuditResponse(BaseModel):
    events: List[Dict[str, Any]]

class AlertResponse(BaseModel):
    alerts: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    version: str
