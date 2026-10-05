from fastapi import APIRouter, HTTPException
from typing import List
from ..schemas import ModelInfoResponse, TrainingStatusResponse

router = APIRouter()

# Mock models
_models = {
    "model-1": ModelInfoResponse(id="model-1", version="v1.0", status="active", metrics={"accuracy": 0.95}),
    "model-2": ModelInfoResponse(id="model-2", version="v1.1", status="training", metrics={"accuracy": 0.0})
}

@router.get("", response_model=List[ModelInfoResponse])
def list_models():
    return list(_models.values())

@router.get("/{model_id}", response_model=ModelInfoResponse)
def get_model(model_id: str):
    if model_id not in _models:
        raise HTTPException(status_code=404, detail="Model not found")
    return _models[model_id]

@router.post("/retrain", response_model=TrainingStatusResponse)
def retrain_model():
    return TrainingStatusResponse(
        status="training_started",
        progress=0.0,
        current_loss=1.5
    )

@router.post("/{model_id}/deploy")
def deploy_model(model_id: str):
    if model_id not in _models:
        raise HTTPException(status_code=404, detail="Model not found")
    _models[model_id].status = "active"
    return {"message": f"Model {model_id} deployed successfully"}

@router.post("/{model_id}/rollback")
def rollback_model(model_id: str):
    if model_id not in _models:
        raise HTTPException(status_code=404, detail="Model not found")
    _models[model_id].status = "rolled_back"
    return {"message": f"Model {model_id} rolled back successfully"}

@router.get("/performance/metrics")
def model_performance():
    return {
        "average_inference_time_ms": 15.2,
        "prediction_accuracy": 0.94,
        "reward_trend": "increasing"
    }
