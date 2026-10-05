import time
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import slices, resources, models, monitoring
from .schemas import HealthResponse

app = FastAPI(
    title="6G Network Slicing Platform API",
    description="API for managing 6G network slices, resources, and AI models.",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

app.include_router(slices.router, prefix="/api/v1/slices", tags=["Slices"])
app.include_router(resources.router, prefix="/api/v1/resources", tags=["Resources"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(monitoring.router, prefix="/api/v1/monitoring", tags=["Monitoring"])

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse(status="healthy", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    print("Starting up 6G Network Slicing Platform API...")

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down 6G Network Slicing Platform API...")
