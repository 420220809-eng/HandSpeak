"""
FastAPI Main Application
Sign Language Recognition API
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.model_loader import ModelLoader
from api.predictor import SignLanguagePredictor


# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="Sign Language Recognition API",
    description="Arabic Sign Language Recognition using GRU and MediaPipe",
    version="1.0.0",
    docs_url="/api/docs",  # Interactive docs at /api/docs
    redoc_url="/api/redoc",  # ReDoc at /api/redoc
    openapi_url="/api/openapi.json"  # OpenAPI schema
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Global Variables (loaded once at startup)
# ============================================================================

predictor: SignLanguagePredictor = None
model_info: Dict[str, Any] = None


# ============================================================================
# Startup Event - Load Model Once
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Load model and initialize predictor at startup
    This runs once when the server starts, not on every request
    """
    global predictor, model_info
    
    print("\n" + "="*60)
    print("Starting Sign Language Recognition API")
    print("="*60 + "\n")
    
    try:
        # Initialize model loader
        loader = ModelLoader(model_path='models/sign_gru_model.h5')
        
        # Load all components
        components = loader.load()
        
        # Get model info
        model_info = loader.get_model_info()
        
        # Initialize predictor with loaded components
        predictor = SignLanguagePredictor(
            model=components['model'],
            label_encoder=components['label_encoder'],
            sign_classes=components['sign_classes'],
            sequence_length=model_info.get('sequence_length', 30),
            confidence_threshold=model_info.get('confidence_threshold', 0.7)
        )
        
        print("✓ API ready to accept requests")
        print(f"✓ Loaded {model_info['num_classes']} sign classes")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load model at startup")
        print(f"Error: {str(e)}")
        print("\nPlease ensure:")
        print("  1. models/sign_gru_model.h5 exists")
        print("  2. data/processed/label_encoder.pkl exists")
        print("  3. data/processed/sign_classes.pkl exists")
        print("\nServer will start but /predict endpoint will fail.\n")
        # Don't crash the server, but predictor will be None


# ============================================================================
# Response Models
# ============================================================================

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str
    model_loaded: bool


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    success: bool
    predicted_sign: str = None
    confidence: float = None
    confidence_percentage: float = None
    meets_threshold: bool = None
    threshold: float = None
    top_3_predictions: Dict[str, float] = None
    extraction_metadata: Dict[str, Any] = None
    error: str = None


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """
    Root endpoint - API information
    """
    return {
        "message": "Sign Language Recognition API",
        "version": "1.0.0",
        "base_url": "http://localhost:5466",
        "endpoints": {
            "health": "GET /api/health - Check API status",
            "predict": "POST /api/predict - Upload video for prediction",
            "model_info": "GET /api/model-info - Get model information",
            "docs": "GET /api/docs - Interactive API documentation"
        }
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Confirms the API is running and model is loaded
    
    Returns:
        JSON with status and model loading state
    """
    model_loaded = predictor is not None
    
    if model_loaded:
        return HealthResponse(
            status="healthy",
            message="API is running and model is loaded",
            model_loaded=True
        )
    else:
        return HealthResponse(
            status="degraded",
            message="API is running but model failed to load",
            model_loaded=False
        )


@app.get("/api/model-info")
async def get_model_info():
    """
    Get information about the loaded model
    
    Returns:
        JSON with model metadata and sign classes
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs for errors."
        )
    
    return JSONResponse(content=model_info)


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_sign(file: UploadFile = File(...)):
    """
    Predict sign language from uploaded video
    
    Args:
        file: Video file (mp4, mov, avi, etc.)
        
    Returns:
        JSON with prediction results:
        - predicted_sign: The predicted sign name (Arabic)
        - confidence: Confidence score (0.0 to 1.0)
        - confidence_percentage: Confidence as percentage
        - meets_threshold: Whether confidence meets minimum threshold
        - top_3_predictions: Top 3 predictions with scores
        - extraction_metadata: Information about video processing
        
    Raises:
        HTTPException: If prediction fails
    """
    # Check if model is loaded
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server may have failed to start properly."
        )
    
    # Validate file type
    allowed_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Read video file bytes
        video_bytes = await file.read()
        
        # Validate file size (max 50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        if len(video_bytes) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: 50MB"
            )
        
        # Run prediction
        result = predictor.predict_from_bytes(video_bytes)
        
        # Return successful prediction
        return PredictionResponse(
            success=True,
            predicted_sign=result['predicted_sign'],
            confidence=result['confidence'],
            confidence_percentage=result['confidence_percentage'],
            meets_threshold=result['meets_threshold'],
            threshold=result['threshold'],
            top_3_predictions=result['top_3_predictions'],
            extraction_metadata=result['extraction_metadata']
        )
        
    except ValueError as e:
        # Handle video processing errors
        return PredictionResponse(
            success=False,
            error=f"Video processing error: {str(e)}"
        )
        
    except Exception as e:
        # Handle unexpected errors
        return PredictionResponse(
            success=False,
            error=f"Prediction failed: {str(e)}"
        )


# ============================================================================
# Run Server (for development)
# ============================================================================

if __name__ == "__main__":
    """
    Run the API server
    For production, use: uvicorn api.main:app --host 0.0.0.0 --port 5466
    """
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5466,  # Default port 5466
        reload=False  # Set to True for development auto-reload
    )
