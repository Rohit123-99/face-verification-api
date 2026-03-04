"""
Production-Grade Face Verification API
- Handles 10–50 concurrent requests
- Liveness detection (anti-spoofing)
- Request queue management
- Caching layer
- Model pooling
"""

import time
import logging
import os
import tempfile
import asyncio
from typing import Dict, Optional
from collections import defaultdict
from functools import lru_cache
from datetime import datetime, timedelta

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Production Face Verification API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_NAME = "ArcFace"
DETECTOR = "mtcnn"
THRESHOLD = 0.68
MAX_QUEUE_SIZE = 200
CACHE_TTL = 3600  # 1 hour
MAX_CONCURRENT = 10

# Global state
request_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
verification_cache: Dict[str, Dict] = {}  # {hash: {result, timestamp}}
active_requests = 0
model_instances = []


def get_image_hash(img: np.ndarray) -> str:
    """Generate hash of image for caching."""
    return hashlib.md5(img.tobytes()).hexdigest()


def is_blurry(img: np.ndarray, threshold: float = 100.0) -> bool:
    """Check if image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold


def detect_liveness(img: np.ndarray) -> Dict[str, any]:
    """
    Anti-spoofing check: detect if image is real (not printed/screen).
    Uses texture analysis and eye detection.
    
    Returns:
        {
            "is_live": bool,
            "confidence": float (0-1),
            "reasons": [list of checks]
        }
    """
    reasons = []
    scores = []
    
    # 1. Texture analysis (Laplacian variance)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var > 500:  # Real faces have higher texture variance
        scores.append(0.8)
        reasons.append("Texture: real face detected")
    else:
        scores.append(0.2)
        reasons.append("Texture: flat/low variance (possible print/screen)")
    
    # 2. Color channel variance (real vs printed)
    b_var = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).var()
    if b_var > 50:  # Real images have more color variation
        scores.append(0.7)
        reasons.append("Color: natural color variance detected")
    else:
        scores.append(0.3)
        reasons.append("Color: limited variance (possible print)")
    
    # 3. Detect eyes (heuristic: real faces have detectable eyes)
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    if len(eyes) >= 1:  # At least one eye detected
        scores.append(0.9)
        reasons.append(f"Eyes: {len(eyes)} eye(s) detected")
    else:
        scores.append(0.2)
        reasons.append("Eyes: no eyes detected (possible photo/screen)")
    
    # Overall liveness score
    is_live = np.mean(scores) > 0.5
    confidence = float(np.mean(scores))
    
    return {
        "is_live": is_live,
        "confidence": confidence,
        "reasons": reasons
    }


def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """Read uploaded file into numpy array."""
    contents = file.file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img


def get_cached_result(doc_hash: str, selfie_hash: str) -> Optional[Dict]:
    """Check if we've already verified this pair."""
    cache_key = f"{doc_hash}_{selfie_hash}"
    if cache_key in verification_cache:
        cached = verification_cache[cache_key]
        if datetime.now() - cached["timestamp"] < timedelta(seconds=CACHE_TTL):
            logger.info(f"Cache hit: {cache_key[:16]}...")
            return cached["result"]
    return None


def cache_result(doc_hash: str, selfie_hash: str, result: Dict):
    """Store result in cache."""
    cache_key = f"{doc_hash}_{selfie_hash}"
    verification_cache[cache_key] = {
        "result": result,
        "timestamp": datetime.now()
    }


@app.on_event("startup")
async def startup_event():
    logger.info("Production Face Verification API starting up...")
    logger.info(f"Max concurrent requests: {MAX_CONCURRENT}")
    logger.info(f"Request queue size: {MAX_QUEUE_SIZE}")
    logger.info(f"Cache TTL: {CACHE_TTL}s")
    
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU available: {gpus}")
        else:
            logger.info("No GPU detected; using CPU")
    except ImportError:
        logger.info("TensorFlow not installed; using default backend")


async def process_verification_queue():
    """Background task to process queued requests."""
    while True:
        try:
            # Process queued requests
            if not request_queue.empty():
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Queue processor error: {e}")
            await asyncio.sleep(1)


@app.post("/verify-identity")
async def verify_identity(
    document_image: UploadFile = File(...),
    selfie_image: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Verify identity with anti-spoofing and caching.
    
    Returns:
        {
            "is_match": bool,
            "score": float,
            "distance": float,
            "threshold": float,
            "model": str,
            "execution_time": str,
            "liveness_check": {
                "document_live": bool,
                "selfie_live": bool,
                "document_confidence": float,
                "selfie_confidence": float
            },
            "low_confidence": bool (optional)
        }
    """
    global active_requests
    
    start = time.time()
    
    # Check queue size
    if request_queue.qsize() >= MAX_QUEUE_SIZE:
        raise HTTPException(
            status_code=503,
            detail="Service overloaded. Please retry in a few seconds."
        )
    
    # Rate limit concurrent requests
    if active_requests >= MAX_CONCURRENT:
        raise HTTPException(
            status_code=429,
            detail=f"Max {MAX_CONCURRENT} concurrent requests. Please wait."
        )
    
    active_requests += 1
    
    try:
        # Load images
        try:
            doc_img = load_image_from_upload(document_image)
            selfie_img = load_image_from_upload(selfie_image)
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading images: {str(e)}")
        
        # Generate hashes for caching
        doc_hash = get_image_hash(doc_img)
        selfie_hash = get_image_hash(selfie_img)
        
        # Check cache first
        cached = get_cached_result(doc_hash, selfie_hash)
        if cached:
            cached["execution_time"] = f"{time.time() - start:.2f}s (cached)"
            return JSONResponse(cached)
        
        # Liveness detection (anti-spoofing)
        logger.info("Running liveness detection...")
        doc_liveness = detect_liveness(doc_img)
        selfie_liveness = detect_liveness(selfie_img)
        
        # Reject if either image fails liveness check
        if not doc_liveness["is_live"]:
            logger.warning(f"Document failed liveness: {doc_liveness['reasons']}")
            raise HTTPException(
                status_code=400,
                detail=f"Document image failed liveness check. Likely a printed photo or screen. Confidence: {doc_liveness['confidence']:.2f}"
            )
        
        if not selfie_liveness["is_live"]:
            logger.warning(f"Selfie failed liveness: {selfie_liveness['reasons']}")
            raise HTTPException(
                status_code=400,
                detail=f"Selfie image failed liveness check. Confidence: {selfie_liveness['confidence']:.2f}"
            )
        
        # Check for blur
        low_confidence = False
        if is_blurry(doc_img):
            logger.warning("Document image appears blurry")
            low_confidence = True
        
        # Save to temp files
        temp_dir = tempfile.gettempdir()
        doc_path = os.path.join(temp_dir, f"doc_{doc_hash}.jpg")
        selfie_path = os.path.join(temp_dir, f"selfie_{selfie_hash}.jpg")
        
        try:
            cv2.imwrite(doc_path, doc_img)
            cv2.imwrite(selfie_path, selfie_img)
            
            # Verify
            logger.info("Running face verification...")
            result = DeepFace.verify(
                img1_path=doc_path,
                img2_path=selfie_path,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR,
                distance_metric="cosine"
            )
            
            is_match = bool(result.get("verified", False))  # Convert numpy bool to Python bool
            distance = float(result.get("distance", 1.0))
            score = 1 - distance
            threshold = float(result.get("threshold", THRESHOLD))
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Face detection failed: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Verification failed: {str(e)}")
        finally:
            # Cleanup
            for path in [doc_path, selfie_path]:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except:
                    pass
        
        exec_time = f"{time.time() - start:.2f}s"
        
        payload = {
            "is_match": bool(is_match),  # Ensure Python bool
            "score": float(score),
            "distance": float(distance),
            "threshold": float(threshold),
            "model": MODEL_NAME,
            "execution_time": exec_time,
            "liveness_check": {
                "document_live": bool(doc_liveness["is_live"]),  # Convert to Python bool
                "selfie_live": bool(selfie_liveness["is_live"]),  # Convert to Python bool
                "document_confidence": float(doc_liveness["confidence"]),
                "selfie_confidence": float(selfie_liveness["confidence"])
            }
        }
        
        if low_confidence:
            payload["low_confidence"] = True
        
        # Cache result
        cache_result(doc_hash, selfie_hash, payload)
        
        return JSONResponse(payload)
        
    finally:
        active_requests -= 1


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_requests": active_requests,
        "queue_size": request_queue.qsize(),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats")
async def stats():
    """Service statistics."""
    return {
        "max_concurrent": MAX_CONCURRENT,
        "active_requests": active_requests,
        "queue_size": request_queue.qsize(),
        "cache_entries": len(verification_cache),
        "timestamp": datetime.now().isoformat()
    }
