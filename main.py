import time
import logging
import os
import tempfile

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace

# Create logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Face Verification API")

# Enable CORS to allow requests from the browser/Swagger UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "ArcFace"  # Lighter & faster than Facenet512
DETECTOR = "mtcnn"  # Uses MTCNN detector (fast and reliable)
THRESHOLD = 0.68  # ArcFace uses higher threshold than Facenet512


@app.on_event("startup")
def startup_event():
    logger.info("Face Verification API starting up...")
    try:
        import tensorflow as tf
        logger.info("TensorFlow version: %s", tf.__version__)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info("GPU available: %s", gpus)
        else:
            logger.info("No GPU detected; will use CPU")
    except ImportError:
        logger.info("TensorFlow not installed; using default backend")


def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """Read uploaded file into numpy array (OpenCV format: BGR)."""
    contents = file.file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img


def is_blurry(img: np.ndarray, threshold: float = 100.0) -> bool:
    """Check if image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold


@app.post("/verify-identity")
async def verify_identity(
    document_image: UploadFile = File(...),
    selfie_image: UploadFile = File(...)
):
    start = time.time()
    
    # Load images
    try:
        doc_img = load_image_from_upload(document_image)
        selfie_img = load_image_from_upload(selfie_image)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading images: {str(e)}")
    
    # Check for blur
    low_confidence = False
    if is_blurry(doc_img):
        logger.warning("Document image appears blurry")
        low_confidence = True
    
    # Save images temporarily for DeepFace verification
    # (DeepFace.verify expects file paths, not numpy arrays)
    # Use tempfile for cross-platform compatibility (Windows, Mac, Linux)
    temp_dir = tempfile.gettempdir()
    doc_path = os.path.join(temp_dir, "doc_temp.jpg")
    selfie_path = os.path.join(temp_dir, "selfie_temp.jpg")
    
    try:
        cv2.imwrite(doc_path, doc_img)
        cv2.imwrite(selfie_path, selfie_img)
        
        # Verify faces using DeepFace
        # DeepFace.verify() detects faces internally and compares embeddings
        result = DeepFace.verify(
            img1_path=doc_path,
            img2_path=selfie_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            distance_metric="cosine"
        )
        
        is_match = result.get("verified", False)
        distance = result.get("distance", 1.0)
        score = 1 - distance  # Convert distance to similarity score
        threshold = result.get("threshold", THRESHOLD)
        
    except ValueError as e:
        # Typically raised when no face is detected
        raise HTTPException(status_code=400, detail=f"Face detection failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Verification failed: {str(e)}")
    finally:
        # Clean up temp files
        for path in [doc_path, selfie_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass
    
    end = time.time()
    exec_time = f"{end - start:.2f}s"
    
    payload = {
        "is_match": is_match,
        "score": float(score),
        "distance": float(distance),
        "threshold": float(threshold),
        "model": MODEL_NAME,
        "execution_time": exec_time,
    }
    
    if low_confidence:
        payload["low_confidence"] = True
    
    return JSONResponse(payload)
