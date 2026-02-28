# Face Verification API

This project provides a FastAPI service to verify a document photo against a live selfie using DeepFace.
It uses GPU acceleration (TensorFlow or ONNX Runtime with GPU support) and a RetinaFace detector.

## Setup

1. Create a virtual environment and activate it (Windows example):

```powershell
python -m venv venv
# If PowerShell prevents running scripts, you can either use cmd.exe:
#   venv\Scripts\activate.bat
# or adjust policy for this session:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# then run:
venv\Scripts\Activate.ps1
```

*PowerShell may block script execution by default; the above hints show how to bypass.*

2. Upgrade `pip` and install dependencies:

```bash
# use the interpreter to upgrade pip (avoids the "run the following command" error)
python -m pip install --upgrade pip

# then install the requirements
pip install -r requirements.txt
```

> If you still see an error about modifying pip, copy the suggested command exactly from the output and run it; it typically looks like:
> `C:\Users\Rohit\Project\Face detection api\venv\Scripts\python.exe -m pip install --upgrade pip`

3. Run the server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The model will load once at startup (~5–10s); subsequent requests should complete in **2–5 seconds**.

## API

### Endpoint

`POST /verify-identity`

### Content Type

`multipart/form-data`

### Form Fields

| Field Name       | Type | Description                     |
|------------------|------|---------------------------------|
| document_image   | File | Passport/Aadhar photo (JPEG/PNG)
| selfie_image     | File | Live selfie image (JPEG/PNG)   |

Example `curl` request:

```bash
curl -X POST "http://localhost:8000/verify-identity" \
  -F "document_image=@/path/to/doc.jpg" \
  -F "selfie_image=@/path/to/selfie.jpg"
```

### Successful Response

```json
{
  "is_match": true,
  "score": 0.85,
  "distance": 0.15,
  "threshold": 0.68,
  "model": "ArcFace",
  "execution_time": "2.5s",
  "low_confidence": true  # optional, present if document was blurry
}
```

- `is_match`: Boolean indicating whether faces are similar enough.
- `score`: Similarity score (1 - cosine distance).
- `distance`: Cosine distance between embeddings.
- `threshold`: Threshold used for deciding match.
- `model`: Embedding model name.
- `execution_time`: Time spent processing the request.
- `low_confidence`: Optional flag when document image quality is poor.

### Error Responses

- **400 Bad Request**
  - No face detected in one of the images.
  - Uploaded file could not be interpreted as an image.

Example curl for error:

```bash
curl -X POST "http://localhost:8000/verify-identity" \
  -F "document_image=@notanimage.txt" \
  -F "selfie_image=@selfie.jpg"
```

The response body contains a JSON object with `detail` describing the error.

> **Note:** For production deployments, secure the endpoint, enable rate limiting, and log requests appropriately.

## Notes

- **Optimized for speed**: Uses ArcFace (lightweight) and MTCNN detector (fast and reliable).
- Expected latency: ~5–10s on first request (model load), ~2–5s on subsequent requests.
- The service uses cosine similarity by default.
- Blurry document images set a `low_confidence` flag.
- The largest face is chosen if multiple faces are present.
- **CORS is enabled** for local testing; restrict origins in production (`allow_origins=["https://yourdomain.com"]`).


## Troubleshooting

### Dependency conflict during `pip install`

If you see an error like:

```
ERROR: Cannot install -r requirements.txt ...
The conflict is caused by:
    deepface 0.0.xx depends on tensorflow>=1.9.0
...
Additionally, some packages in these conflicts have no matching distributions available for your environment:
    tensorflow
```

it means pip cannot find a TensorFlow wheel for your current Python version (the traceback often shows `cp314` for Python 3.14). DeepFace currently requires TensorFlow, and TensorFlow does **not yet publish wheels for Python 3.14**.

**Solution**: use a Python version that TensorFlow supports (3.11 is the latest supported release as of early 2026). To fix:

1. Install Python 3.11 (from the official installer or [python.org](https://python.org)).
2. Delete the existing virtual environment (`rmdir /s /q venv` on Windows).
3. Recreate it with the 3.11 interpreter:
   ```powershell
   C:\Path\To\Python311\python.exe -m venv venv
   venv\Scripts\Activate.ps1   # or activate.bat
   ```
4. Repeat the `pip install -r requirements.txt` step – the conflict should disappear.

Alternatively, if you wish to avoid TensorFlow entirely you can remove `tf-keras` from `requirements.txt` and install `onnxruntime`/`onnxruntime-gpu` only; DeepFace will still work but certain models may require TensorFlow.

Always check the Python version with `python --version` before creating the venv to avoid this issue.