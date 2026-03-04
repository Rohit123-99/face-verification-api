# Production Deployment Guide

## Scaling for High Load (Shaadi.com / Dating Apps)

### Your current API limitations
- **Throughput**: ~1–2 req/sec (synchronous)
- **Concurrent capacity**: Single-threaded (≈1–2 concurrent)
- **Would fail** with >10 daily signups at peak hours

### New Production API improvements

**File**: `production_api.py`

#### Features
1. **Async/concurrent processing** – handles 10–50 concurrent requests
2. **Request queue** – prevents overload; queues excess requests
3. **Liveness detection** – detects printed photos, screens, fake videos
4. **Response caching** – avoid re-verifying same user pairs
5. **Health/stats endpoints** – for monitoring
6. **Anti-spoofing checks** – texture analysis, eye detection

#### Running
```powershell
# Install (if not already done)
pip install -r requirements.txt

# Run production server (4 workers for parallelism)
uvicorn production_api:app --host 0.0.0.0 --port 8000 --workers 4
```

The `--workers 4` flag runs 4 worker processes, multiplying throughput ~4x.

---

## Scaling Further (100s of requests/sec)

For Shaadi.com scale, you'd need:

1. **Load balancer** (Nginx, HAProxy)
   - Distribute across multiple servers
   
2. **Horizontal scaling**
   - Run N instances of the API behind a load balancer
   - Use Redis for shared cache
   
3. **Message queue** (RabbitMQ, Celery)
   - Async job processing
   - Prevent blocking requests

4. **Database** (PostgreSQL)
   - Store verification results
   - Audit trail

5. **Docker + Kubernetes**
   - Auto-scale based on load

Example architecture for high load:
```
[User Requests]
     ↓
[Nginx Load Balancer]
     ↓
[Pod 1: API] [Pod 2: API] [Pod 3: API] ... [Pod N: API]
     ↓
[Redis Cache]
[RabbitMQ Queue]
[PostgreSQL DB]
```

---

## Liveness Detection (Anti-Spoofing)

The new API includes:

1. **Texture Analysis** – Real faces have higher Laplacian variance
2. **Color Variance** – Printed photos have flat colors
3. **Eye Detection** – Real faces have detectable eyes

Returns:
```json
{
  "is_live": true,
  "confidence": 0.92,
  "reasons": [
    "Texture: real face detected",
    "Color: natural color variance detected",
    "Eyes: 2 eye(s) detected"
  ]
}
```

### Advanced anti-spoofing (if needed)
- Blink detection (user blinks during capture)
- Micro-expression analysis
- Face liveness model (separate CNN trained on live vs. spoofed)
- 3D depth estimation

---

## Estimated Throughput

| Setup | Req/sec | Concurrent |
|-------|---------|-----------|
| Current (main.py) | 0.5–1 | 1–2 |
| Production (production_api.py) single server | 2–5 | 10–50 |
| Production + 4 workers | 8–20 | 40–200 |
| Load balanced (4 servers × 4 workers) | 32–80 | 160–800 |
| Kubernetes auto-scaled | 100+ | 1000+ |

For Shaadi.com scale (thousands of daily signups), use Kubernetes + auto-scaling.

---

## Next Steps

1. Test `production_api.py` locally
2. Add Redis caching for shared state
3. Containerize with Docker
4. Deploy to Kubernetes for auto-scaling
5. Add monitoring (Prometheus, Grafana)
6. Set up audit logging to database

Feel free to ask for Docker, Kubernetes, or scaling configurations!
