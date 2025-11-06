$ErrorActionPreference = "Stop"

Write-Host "Starting FastAPI server on http://127.0.0.1:8000 with extended timeouts for large problems" -ForegroundColor Green
py -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --timeout-keep-alive 1800 --timeout-graceful-shutdown 60

