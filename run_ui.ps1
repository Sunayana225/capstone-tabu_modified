$ErrorActionPreference = "Stop"

Write-Host "Starting Streamlit UI on http://127.0.0.1:8501" -ForegroundColor Green
py -m streamlit run app/ui.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false

