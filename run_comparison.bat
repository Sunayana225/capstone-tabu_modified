@echo off
echo Starting Algorithm Comparison Dashboard...
echo.
echo Make sure the API is running first with run_api.ps1
echo If API is not running, press Ctrl+C and start it first.
echo.
pause
echo.
echo Starting Streamlit comparison dashboard...
streamlit run app/comparison_ui.py --server.port 8502 --server.headless false
pause