@echo off
title FireDetect AI - Starting...
echo.
echo  ==========================================
echo   🔥 FireDetect AI - Human Detection System
echo  ==========================================
echo.
echo  Starting website... please wait.
echo  Your browser will open automatically.
echo.
cd /d "%~dp0"
python -m streamlit run website/app.py --server.port 8501
pause
