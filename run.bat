@echo off
REM Starts both programs, each in its own window.
REM Camera IPs, user and password live in the ".env" file next to face_api.py / test2.py.
cd /d "%~dp0"

start "Fetch images (watch)" cmd /k python fetch_image.py --watch --interval 10
start "Face API server" cmd /k python face_api.py

