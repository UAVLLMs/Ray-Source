@echo off
cd /d D:\lanqun-site\ragv6-standalone\retrieval-service-vnext
setlocal EnableExtensions

rem One worker is intentional: DINOv2 and its image index remain resident in
rem this process. If uvicorn exits unexpectedly, the watchdog restarts it.
:restart
echo [%date% %time%] Starting canonical multimodal API on 8014>>vnext-exact-api-8014.log
python -m uvicorn api_server:app --host 127.0.0.1 --port 8014 --workers 1 >>vnext-exact-api-8014.log 2>&1
echo [%date% %time%] API exited with code %errorlevel%; restarting in 2 seconds>>vnext-exact-api-8014.log
timeout /t 2 /nobreak >nul
goto restart
