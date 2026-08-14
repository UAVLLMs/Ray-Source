@echo off
echo DISABLED: legacy experimental backend 8012. Use canonical 8014 only.
exit /b 1
cd /d D:\lanqun-site\ragv6-standalone\retrieval-service-vnext
python -m uvicorn api_server_vnext:app --host 127.0.0.1 --port 8012 --workers 1 --log-level info >> experiments\vnext-api-8012.log 2>&1
