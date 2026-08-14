@echo off
echo DISABLED: experimental backend 8015. Use canonical 8014 only.
exit /b 1
cd /d D:\lanqun-site\ragv6-standalone\retrieval-service-vnext
python -m uvicorn api_server_vnext_exact:app --host 127.0.0.1 --port 8015 > vnext-exact-api-8015.log 2>&1
