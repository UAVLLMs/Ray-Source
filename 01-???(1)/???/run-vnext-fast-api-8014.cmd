@echo off
cd /d D:\lanqun-site\ragv6-standalone\retrieval-service-vnext
"C:\Users\lenovo\AppData\Local\Programs\Python\Python312\python.exe" -m uvicorn api_server_vnext_fast:app --host 127.0.0.1 --port 8014 1>vnext-fast-api-8014.stdout.log 2>vnext-fast-api-8014.stderr.log
