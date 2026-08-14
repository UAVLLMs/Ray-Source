@echo off
cd /d D:\lanqun-site\ragv6-standalone\retrieval-service-vnext
if exist experiments\manual30-exact-8015-round8.exit del experiments\manual30-exact-8015-round8.exit
python experiments\fast_latency_benchmark.py --base-url http://127.0.0.1:8015 --ids q45-brushpacer --output experiments\manual30-exact-8015-round8.json > experiments\manual30-exact-8015-round8.log 2>&1
echo %ERRORLEVEL%> experiments\manual30-exact-8015-round8.exit
