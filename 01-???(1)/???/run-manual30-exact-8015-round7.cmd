@echo off
cd /d D:\lanqun-site\ragv6-standalone\retrieval-service-vnext
if exist experiments\manual30-exact-8015-round7.exit del experiments\manual30-exact-8015-round7.exit
python experiments\fast_latency_benchmark.py --base-url http://127.0.0.1:8015 --ids q19-fridge-deodorizer,q21-fridge-cleaning,q32-camera-computer-charge,q44-senseiq,q45-brushpacer,q46-brushsync --output experiments\manual30-exact-8015-round7.json > experiments\manual30-exact-8015-round7.log 2>&1
echo %ERRORLEVEL%> experiments\manual30-exact-8015-round7.exit
