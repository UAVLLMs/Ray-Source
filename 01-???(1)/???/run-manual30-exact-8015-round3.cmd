@echo off
cd /d D:\lanqun-site\ragv6-standalone\retrieval-service-vnext
if exist experiments\manual30-exact-8015-round3.exit del experiments\manual30-exact-8015-round3.exit
python experiments\fast_latency_benchmark.py --base-url http://127.0.0.1:8015 --ids q17-fridge-panel,q31-camera-command-dial,q33-jetski-indicators,q42-toothbrush-cleaning,q43-toothbrush-battery,q47-toothbrush-storage,q49-washer-rinsing,q50-washer-overflow-rinse --output experiments\manual30-exact-8015-round3.json > experiments\manual30-exact-8015-round3.log 2>&1
echo %ERRORLEVEL%> experiments\manual30-exact-8015-round3.exit
