@echo off
cd /d D:\lanqun-site\ragv6-standalone\retrieval-service-vnext
if exist experiments\manual30-exact-8015-round4.exit del experiments\manual30-exact-8015-round4.exit
python experiments\fast_latency_benchmark.py --base-url http://127.0.0.1:8015 --ids q16-fridge-safety,q17-fridge-panel,q21-fridge-cleaning,q26-camera-memory-card,q27-camera-flash,q32-camera-computer-charge,q34-jetski-front-storage,q35-jetski-glove-compartment,q36-jetski-sponson,q38-jetski-boarding,q42-toothbrush-cleaning,q43-toothbrush-battery,q47-toothbrush-storage,q48-washer-safety,q49-washer-rinsing --output experiments\manual30-exact-8015-round4.json > experiments\manual30-exact-8015-round4.log 2>&1
echo %ERRORLEVEL%> experiments\manual30-exact-8015-round4.exit
