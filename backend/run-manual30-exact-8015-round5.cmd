@echo off
cd /d D:\lanqun-site\ragv6-standalone\retrieval-service-vnext
if exist experiments\manual30-exact-8015-round5.exit del experiments\manual30-exact-8015-round5.exit
python experiments\fast_latency_benchmark.py --base-url http://127.0.0.1:8015 --ids q16-fridge-safety,q27-camera-flash,q31-camera-command-dial,q36-jetski-sponson,q38-jetski-boarding,q43-toothbrush-battery,q48-washer-safety --output experiments\manual30-exact-8015-round5.json > experiments\manual30-exact-8015-round5.log 2>&1
echo %ERRORLEVEL%> experiments\manual30-exact-8015-round5.exit
