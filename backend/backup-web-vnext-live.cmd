@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SITE_ROOT=D:\lanqun-site"
set "WEB_ROOT=D:\lanqun-site\ragv6-standalone\web-client"
set "VNEXT_ROOT=D:\lanqun-site\ragv6-standalone\retrieval-service-vnext"
set "BACKUP_ROOT=D:\lanqun-site\ragv6-standalone\backups"
for /f "skip=1 tokens=1 delims=." %%T in ('wmic os get localdatetime') do if not defined BACKUP_TS set "BACKUP_TS=%%T"
set "BACKUP_TS=!BACKUP_TS:~0,8!T!BACKUP_TS:~8,6!"
set "BACKUP_DIR=!BACKUP_ROOT!\web-vnext-live-!BACKUP_TS!"

if exist "!BACKUP_DIR!" (
  echo LIVE_BACKUP_STATUS=REFUSED_EXISTING
  exit /b 2
)

mkdir "!BACKUP_DIR!\web-client\backend-switch" || exit /b 3
mkdir "!BACKUP_DIR!\retrieval-service-vnext\experiments" || exit /b 4
copy /b /y "!WEB_ROOT!\server.js" "!BACKUP_DIR!\web-client\server.js" >nul || exit /b 5
copy /b /y "!SITE_ROOT!\run-raysource-web.cmd" "!BACKUP_DIR!\run-raysource-web.cmd" >nul || exit /b 6
copy /b /y "!SITE_ROOT!\switch-raysource-backend.cmd" "!BACKUP_DIR!\switch-raysource-backend.cmd" >nul || exit /b 7
copy /b /y "!WEB_ROOT!\backend-switch\backend-legacy.json" "!BACKUP_DIR!\web-client\backend-switch\backend-legacy.json" >nul || exit /b 8
copy /b /y "!WEB_ROOT!\backend-switch\backend-vnext.json" "!BACKUP_DIR!\web-client\backend-switch\backend-vnext.json" >nul || exit /b 9
copy /b /y "!WEB_ROOT!\backend-switch\backend-active.json" "!BACKUP_DIR!\web-client\backend-switch\backend-active.json" >nul || exit /b 10
copy /b /y "!WEB_ROOT!\backend-switch\activate-web-code.ps1" "!BACKUP_DIR!\web-client\backend-switch\activate-web-code.ps1" >nul || exit /b 11
copy /b /y "!VNEXT_ROOT!\api_server_vnext_fast.py" "!BACKUP_DIR!\retrieval-service-vnext\api_server_vnext_fast.py" >nul || exit /b 12
copy /b /y "!VNEXT_ROOT!\experiments\evidence_coverage_fast.py" "!BACKUP_DIR!\retrieval-service-vnext\experiments\evidence_coverage_fast.py" >nul || exit /b 13

(
  echo Backup purpose: deployed reversible web switch
  echo Active backend: vnext-fast at http://127.0.0.1:8013
  echo Legacy backend retained: http://127.0.0.1:8011
  echo Switch command: D:\lanqun-site\switch-raysource-backend.cmd legacy^|vnext^|status
  echo Secrets copied: NO
) > "!BACKUP_DIR!\BACKUP_INFO.txt"

echo LIVE_BACKUP_STATUS=SUCCESS
echo LIVE_BACKUP_PATH=!BACKUP_DIR!
certutil -hashfile "!BACKUP_DIR!\web-client\server.js" SHA256
certutil -hashfile "!BACKUP_DIR!\retrieval-service-vnext\api_server_vnext_fast.py" SHA256
exit /b 0
