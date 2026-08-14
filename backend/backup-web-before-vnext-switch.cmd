@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SITE_ROOT=D:\lanqun-site"
set "WEB_ROOT=D:\lanqun-site\ragv6-standalone\web-client"
set "VNEXT_ROOT=D:\lanqun-site\ragv6-standalone\retrieval-service-vnext"
set "BACKUP_ROOT=D:\lanqun-site\ragv6-standalone\backups"
for /f "skip=1 tokens=1 delims=." %%T in ('wmic os get localdatetime') do if not defined BACKUP_TS set "BACKUP_TS=%%T"
set "BACKUP_TS=!BACKUP_TS:~0,8!T!BACKUP_TS:~8,6!"
set "BACKUP_DIR=!BACKUP_ROOT!\web-before-vnext-switch-!BACKUP_TS!"

if exist "!BACKUP_DIR!" (
  echo WEB_BACKUP_STATUS=REFUSED_EXISTING
  exit /b 2
)

mkdir "!BACKUP_DIR!\web-client" || exit /b 3
mkdir "!BACKUP_DIR!\retrieval-service-vnext" || exit /b 4
copy /b /y "!WEB_ROOT!\server.js" "!BACKUP_DIR!\web-client\server.js" >nul || exit /b 5
copy /b /y "!SITE_ROOT!\run-raysource-web.cmd" "!BACKUP_DIR!\run-raysource-web.cmd" >nul || exit /b 6
copy /b /y "!SITE_ROOT!\restart-raysource-web.ps1" "!BACKUP_DIR!\restart-raysource-web.ps1" >nul || exit /b 7
copy /b /y "!VNEXT_ROOT!\api_server_vnext_fast.py" "!BACKUP_DIR!\retrieval-service-vnext\api_server_vnext_fast.py" >nul || exit /b 8

(
  echo Backup purpose: before switching the public web chat from legacy 8011 to vnext-fast 8013
  echo Legacy backend preserved: http://127.0.0.1:8011
  echo Vnext backend preserved: http://127.0.0.1:8013
  echo Web service: http://127.0.0.1:3011
  echo Secrets copied: NO
) > "!BACKUP_DIR!\BACKUP_INFO.txt"

echo WEB_BACKUP_STATUS=SUCCESS
echo WEB_BACKUP_PATH=!BACKUP_DIR!
certutil -hashfile "!BACKUP_DIR!\web-client\server.js" SHA256
certutil -hashfile "!BACKUP_DIR!\retrieval-service-vnext\api_server_vnext_fast.py" SHA256
exit /b 0
