@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SOURCE_DIR=D:\lanqun-site\ragv6-standalone\retrieval-service-vnext"
set "BACKUP_ROOT=D:\lanqun-site\ragv6-standalone\backups"
for /f "skip=1 tokens=1 delims=." %%T in ('wmic os get localdatetime') do if not defined BACKUP_TS set "BACKUP_TS=%%T"
set "BACKUP_TS=!BACKUP_TS:~0,8!T!BACKUP_TS:~8,6!"
set "BACKUP_DIR=!BACKUP_ROOT!\retrieval-service-vnext-fast-!BACKUP_TS!"

if exist "!BACKUP_DIR!" (
  echo FAST_BACKUP_STATUS=REFUSED_EXISTING
  exit /b 2
)

mkdir "!BACKUP_DIR!\experiments" || exit /b 3

copy /b /y "!SOURCE_DIR!\api_server_vnext_fast.py" "!BACKUP_DIR!\api_server_vnext_fast.py" >nul || exit /b 4
copy /b /y "!SOURCE_DIR!\start-vnext-fast-api-8013.cmd" "!BACKUP_DIR!\start-vnext-fast-api-8013.cmd" >nul || exit /b 5
copy /b /y "!SOURCE_DIR!\run-vnext-fast-api-8013.cmd" "!BACKUP_DIR!\run-vnext-fast-api-8013.cmd" >nul || exit /b 6
copy /b /y "!SOURCE_DIR!\restart-vnext-fast-api-8013.cmd" "!BACKUP_DIR!\restart-vnext-fast-api-8013.cmd" >nul || exit /b 7
copy /b /y "!SOURCE_DIR!\experiments\evidence_coverage_fast.py" "!BACKUP_DIR!\experiments\evidence_coverage_fast.py" >nul || exit /b 8
copy /b /y "!SOURCE_DIR!\experiments\fast_latency_benchmark.py" "!BACKUP_DIR!\experiments\fast_latency_benchmark.py" >nul || exit /b 9
copy /b /y "!SOURCE_DIR!\experiments\fast-latency-full.json" "!BACKUP_DIR!\experiments\fast-latency-full.json" >nul || exit /b 10

(
  echo Backup type: vnext-fast delta bundle
  echo Base full snapshot: D:\lanqun-site\ragv6-standalone\backups\retrieval-service-vnext-20260803T084904
  echo Service endpoint: http://127.0.0.1:8013/vnext/chat
  echo Production service changed: NO
  echo Original vnext 8012 changed: NO
) > "!BACKUP_DIR!\FAST_BACKUP_INFO.txt"

echo FAST_BACKUP_STATUS=SUCCESS
echo FAST_BACKUP_PATH=!BACKUP_DIR!
certutil -hashfile "!BACKUP_DIR!\experiments\evidence_coverage_fast.py" SHA256
certutil -hashfile "!BACKUP_DIR!\experiments\fast-latency-full.json" SHA256
exit /b 0
