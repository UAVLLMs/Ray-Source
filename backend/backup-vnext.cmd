@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SOURCE_DIR=D:\lanqun-site\ragv6-standalone\retrieval-service-vnext"
set "BACKUP_ROOT=D:\lanqun-site\ragv6-standalone\backups"

for /f "tokens=2 delims==" %%I in ('wmic os get LocalDateTime /value 2^>nul') do if not defined RAW_TS set "RAW_TS=%%I"
if not defined RAW_TS (
  echo BACKUP_ERROR=TIMESTAMP_UNAVAILABLE
  exit /b 20
)
set "BACKUP_TS=!RAW_TS:~0,8!T!RAW_TS:~8,6!"
set "BACKUP_DIR=!BACKUP_ROOT!\retrieval-service-vnext-!BACKUP_TS!"

if not exist "!SOURCE_DIR!\experiments\evidence_coverage_vnext.py" (
  echo BACKUP_ERROR=SOURCE_NOT_FOUND
  exit /b 21
)
if exist "!BACKUP_DIR!" (
  echo BACKUP_ERROR=TARGET_ALREADY_EXISTS
  echo BACKUP_PATH=!BACKUP_DIR!
  exit /b 22
)

if not exist "!BACKUP_ROOT!" mkdir "!BACKUP_ROOT!"
mkdir "!BACKUP_DIR!"
if errorlevel 1 (
  echo BACKUP_ERROR=TARGET_CREATE_FAILED
  exit /b 23
)

robocopy "!SOURCE_DIR!" "!BACKUP_DIR!" /E /COPY:DAT /DCOPY:DAT /R:2 /W:2 /XJ /NP /XD __pycache__ /XF *.pyc vnext-api-8012.log
set "ROBOCOPY_RC=!ERRORLEVEL!"
if !ROBOCOPY_RC! GEQ 8 (
  echo BACKUP_ERROR=ROBOCOPY_FAILED
  echo ROBOCOPY_RC=!ROBOCOPY_RC!
  echo BACKUP_PATH=!BACKUP_DIR!
  exit /b !ROBOCOPY_RC!
)

(
  echo Backup type: isolated retrieval-service-vnext snapshot
  echo Source: !SOURCE_DIR!
  echo Created local time: !BACKUP_TS!
  echo Production retrieval-service and port 8011 were not modified.
  echo Excluded: __pycache__, *.pyc, active vnext API log.
) > "!BACKUP_DIR!\BACKUP_INFO.txt"

echo BACKUP_STATUS=SUCCESS
echo BACKUP_PATH=!BACKUP_DIR!
echo ROBOCOPY_RC=!ROBOCOPY_RC!
exit /b 0
