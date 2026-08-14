@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SOURCE_DIR=D:\lanqun-site\ragv6-standalone\retrieval-service-vnext"
set "BACKUP_DIR=D:\lanqun-site\ragv6-standalone\backups\retrieval-service-vnext-20260803T084904"
set "VERIFY_FAILED=0"

if not exist "%BACKUP_DIR%\BACKUP_INFO.txt" (
  echo BACKUP_INFO=MISS
  exit /b 2
)
echo BACKUP_INFO=OK

for %%F in (
  "api_server_vnext.py"
  "experiments\evidence_coverage_vnext.py"
  "data\retrieval_chunks.json"
  "data\section_chunks.json"
  "data\index\dense.faiss"
  "data\index\retrieval_index.pkl"
) do (
  echo VERIFY_FILE=%%~F
  if not exist "%SOURCE_DIR%\%%~F" (
    echo SOURCE_FILE=MISS
    set "VERIFY_FAILED=1"
  ) else if not exist "%BACKUP_DIR%\%%~F" (
    echo BACKUP_FILE=MISS
    set "VERIFY_FAILED=1"
  ) else (
    fc /b "%SOURCE_DIR%\%%~F" "%BACKUP_DIR%\%%~F" >nul
    if errorlevel 1 (
      echo BYTE_MATCH=NO
      set "VERIFY_FAILED=1"
    ) else (
      echo BYTE_MATCH=YES
      certutil -hashfile "%BACKUP_DIR%\%%~F" SHA256
    )
  )
)

echo SERVICE_8011_BEGIN
netstat -ano | findstr /R /C:"127.0.0.1:8011 .*LISTENING"
if errorlevel 1 set "VERIFY_FAILED=1"
echo SERVICE_8011_END

echo SERVICE_8012_BEGIN
netstat -ano | findstr /R /C:"127.0.0.1:8012 .*LISTENING"
if errorlevel 1 set "VERIFY_FAILED=1"
echo SERVICE_8012_END

if "%VERIFY_FAILED%"=="0" (
  echo BACKUP_VERIFY=SUCCESS
  exit /b 0
)

echo BACKUP_VERIFY=FAILED
exit /b 1
