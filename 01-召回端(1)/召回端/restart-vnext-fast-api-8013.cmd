@echo off
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:"127.0.0.1:8013 .*LISTENING"') do (
  echo STOPPING_FAST_PID=%%P
  taskkill /PID %%P /F
)
timeout /t 1 /nobreak >nul
call D:\lanqun-site\ragv6-standalone\retrieval-service-vnext\start-vnext-fast-api-8013.cmd
