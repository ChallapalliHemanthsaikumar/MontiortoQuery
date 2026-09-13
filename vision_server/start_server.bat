@echo off
REM Wildlife Vision Server — start/stop/status
REM Usage:
REM   start_server.bat start    — start server in background
REM   start_server.bat stop     — stop the background server
REM   start_server.bat status   — check if server is running
REM   start_server.bat logs     — tail the log file

set SCRIPT_DIR=%~dp0
set PID_FILE=%SCRIPT_DIR%server.pid
set LOG_FILE=%SCRIPT_DIR%server.log

if "%1"=="start" goto start
if "%1"=="stop" goto stop
if "%1"=="status" goto status
if "%1"=="logs" goto logs
if "%1"=="restart" goto restart

echo Usage: start_server.bat [start^|stop^|status^|logs^|restart]
goto end

:start
echo Starting Wildlife Vision Server...
if exist "%PID_FILE%" (
    for /f %%i in (%PID_FILE%) do (
        tasklist /fi "PID eq %%i" 2>nul | find "%%i" >nul
        if not errorlevel 1 (
            echo Server already running (PID %%i^)
            goto end
        )
    )
)

start /b "" python -u "%SCRIPT_DIR%server.py" > "%LOG_FILE%" 2>&1
timeout /t 2 /nobreak >nul

REM Find the python process running our server
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%server.py%%' and name='python.exe'" get processid /format:list 2^>nul ^| find "="') do (
    echo %%a> "%PID_FILE%"
    echo Server started (PID %%a^)
    echo Log file: %LOG_FILE%
    goto end
)
echo Server started. Check logs: %LOG_FILE%
goto end

:stop
if not exist "%PID_FILE%" (
    echo No PID file found. Killing any python server.py processes...
    for /f "tokens=2" %%a in ('wmic process where "commandline like '%%server.py%%' and name='python.exe'" get processid /format:list 2^>nul ^| find "="') do (
        taskkill /pid %%a /f >nul 2>&1
        echo Killed process %%a
    )
    goto end
)
for /f %%i in (%PID_FILE%) do (
    taskkill /pid %%i /f >nul 2>&1
    echo Server stopped (PID %%i^)
)
del "%PID_FILE%" >nul 2>&1
goto end

:status
echo Checking for vision server processes...
wmic process where "commandline like '%%server.py%%' and name='python.exe'" get processid,commandline 2>nul | find "server.py"
if errorlevel 1 (
    echo Server is NOT running.
) else (
    echo Server is RUNNING.
)
goto end

:logs
if exist "%LOG_FILE%" (
    type "%LOG_FILE%"
) else (
    echo No log file found.
)
goto end

:restart
call :stop
timeout /t 2 /nobreak >nul
call :start
goto end

:end
