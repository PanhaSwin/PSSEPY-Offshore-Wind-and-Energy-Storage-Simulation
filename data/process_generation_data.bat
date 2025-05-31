@echo off
cd /d C:\Users\panha\Desktop\data

:loop
cls
echo Running yankapoo.py...
python yankapoo.py

echo.
echo [Ctrl+C to interrupt and rerun] or [Close window to exit]
pause
goto loop
