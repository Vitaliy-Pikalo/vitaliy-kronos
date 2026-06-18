@echo off
REM Vitaliy Kronos Signal Bot -- daily launcher
REM Called by Windows Task Scheduler at 12:55 UTC (8:55 AM ET)

set PYTHONUTF8=1
cd /d C:\vk
python src\signal_bot.py >> C:\vk\logs\bot.log 2>&1
