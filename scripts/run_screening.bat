@echo off
cd /d "%~dp0.."

REM --- EDIT: set to your shared venv python.exe if you want ---
REM set PYTHON_EXE=D:\PycharmProjects\DeepLearning-env\Scripts\python.exe
set PYTHON_EXE=D:\PycharmProjects\DeepLearning-env\Scripts\python.exe

"%PYTHON_EXE%" src\predict.py --config configs\screening.yaml

REM ranking step (optional; will work after you define ranking.yaml and prices)
REM "%PYTHON_EXE%" src\rank.py --config configs\ranking.yaml

pause