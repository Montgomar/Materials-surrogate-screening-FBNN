@echo off
cd /d "%~dp0.."

REM --- Use the shared deep learning venv Python (EDIT THIS PATH) ---
set PYTHON_EXE=D:\PycharmProjects\DeepLearning-env\Scripts\python.exe
@REM set PYTHON_EXE=C:\envs\deep_env\Scripts\python.exe

REM Safety check
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python interpreter not found: %PYTHON_EXE%
  echo Please edit scripts\run_train.bat and set PYTHON_EXE to your venv python.exe
  pause
  exit /b 1
)

REM Install deps into that venv (should be mostly "Requirement already satisfied")
@REM "%PYTHON_EXE%" -m pip install -r requirements.txt

REM Run training
"%PYTHON_EXE%" src\train.py --config configs\train.yaml

pause