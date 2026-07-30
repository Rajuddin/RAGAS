@echo off
REM -------------------------------------------------------------------------------
REM run_app.bat
REM One-shot launcher for the Streamlit UI. Creates the virtual environment and
REM installs all dependencies automatically on first run, then starts the app.
REM -------------------------------------------------------------------------------

cd /d "%~dp0"

echo.
echo ================================================
echo   RAGAS Evaluation - Streamlit UI
echo ================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.9+.
    pause
    exit /b 1
)
python --version

REM Virtual environment
if not exist ".venv" (
    echo.
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

REM Install dependencies
echo.
echo Installing dependencies...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

REM Config check
if not exist "config.yaml" (
    echo.
    echo No config.yaml found - creating one from config.yaml.temp.
    echo Fill in your LLM/API keys ^(use the sidebar in the app, or edit the file directly^).
    copy config.yaml.temp config.yaml >nul
)

REM Launch
REM The Allure CLI ^(used by the in-app "Open Allure Report" button^) is downloaded
REM automatically into .tools\ the first time the button is clicked -- no separate
REM install step needed here, only a Java runtime on the machine.
echo.
echo Starting Streamlit app...
echo.
REM Invoke this venv's own interpreter directly, not the bare "streamlit" command --
REM if a differently-located venv's Scripts dir is earlier on PATH (or this .venv was
REM ever copied from elsewhere), "streamlit" can resolve to a launcher stub that
REM silently runs a different venv's Python, leaving two servers fighting over the
REM same port with mismatched code.
".venv\Scripts\python.exe" -m streamlit run app.py
pause
