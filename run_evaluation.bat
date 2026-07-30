@echo off
REM -------------------------------------------------------------------------------
REM run_evaluation.bat
REM Install dependencies and execute RAGAS evaluation with Allure reporting.
REM -------------------------------------------------------------------------------

cd /d "%~dp0"

echo.
echo ================================================
echo   RAGAS Evaluation Framework
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
    echo Edit config.yaml with your real API keys before running again.
    copy config.yaml.temp config.yaml >nul
)

REM Create allure-results dir
if not exist "allure-results" mkdir allure-results

REM Run pytest
echo.
echo Running RAGAS evaluation...
echo.
pytest tests\ --alluredir=allure-results -v --tb=short %*

set PYTEST_EXIT=%errorlevel%

REM Generate Allure report if available
allure --version >nul 2>&1
if not errorlevel 1 (
    echo.
    echo Generating Allure report...
    allure generate allure-results --clean -o allure-report
    echo.
    echo Report saved to: %cd%\allure-report\index.html
    allure open allure-report
)

echo.
echo ================================================
if %PYTEST_EXIT%==0 (
    echo   Evaluation COMPLETE - all tests passed
) else (
    echo   Evaluation COMPLETE - some tests failed
)
echo ================================================
echo.
pause
exit /b %PYTEST_EXIT%
