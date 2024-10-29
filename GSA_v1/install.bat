@echo off
echo Installing required Python packages...

pip install numpy matplotlib sunpy scikit-image joblib EntropyHub

if %errorlevel% neq 0 (
    echo.
    echo There was an error during the installation of the packages.
    exit /b %errorlevel%
)

echo.
echo All packages installed successfully.
PAUSE

