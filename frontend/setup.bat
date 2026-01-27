@echo off
REM Setup script for the frontend (Windows)

echo Setting up sklearn-diagnose chatbot frontend...

REM Check if npm is installed
where npm >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: npm is not installed. Please install Node.js and npm first.
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
call npm install

echo.
echo Setup complete! You can now run:
echo   npm run dev    - Start development server
echo   npm run build  - Build for production
echo.
