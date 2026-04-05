@echo off
:: PageIndex API Server Startup Script
:: Starts the FastAPI backend on port 7777

echo.
echo =====================================================
echo  PageIndex RAG API Server
echo =====================================================
echo.

:: Check for .env file
if not exist ".env" (
    echo [WARNING] No .env file found!
    echo Please copy .env.example to .env and add your API key.
    echo.
    echo Example:
    echo   copy .env.example .env
    echo   Then edit .env and add your GEMINI_API_KEY or OPENAI_API_KEY
    echo.
    pause
    exit /b 1
)

echo [INFO] Starting PageIndex API on http://localhost:7777
echo [INFO] API docs available at http://localhost:7777/docs
echo.

:: Activate venv and start server with 16 workers
call venv\Scripts\activate
uvicorn api_server:app --host 0.0.0.0 --port 7777 --workers 16
