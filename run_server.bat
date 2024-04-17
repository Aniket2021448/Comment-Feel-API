@echo off

rem Install dependencies from requirements.txt
pip install -r requirements.txt

rem Start the FastAPI server using uvicorn
uvicorn API:app
