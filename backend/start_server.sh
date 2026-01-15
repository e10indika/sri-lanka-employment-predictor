#!/bin/bash
# Start the FastAPI backend server

echo "Starting FastAPI Backend Server..."
echo "API Documentation will be available at: http://localhost:8000/docs"
echo "ReDoc Documentation will be available at: http://localhost:8000/redoc"
echo ""

# Activate virtual environment if it exists
if [ -d "../venv" ]; then
    source ../venv/bin/activate
elif [ -d "$HOME/venvs/py3kernel" ]; then
    source $HOME/venvs/py3kernel/bin/activate
fi

# Run the server
cd api
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
