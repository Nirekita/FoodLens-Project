#!/bin/bash
# ============================================================
#  run.sh  —  Start NutriLens API + open the app
#  Usage: bash run.sh
# ============================================================

echo "🔍 Starting NutriLens..."
echo ""

# Check model files exist
if [ ! -f "food_scorer.pkl" ]; then
  echo "❌ food_scorer.pkl not found."
  echo "   Run colab_training.py on Google Colab first, then move the .pkl files here."
  exit 1
fi

# Start API in background
echo "✅ Starting API on http://localhost:8000"
uvicorn api.main:app --port 8000 --reload &
API_PID=$!

# Wait for API to be ready
sleep 2

# Open frontend in browser
echo "✅ Opening app in browser..."
open frontend/index.html

echo ""
echo "App is running."
echo "API docs: http://localhost:8000/docs"
echo "Press Ctrl+C to stop."
echo ""

# Keep running until Ctrl+C
wait $API_PID