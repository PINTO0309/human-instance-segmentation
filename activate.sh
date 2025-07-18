#!/bin/bash
# Activation script for human-edge-detection project
# This script activates the virtual environment and sets up LD_LIBRARY_PATH

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if virtual environment exists
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Virtual environment not found. Running 'uv sync' first..."
    uv sync
fi

# Activate virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Set LD_LIBRARY_PATH for TensorRT
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SCRIPT_DIR/.venv/lib/python3.10/site-packages/tensorrt_libs

echo "✓ Virtual environment activated"
echo "✓ LD_LIBRARY_PATH updated for TensorRT"
echo ""
echo "You can now run commands like:"
echo "  uv run python main.py"
echo "  uv run python validate.py checkpoints/best_model.pth"