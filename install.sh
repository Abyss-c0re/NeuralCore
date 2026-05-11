#!/bin/bash
# NeuralCore - Clean uv editable install script
# Keeps framework separate from client tools/workflows

set -e  # Exit on error

echo "=== NeuralCore Framework Installation ==="

# 1. Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating new uv virtual environment..."
    uv venv --python 3.12  # or 3.14+ as per pyproject.toml
else
    echo "Using existing .venv"
fi

# 2. Activate the environment (for the script session)
source .venv/bin/activate

# 3. Install NeuralCore in editable mode
echo "Installing NeuralCore in editable mode (-e)..."
uv pip install -e .

echo "✅ Installation completed successfully!"

# 4. Quick verification
echo "Running import test..."
python -c "
import neuralcore
print('✅ NeuralCore imported successfully')
print('Version:', neuralcore.__version__ if hasattr(neuralcore, '__version__') else 'dev')
print('Location:', neuralcore.__file__)
"

echo ""
echo "Next steps:"
echo "   uv run your_client_script.py"
echo "   Or add to your pyproject.toml: uv add -e ."