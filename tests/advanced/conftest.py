import sys
from pathlib import Path

# Ensure NeuralCore src and test tools are on sys.path
_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))
