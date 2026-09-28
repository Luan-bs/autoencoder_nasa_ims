"""pytest configuration — adds src/ to sys.path so tests import directly from src/."""
import sys
from pathlib import Path

# Ensure src/ is on the path for all test runs
sys.path.insert(0, str(Path(__file__).parent / "src"))
