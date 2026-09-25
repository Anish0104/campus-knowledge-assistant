import sys
from pathlib import Path

# Let tests import the project modules (search, generate, api, ...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
