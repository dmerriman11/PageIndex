import sys
from pathlib import Path

# Make top-level engine modules (app_settings, model_catalog, …) and the
# pageindex package importable from tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
