"""Put benches/ on sys.path so tests import the bench modules the same way
the bench scripts do (`from common import ...`)."""

import sys
from pathlib import Path

BENCHES = Path(__file__).resolve().parent.parent / "benches"
if str(BENCHES) not in sys.path:
    sys.path.insert(0, str(BENCHES))
