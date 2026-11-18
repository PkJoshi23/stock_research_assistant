"""Streamlit Community Cloud entry point.

Cloud runs a single main file from the repo root; the actual app lives in
``src/ui/app.py``. ``runpy`` executes that file with the same semantics as
``streamlit run src/ui/app.py``, so there is exactly one implementation and no
duplicated UI code to drift.

Locally you can still run either:
    streamlit run streamlit_app.py
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "src" / "ui" / "app.py"), run_name="__main__")
