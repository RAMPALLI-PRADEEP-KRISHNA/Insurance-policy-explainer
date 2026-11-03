"""Compatibility entry point bridging src/app.py to frontend/app.py.

Allows Streamlit Cloud or local execution via either:
    streamlit run src/app.py
or:
    streamlit run frontend/app.py
"""

import sys
from pathlib import Path

# Ensure project root is at the head of sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.app import (
    main,
    process_uploaded_policy,
    answer_policy_question,
    scan_policy_red_flags,
    compare_uploaded_policies,
    ProcessResult,
)

if __name__ == "__main__":
    main()
