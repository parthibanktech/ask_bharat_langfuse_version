"""
Run TruLens Dashboard
---------------------
Helper script to launch the TruLens dashboard for viewing evaluation results.

Usage:
    uv run python evaluation_tools/run_dashboard.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Try the newer trulens package structure
    from trulens.dashboard import run_dashboard
    from trulens.core import TruSession
    
    print("Starting TruLens Dashboard...")
    print("Dashboard will be available at http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    
    session = TruSession()
    run_dashboard(session)
    
except ImportError as e:
    try:
        # Try alternative import (older versions)
        from trulens_eval import Tru
        
        print("Starting TruLens Dashboard (legacy import)...")
        print("Dashboard will be available at http://localhost:8501")
        print("Press Ctrl+C to stop the dashboard")
        
        tru = Tru()
        tru.run_dashboard()
        
    except ImportError as e2:
        print("Error: Could not import TruLens dashboard.")
        print(f"Attempted imports failed: {e}, {e2}")
        print("\nPlease ensure TruLens is installed:")
        print("  uv sync --extra evaluation")
        print("  or")
        print("  uv pip install trulens-eval trulens-providers-openai")
        sys.exit(1)

