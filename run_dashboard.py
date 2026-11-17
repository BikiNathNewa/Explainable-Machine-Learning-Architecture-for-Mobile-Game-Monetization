"""Launcher script for Maya's Pre-Production Greenlight Dashboard.
Runs the Flask application with development configuration.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.app import app


def main():
    parser = argparse.ArgumentParser(description="Launch Maya's Game Monetization XAI Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind (default: 5000)")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug reload mode")
    args = parser.parse_args()

    print("=" * 70)
    print("[*] MAYA'S PRE-PRODUCTION GREENLIGHT STUDIO (XAI SYSTEM)")
    print("=" * 70)
    print(f"[*] Server running on: http://{args.host}:{args.port}")
    print("[*] Model Architecture: XGBoost (100 estimators, max depth 4)")
    print("[*] Explainability: TreeSHAP (Deterministic) + DiCE Recourse")
    print("[*] Theme: Esports Light Mode (Luminous Porcelain & Cyan/Indigo HUD)")
    print("=" * 70)


    app.run(host=args.host, port=args.port, debug=not args.no_debug)


if __name__ == "__main__":
    main()
