"""Root entry point for Streamlit Community Cloud.

Streamlit Cloud expects the app at the repo root. This file
bootstraps the actual Orbit app from the orbit/ directory.
"""
import os
import sys
import runpy

# Add project root and orbit dir to path
_root = os.path.dirname(os.path.abspath(__file__))
for _p in (_root, os.path.join(_root, "orbit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Disable tqdm before anything else imports it
os.environ["TQDM_DISABLE"] = "1"

# Run the actual app
runpy.run_module("app", run_name="__main__", alter_sys=True)
