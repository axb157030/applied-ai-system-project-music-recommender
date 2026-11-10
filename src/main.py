"""
Console entry point.

    python -m src.main                  talk to the recommender in the terminal
    streamlit run app.py                the same recommender with a web UI

Both use the same Interview class (src/interview.py), so they behave the same way.
"""

import sys

from .interview import main

if __name__ == "__main__":
    # Optional arguments: path to the songs CSV, path to the docs folder.
    main(*sys.argv[1:3])
