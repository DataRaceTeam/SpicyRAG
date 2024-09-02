import os
import sys

import pandas as pd


def pytest_configure():
    # Add directories to the Python path
    base_dirs = ["/app/"]
    pd.options.mode.chained_assignment = None  # default='warn'

    # Add all subdirectories under base_dir to the Python path
    for base_dir in base_dirs:
        for root, _, _ in os.walk(base_dir):
            sys.path.insert(0, root)
