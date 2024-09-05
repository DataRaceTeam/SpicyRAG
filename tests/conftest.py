import os
import sys

import pandas as pd
import pytest


def pytest_configure():
    # Add directories to the Python path
    base_dirs = ["/app/"]
    pd.options.mode.chained_assignment = None  # default='warn'

    # Add all subdirectories under base_dir to the Python path
    for base_dir in base_dirs:
        for root, _, _ in os.walk(base_dir):
            sys.path.insert(0, root)


@pytest.fixture
def text_hmao_npa_df():
    with open("data/hmao_npa.txt") as file:
        raw_text = file.read()

    raw_docs = raw_text.split("\n")
    index = [i for i in range(len(raw_docs)) if i % 2 == 0]

    docs_df = (
        pd.DataFrame({"index": range(len(raw_docs)), "document": raw_docs})
        .iloc[index]
        .reset_index()
        .drop(columns=["index", "level_0"])
    )

    return docs_df
