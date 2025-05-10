import os
import sys
import pytest
import pandas as pd
from datetime import datetime, timedelta

# Set up path to import script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..", "inference_scripts")
sys.path.append(parent_dir)

from data_processing import process_data


@pytest.fixture
def sample_sp500_data():
    """Creates a small sample SP500-like dataframe"""
    dates = pd.date_range(end=datetime.today(), periods=1100)
    data = {
        "Close": [3000 + i * 0.5 for i in range(1100)],
    }
    df = pd.DataFrame(data, index=dates)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Date"}, inplace=True)
    return df


def test_process_data_structure(sample_sp500_data):
    processed = process_data(sample_sp500_data.copy())
    print("processed data", processed.head())

    # Expect a single row
    assert processed.shape[0] == 1, "Processed data should return only the last row"

    # Expect 10 features
    assert processed.shape[1] == 10, "Processed data should have 10 columns"

    # Columns must match expected names
    expected_columns = [
        "Close_Ratio_2",
        "Trend_2",
        "Close_Ratio_5",
        "Trend_5",
        "Close_Ratio_60",
        "Trend_60",
        "Close_Ratio_250",
        "Trend_250",
        "Close_Ratio_1000",
        "Trend_1000",
    ]
    assert list(processed.columns) == expected_columns, "Feature columns do not match"


def test_process_data_nan_free(sample_sp500_data):
    processed = process_data(sample_sp500_data.copy())
    assert (
        not processed.isnull().values.any()
    ), "Processed data should not contain NaN values"
