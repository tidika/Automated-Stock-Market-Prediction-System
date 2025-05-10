import os
import sys
import pytest
import pandas as pd
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory

# Import from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..", "training_scripts")
sys.path.append(parent_dir)

from data_processing import (
    load_data,
    add_target_column,
    add_features,
    save_data,
    process_data,
)


@pytest.fixture
def sample_data():
    # Generate mock stock data
    dates = pd.date_range(end=datetime.today(), periods=10)
    df = pd.DataFrame(
        {
            "Date": dates,
            "Close": [100 + i for i in range(10)],
            "Open": [99 + i for i in range(10)],
            "High": [101 + i for i in range(10)],
            "Low": [98 + i for i in range(10)],
            "Volume": [1000 + i * 10 for i in range(10)],
        }
    )
    # df.set_index("Date", inplace=True)
    return df


def test_load_data_creates_dataframe(tmp_path):
    # Save dummy data
    csv_path = tmp_path / "sample.csv"
    df = pd.DataFrame({"Close": [1, 2, 3]})
    df.to_csv(csv_path, index=False)

    # Test loading
    loaded = load_data(str(csv_path))
    assert isinstance(loaded, pd.DataFrame)
    assert not loaded.empty


def test_add_target_column_creates_target(sample_data):
    df = add_target_column(sample_data.copy())
    assert "Target" in df.columns
    assert df["Target"].isin([0, 1]).all()


def test_add_features_creates_new_columns(sample_data):
    df = add_target_column(sample_data.copy())
    processed, feature_data = add_features(df, [2])
    assert any(col.startswith("Close_Ratio") for col in processed.columns)
    assert any(col.startswith("Trend") for col in processed.columns)
    assert isinstance(feature_data, pd.DataFrame)


def test_save_data_creates_files(sample_data):
    df = add_target_column(sample_data.copy())
    processed, feature_data = add_features(df, [2])

    with TemporaryDirectory() as tmpdir:
        save_data(processed, feature_data, tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "train.csv"))
        assert os.path.exists(os.path.join(tmpdir, "features.csv"))


def test_process_data_end_to_end(tmp_path):
    # Prepare dummy input CSV
    data = pd.DataFrame(
        {
            "Date": pd.date_range(start="2020-01-01", periods=10),
            "Close": [100 + i for i in range(10)],
            "Open": [100 + i for i in range(10)],
            "High": [100 + i for i in range(10)],
            "Low": [100 + i for i in range(10)],
            "Volume": [1000 + i for i in range(10)],
        }
    )
    input_path = tmp_path / "input.csv"
    data.to_csv(input_path, index=False)

    # Run processing
    process_data(str(input_path), str(tmp_path), [2, 5])

    # Check outputs
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "features.csv").exists()

    df = pd.read_csv(tmp_path / "train.csv")
    assert not df.empty
