import os
import sys
import pytest
import pandas as pd
from datetime import datetime, timedelta

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..", "training_scripts")
sys.path.append(os.path.abspath(parent_dir))

from data_ingestion import fetch_data


@pytest.fixture
def temp_output_dir(tmp_path):
    # Use pytest's tmp_path fixture for a temporary directory
    return tmp_path


def test_fetch_data_creates_csv(temp_output_dir):
    years = 2
    fetch_data(years_to_filter=years, output_dir=str(temp_output_dir))

    expected_file = temp_output_dir / "sp500_input.csv"
    assert expected_file.exists(), "Expected output CSV file was not created."

    df = pd.read_csv(
        expected_file, index_col=0, parse_dates=True
    )  # first column (Date) becomes the index

    # Ensure index is datetime
    df.index = pd.to_datetime(df.index, utc=True)

    # Get only the date (ignoring time) of the first entry in the index
    original_code_start_date = df.index[0].date()

    # validate the date range
    start_date_expected = datetime.today() - timedelta(days=years * 365)

    # Convert expected start date to a date (ignoring time)
    start_date_expected = start_date_expected.date()

    print("original_code_start_date", original_code_start_date)
    print("expected_start_date", start_date_expected)

    assert not df.empty, "CSV file is empty."
    assert original_code_start_date == start_date_expected, "Dates do not match"


def test_invalid_years_to_filter(temp_output_dir):
    with pytest.raises(ValueError, match="must be a positive integer"):
        fetch_data(years_to_filter=0, output_dir=str(temp_output_dir))
