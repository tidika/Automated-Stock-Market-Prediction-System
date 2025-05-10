import os
import sys
import argparse
from datetime import datetime, timedelta
from curl_cffi import requests
import yfinance as yf


def fetch_data(years_to_filter: int, output_dir: str) -> None:
    """
    Fetch historical S&P 500 market data, filter it based on the specified number of years,
    and save the data as a CSV file to the specified output directory.

    Args:
        years_to_filter (int): Number of years of historical data to filter.
        output_dir (str): Directory where the filtered S&P 500 data will be saved.

    Raises:
        ValueError: If `years_to_filter` is not a positive integer.
    """
    if years_to_filter <= 0:
        raise ValueError("The value for 'years_to_filter' must be a positive integer.")

    try:
        # To prevent rate limit error
        session = requests.Session(impersonate="chrome")
        ticker = yf.Ticker("...", session=session)

        print("Fetching historical data for the S&P 500...")
        sp500 = yf.Ticker("^GSPC").history(period="max")

        # Calculate the date range
        end_date = datetime.today()
        start_date = end_date - timedelta(days=int(years_to_filter) * 365)
        print(f"Filtering data from {start_date.date()} to {end_date.date()}...")

        # Filter data
        filtered_data = sp500.loc[start_date.strftime("%Y-%m-%d") :]

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save data to CSV
        filename = "sp500_input.csv"
        output_path = os.path.join(output_dir, filename)
        filtered_data.to_csv(output_path, index=True)
        print(f"Data saved successfully to: {output_path}")

    except Exception as e:
        print(f"An error occurred while fetching or saving data: {e}")
        sys.exit(1)


def main():
    """
    Main entry point for the script. Handles argument parsing and orchestrates the data fetching process.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Fetch and save historical S&P 500 market data."
    )
    parser.add_argument(
        "--years-to-filter",
        type=int,
        required=True,
        help="Number of historical years to filter.",
    )
    args = parser.parse_args()

    # Define output directory
    output_dir = "/opt/ml/processing/output"

    # Fetch and save S&P 500 data
    fetch_data(args.years_to_filter, output_dir)


if __name__ == "__main__":
    main()
