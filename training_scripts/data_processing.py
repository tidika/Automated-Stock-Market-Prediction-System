import os
import argparse
import logging
import pandas as pd


def load_data(input_path):
    """Loads the raw data from the input path.

    Args:
        input_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    logging.info(f"Loading data from {input_path}")
    try:
        return pd.read_csv(input_path, parse_dates=True)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise


def add_target_column(data):
    """Adds a target column indicating if the next day's close price is higher.

    Args:
        data (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with the target column added.
    """
    logging.info("Adding target column")
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    return data


def add_features(data, horizons):
    """Adds rolling averages and trends as new features.

    Args:
        data (pd.DataFrame): Input dataframe.
        horizons (list): List of horizons for feature generation.

    Returns:
        pd.DataFrame: Dataframe with new features added.
    """
    logging.info("Adding new features")
    new_predictors = []

    for horizon in horizons:
        logging.info(f"Processing horizon: {horizon}")
        rolling_averages = data.rolling(horizon).mean()

        ratio_column = f"Close_Ratio_{horizon}"
        trend_column = f"Trend_{horizon}"

        data[ratio_column] = data["Close"] / rolling_averages["Close"]
        data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]

        new_predictors.extend([ratio_column, trend_column])
        features = data.columns[-10:].to_list()
        feature_data = data[features]

    return data, feature_data


def save_data(data,feature_data,output_dir):
    """Saves the processed data to the output directory.

    Args:
        data (pd.DataFrame): Processed dataframe.
        output_dir (str): Directory to save the output.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "train.csv")
    output_feature = os.path.join(output_dir, "features.csv")
    logging.info(f"Saving processed data to {output_file}")
    data.to_csv(output_file, index=False)
    feature_data.to_csv(output_feature, index=False) #this to be used for model monitoring


def process_data(input_path, output_dir, horizons):
    """Processes the raw data and saves it in the specified output directory.

    Args:
        input_path (str): Path to the input CSV file.
        output_dir (str): Directory to save the processed data.
        horizons (list): List of horizons for feature generation.
    """
    # Load the data
    data = load_data(input_path)

    # Add target column
    data = add_target_column(data)

    # Add new features
    data, feature_data = add_features(data, horizons)

    # Drop rows with missing values
    data = data.dropna()
    logging.info(f"Data after cleaning: {data.shape[0]} rows")

    # Save the processed data
    save_data(data, feature_data, output_dir)


def main():
    """
    Main entry point for the script. Handles argument parsing and orchestrates data processing and generation of new features.
    """

    # Parse data input path and output directory as command-line arguments
    parser = argparse.ArgumentParser(
        description="Train and backtest a RandomForest model."
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input CSV file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed data.",
    )
    args = parser.parse_args()

    # List of horizons for feature generation
    horizons = [2, 5, 60, 250, 1000]

    # Process the data
    process_data(args.input_path, args.output_dir, horizons)


if __name__ == "__main__":
    main()
