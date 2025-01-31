import logging
import argparse
import os
import joblib
import pandas as pd
import xgboost as xgb


def setup_logging():
    """Sets up logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )


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
    except FileNotFoundError:
        logging.error(f"File not found: {input_path}")
        raise
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise


def train_model(data, features, model_dir):
    """Trains the model, performs backtesting, and saves predictions and the model.

    Args:
        data (pd.DataFrame): Input data containing features and target.
        features (list): List of feature column names.
        model_dir (str): Directory to save the trained model.
    """
    # Create the model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    x_train = data[features].iloc[:-100]  # all data except the last 100

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "eta": 0.1,
    }

    # Convert training data to DMatrix
    dtrain = xgb.DMatrix(data=x_train, label=y_train)
    num_round = 100  # Number of boosting rounds
    model = xgb.train(params, dtrain, num_boost_round=num_round)

    # Save the model
    logging.info(f"Saving trained model to {model_dir}")
    model.save_model(os.path.join(model_dir, "model.xgb"))
    logging.info("Model training, prediction, and saving completed successfully.")


def main():
    """Main function to load data, train the model, and save the results."""
    setup_logging()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train Stock Market data using a Xgboost model."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR"),
        help="Directory to save model artifacts.",
    )
    args = parser.parse_args()

    input_path = "/opt/ml/input/data/train/train.csv"  # Path to the input CSV file.

    sp500data = load_data(input_path)
    features = sp500data.columns[-10:].to_list()
    train_model(sp500data, features, args.model_dir)


if __name__ == "__main__":
    main()
