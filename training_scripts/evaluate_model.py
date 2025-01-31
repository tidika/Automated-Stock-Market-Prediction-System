import argparse
import os
import pandas as pd
import json
from sklearn.metrics import precision_score
import logging
import tarfile
import xgboost as xgb


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    # Input and output paths
    parser.add_argument("--input-path", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save evaluation results")
    

    return parser.parse_args()

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

def extract_model(archive_path, extract_to):
    """Extract the model.tar.gz archive."""
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print(f"Extracted to: {extract_to}")

def load_xgboost_model(model_dir):
    """Load an XGBoost model from a tar.gz file."""
    archive_path = os.path.join(model_dir, "model.tar.gz")
    extract_to = os.path.join(model_dir, "extracted")
    
    # Step 1: Extract the tar.gz file
    extract_model(archive_path, extract_to)
    
    # Step 2: Locate the model file
    model_path = os.path.join(extract_to, "model.xgb")  
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Step 3: Load the model
    model = xgb.Booster()
    model.load_model(model_path)
    print("Model loaded successfully.")
    return model
    
def evaluate_model(data, features, model):
    """Evaluate the model on the test dataset."""
    
    x_test = data[features].iloc[-100:]
    y_test = data["Target"].iloc[-100:]
    preds = model.predict(xgb.DMatrix(x_test))
    preds[preds >=0.5] = 1
    preds[preds <0.5] = 0
    preds = pd.Series(preds, index=y_test.index, name="Predictions")
    combined = pd.concat([y_test, preds], axis=1)
    print("printing combined dataframe")
    print(combined)
    
    # Rename columns for clarity
    combined.columns = ["Target", "Predictions"]
    
    #Compute evaluation metrics
    score = precision_score(combined["Target"], combined["Predictions"])
    
    #Return metrics as a dictionary
    return {
        "precision":score,
    }
 
def save_results(metrics, output_dir):
    """Save the evaluation metrics to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "evaluation.json")

    # Save metrics as a JSON file
    with open(results_path, "w") as f:
        json.dump(metrics, f)
    print(f"Evaluation results saved to {results_path}")


def main():
    # Parse arguments
    args = parse_args()

    # Load data, model, and evaluate
    data = load_data(args.input_path)
    model = load_xgboost_model(args.model_path)
    features = data.columns[-10:].to_list()
    metrics = evaluate_model(data, features, model)

    # Print metrics and save to output
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    save_results(metrics, args.output_path)


if __name__ == "__main__":
    main()



