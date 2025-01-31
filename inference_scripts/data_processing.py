# data_processing.py
import pandas as pd
import numpy as np
import argparse
import os

def preprocess_data(sp500):
    # Extract features for prediction
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    
    horizons = [2,5,60,250,1000]
    new_predictors = []
    
    for horizon in horizons:
        rolling_averages = sp500.rolling(horizon).mean()
        
        ratio_column = f"Close_Ratio_{horizon}"
        sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
        
        trend_column = f"Trend_{horizon}"
        sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
        
        new_predictors+= [ratio_column, trend_column]

    sp500data = sp500.dropna()
    features = sp500data.columns[-10:].to_list()
    data = sp500data[features]
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed data.")
    args = parser.parse_args()

    input_path = os.path.join(args.input_dir, "sp500_input.csv")
    output_path = os.path.join(args.output_dir, "sp500_processed.csv")

    data = pd.read_csv(input_path)
    processed_data = preprocess_data(data)

    os.makedirs(args.output_dir, exist_ok=True)
    processed_data.to_csv(output_path, index=False)