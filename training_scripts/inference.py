# Function to load the model during inference
import os
import xgboost as xgb
def model_fn(model_dir):
    """Load model from the directory where it was saved during training."""
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "model.xgb"))
    return model