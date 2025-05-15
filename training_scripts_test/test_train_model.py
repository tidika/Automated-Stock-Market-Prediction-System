import os
import sys
import pytest
import pandas as pd
import xgboost as xgb

# Import from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..", "training_scripts")
sys.path.append(parent_dir)

from train_model import load_data, train_model


@pytest.fixture
def sample_data():
    """Creates a sample DataFrame for testing."""
    df = pd.DataFrame(
        {
            "f1": range(200),
            "f2": range(100, 300),
            "f3": [i * 2 for i in range(200)],
            "Target": [1 if i % 2 == 0 else 0 for i in range(200)],
        }
    )
    return df


def test_train_model_saves_file(sample_data, tmp_path):
    """Test that train_model creates a model.xgb file."""
    features = ["f1", "f2", "f3"]
    model_dir = tmp_path / "model"

    train_model(sample_data, features, str(model_dir))

    model_path = model_dir / "model.xgb"
    assert model_path.exists()

    # load model to verify it's a valid XGBoost model
    loaded_model = xgb.Booster()
    loaded_model.load_model(str(model_path))
    assert isinstance(loaded_model, xgb.Booster)


def test_load_data(tmp_path):
    """Test that load_data reads a CSV correctly."""
    dummy_csv = tmp_path / "dummy.csv"
    df = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6], "Target": [0, 1, 0]})
    df.to_csv(dummy_csv, index=False)

    loaded_df = load_data(str(dummy_csv))
    pd.testing.assert_frame_equal(df, loaded_df)
