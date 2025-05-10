import os
import sys
import json
import shutil
import pandas as pd
import pytest
import xgboost as xgb

# Import from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..", "training_scripts")
sys.path.append(parent_dir)

from tempfile import TemporaryDirectory
from evaluate_model import (
    load_data,
    extract_model,
    load_xgboost_model,
    evaluate_model,
    save_results,
)

# Replace `your_script_filename` with the actual name of the file you're testing (without `.py`)


@pytest.fixture
def sample_dataframe():
    df = pd.DataFrame(
        {
            "Feature1": range(200),
            "Feature2": range(200),
            "Feature3": range(200),
            "Feature4": range(200),
            "Feature5": range(200),
            "Feature6": range(200),
            "Feature7": range(200),
            "Feature8": range(200),
            "Feature9": range(200),
            "Feature10": range(200),
            "Target": [1 if i % 2 == 0 else 0 for i in range(200)],
        }
    )
    return df


@pytest.fixture
def dummy_model_and_data(tmp_path, sample_dataframe):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)

    # Create dummy model
    dtrain = xgb.DMatrix(
        sample_dataframe.iloc[:, :-1], label=sample_dataframe["Target"]
    )
    model = xgb.train({}, dtrain, num_boost_round=1)

    extracted_dir = model_dir / "extracted"
    extracted_dir.mkdir()

    model_path = extracted_dir / "model.xgb"
    model.save_model(model_path)

    # Create tar.gz archive
    tar_path = model_dir / "model.tar.gz"
    import tarfile

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_path, arcname="model.xgb")

    return str(model_dir), sample_dataframe


def test_load_data(tmp_path, sample_dataframe):
    test_file = tmp_path / "test.csv"
    sample_dataframe.to_csv(test_file, index=False)

    df = load_data(str(test_file))
    assert not df.empty
    assert list(df.columns) == list(sample_dataframe.columns)


def test_extract_model(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    extracted_dir = model_dir / "extracted"
    extracted_dir.mkdir()

    dummy_file = extracted_dir / "model.xgb"
    dummy_file.write_text("dummy")

    archive_path = model_dir / "model.tar.gz"
    import tarfile

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(dummy_file, arcname="model.xgb")

    shutil.rmtree(extracted_dir)
    extract_model(str(archive_path), str(extracted_dir))

    assert (extracted_dir / "model.xgb").exists()


def test_load_xgboost_model(dummy_model_and_data):
    model_dir, _ = dummy_model_and_data
    model = load_xgboost_model(model_dir)
    assert isinstance(model, xgb.Booster)


def test_evaluate_model(dummy_model_and_data):
    model_dir, df = dummy_model_and_data
    model = load_xgboost_model(model_dir)
    features = df.columns[:-1].to_list()
    metrics = evaluate_model(df, features, model)
    assert "precision" in metrics
    assert 0.0 <= metrics["precision"] <= 1.0


def test_save_results(tmp_path):
    output_dir = tmp_path / "output"
    metrics = {"precision": 0.85}

    save_results(metrics, str(output_dir))

    eval_path = output_dir / "evaluation.json"
    assert eval_path.exists()

    with open(eval_path, "r") as f:
        result = json.load(f)
    assert result == metrics
