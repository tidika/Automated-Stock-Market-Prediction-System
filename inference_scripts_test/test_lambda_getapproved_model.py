import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Set up path to import script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..", "inference_scripts")
sys.path.append(parent_dir)

from lambda_getapproved_model import *


@pytest.fixture
def mock_event():
    return {
        "model_package_group_name": "my-model-group",
        "lambda_execution_role": "arn:aws:iam::123456789012:role/SageMakerRole",
    }


@patch("lambda_getapproved_model.sm_client", new_callable=MagicMock)
def test_lambda_handler_success(mock_sm_client, mock_event):
    # Mock list_model_packages
    mock_sm_client.list_model_packages.return_value = {
        "ModelPackageSummaryList": [
            {
                "ModelPackageArn": "arn:aws:sagemaker:us-east-1:123456789012:model-package/my-model-group/5"
            }
        ]
    }

    # Mock describe_model_package
    mock_sm_client.describe_model_package.return_value = {
        "InferenceSpecification": {
            "Containers": [
                {
                    "Image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image",
                    "ModelDataUrl": "s3://my-bucket/model.tar.gz",
                }
            ]
        },
        "ModelMetrics": {
            "ModelDataQuality": {
                "Statistics": {"S3Uri": "s3://my-bucket/statistics.json"},
                "Constraints": {"S3Uri": "s3://my-bucket/constraints.json"},
            }
        },
    }

    # Mock create_model to simulate success
    mock_sm_client.create_model.return_value = {
        "ModelArn": "arn:aws:sagemaker:us-east-1:123456789012:model/my-model"
    }

    # Run the handler
    response = lambda_handler(mock_event, None)

    assert response["statusCode"] == 200
    assert "modelArn" in response
    assert "s3uriConstraints" in response
    assert "s3uriStatistics" in response
    assert "modelName" in response
    assert response["modelName"] == "my-model-group-5"
