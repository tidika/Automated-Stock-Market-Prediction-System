import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Set up path to import script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..", "inference_scripts")
sys.path.append(parent_dir)

from lambda_detectviolation import *


@pytest.fixture
def fake_event():
    return {
        "sagemaker_pipeline_name": "test-pipeline",
        "SNS_topic_arn": "arn:aws:sns:us-east-1:123456789012:MyTopic",
    }


@patch("lambda_detectviolation.s3_client.get_object")
@patch("lambda_detectviolation.send_sns_notification")
@patch("lambda_detectviolation.trigger_retraining")
def test_data_drift_triggers_retraining(
    mock_trigger, mock_notify, mock_get_object, fake_event
):
    # Mock S3 to return a JSON with violations
    mock_get_object.return_value = {
        "Body": MagicMock(read=lambda: b'{"violations": ["example_violation"]}')
    }

    lambda_handler(fake_event, None)

    mock_notify.assert_called_once_with(fake_event["SNS_topic_arn"])
    mock_trigger.assert_called_once_with(fake_event["sagemaker_pipeline_name"])


@patch("lambda_detectviolation.s3_client.get_object")
@patch("lambda_detectviolation.send_sns_notification")
@patch("lambda_detectviolation.trigger_retraining")
def test_no_data_drift_does_not_trigger(
    fake_trigger, fake_notify, mock_get_object, fake_event
):
    # Mock S3 to return a JSON with no violations
    mock_get_object.return_value = {
        "Body": MagicMock(read=lambda: b'{"violations": []}')
    }

    lambda_handler(fake_event, None)

    fake_notify.assert_not_called()
    fake_trigger.assert_not_called()
