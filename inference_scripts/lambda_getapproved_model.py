"""
This Lambda function queries SageMaker Model Registry for a specific model package
group provided on input to identify the latest approved model version and return related metadata.
The output includes:
(1) model package arn (2) packaged model name (3) S3 URI for statistics baseline
(4) S3 URI for constraints baseline
The output is then used as input into the next step in the pipeline that
performs batch monitoring and scoring using the latest approved model.
"""

import logging
import os

import boto3
import botocore
import json
from botocore.exceptions import ClientError

# Initialize clients
s3 = boto3.client("s3")
sm_client = boto3.client("sagemaker")


def lambda_handler(event, context):
    logger = logging.getLogger()
    logger.setLevel(os.getenv("LOGGING_LEVEL", logging.INFO))

    # The model package group name
    model_package_group_name = event["model_package_group_name"]
    execution_role_arn = event["lambda_execution_role"]

    print(model_package_group_name)
    print(execution_role_arn)

    try:

        model_name = latest_model_name(model_package_group_name)
        create_model(model_package_group_name, model_name, execution_role_arn)
        model_package_arn = get_latest_model_version_arn(model_package_group_name)

        s3_baseline_uri_response = sm_client.describe_model_package(
            ModelPackageName=model_package_arn
        )

        s3_baseline_uri_statistics = s3_baseline_uri_response["ModelMetrics"][
            "ModelDataQuality"
        ]["Statistics"]["S3Uri"]
        s3_baseline_uri_constraints = s3_baseline_uri_response["ModelMetrics"][
            "ModelDataQuality"
        ]["Constraints"]["S3Uri"]

        logger.info(
            f"Identified the latest data quality baseline statistics for approved model package: {s3_baseline_uri_statistics}"
        )
        logger.info(
            f"Identified the latest data quality baseline constraints for approved model package: {s3_baseline_uri_constraints}"
        )

        return {
            "statusCode": 200,
            "modelArn": model_package_arn,
            "s3uriConstraints": s3_baseline_uri_constraints,
            "s3uriStatistics": s3_baseline_uri_statistics,
            "modelName": model_name,
        }

    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)


# Constants
TAGS = [
    {"Key": "Project", "Value": "StockPrediction"},
    {"Key": "Environment", "Value": "Production"},
    {"Key": "Owner", "Value": "Tochi"},
]


def latest_model_name(model_package_group_name):
    """
    Determines the latest model name to be used by batch transform.
    Extracts the model package group name and version from the ARN.
    """
    model_package_group_arn = get_latest_model_version_arn(model_package_group_name)
    print(model_package_group_arn)
    # Split the ARN to extract components
    arn_parts = model_package_group_arn.split(":")
    if len(arn_parts) < 6:
        raise ValueError("Invalid model package group ARN format.")

    # Extract the model package group name and version
    model_package_info = arn_parts[5].split("/")
    if len(model_package_info) != 3:
        raise ValueError("Invalid model package group ARN format.")

    model_package_group_name = model_package_info[1]
    model_package_version = model_package_info[2]

    # Construct the desired model name
    model_name = f"{model_package_group_name}-{model_package_version}"
    return model_name


def get_latest_model_version_arn(model_package_group_name):
    """Fetches the latest model package ARN from a model package group."""
    response = sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
    )
    return response["ModelPackageSummaryList"][0]["ModelPackageArn"]


def create_model(model_package_group_name, model_name, execution_role_arn):
    """Creates a SageMaker model with error handling for duplicate models."""
    latest_model_arn = get_latest_model_version_arn(model_package_group_name)
    model_details = sm_client.describe_model_package(ModelPackageName=latest_model_arn)

    try:
        response = sm_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": model_details["InferenceSpecification"]["Containers"][0][
                    "Image"
                ],
                "ModelDataUrl": model_details["InferenceSpecification"]["Containers"][
                    0
                ]["ModelDataUrl"],
            },
            ExecutionRoleArn=execution_role_arn,
            Tags=TAGS,
        )
        print(f"Created model: {model_name}")
        return response

    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            print(f"Model '{model_name}' already exists. Using existing model.")
        else:
            print(f"An error occurred: {e}")
            raise  # Re-raise unexpected errors

    return None
