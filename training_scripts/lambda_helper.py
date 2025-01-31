import json
import boto3
from botocore.exceptions import ClientError

def lambda_handler(event, context):
    """Handles AWS Lambda events to create or update a SageMaker endpoint configuration and endpoint.

    Args:
        event (dict): Event data passed to the Lambda function. Expected keys:
            - model_name (str): Name of the SageMaker model.
            - endpoint_config_name (str): Name of the endpoint configuration.
            - endpoint_name (str): Name of the endpoint to be created or updated.
        context (object): Lambda Context runtime methods and attributes.

    Returns:
        dict: Response from the SageMaker create or update endpoint operation.
    """
    sm_client = boto3.client("sagemaker")

    model_name = event["model_name"]
    endpoint_config_name = event["endpoint_config_name"]
    endpoint_name = event["endpoint_name"]
    print("Endpoint name is: ", endpoint_name)

    # Create the Endpoint Configuration
    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "InstanceType": "ml.m5.large",
                "InitialVariantWeight": 1,
                "InitialInstanceCount": 1,
                "ModelName": model_name,
                "VariantName": "AllTraffic",
            }
        ],
    )
    print(f"Endpoint configuration {endpoint_config_name} created.")

    # Check if the endpoint already exists
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            endpoint_exists = False
        else:
            raise

    if endpoint_exists:
        # Update the existing endpoint
        update_endpoint_response = sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print(f"Updated endpoint {endpoint_name}.")
        return {
            "statusCode": 200,
            "body": json.dumps(
                f"Updated endpoint {endpoint_name}."
            ),
        }
    else:
        # Create the new Endpoint
        create_endpoint_response = sm_client.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
        print(f"New endpoint {endpoint_name} created.")
        return {
            "statusCode": 200,
            "body": json.dumps(
                f"Created new endpoint {endpoint_name}."
            ),
        }