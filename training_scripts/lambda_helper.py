import json
import boto3


def lambda_handler(event, context):
    """Lambda function to create a new SageMaker endpoint and delete other endpoints starting with 'stock-prediction-endpoint'."""
    sm_client = boto3.client("sagemaker")

    # The name of the model created in the Pipeline CreateModelStep
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

    # Create the new Endpoint
    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )
    print(f"New endpoint {endpoint_name} created.")

    # # List all existing endpoints
    # list_endpoints_response = sm_client.list_endpoints()
    # existing_endpoints = list_endpoints_response["Endpoints"]

    # for endpoint in existing_endpoints:
    #     existing_endpoint_name = endpoint["EndpointName"]
    #     endpoint_status = endpoint["EndpointStatus"]

    #     # Delete endpoints that start with 'stock-prediction-endpoint', excluding the newly created one
    #     if (
    #         existing_endpoint_name != endpoint_name
    #         and existing_endpoint_name.startswith("stock-prediction-endpoint")
    #         and endpoint_status == "InService"
    #     ):
    #         print(f"Deleting endpoint: {existing_endpoint_name}")
    #         sm_client.delete_endpoint(EndpointName=existing_endpoint_name)

    return {
        "statusCode": 200,
        "body": json.dumps(
            f"Created new endpoint {endpoint_name} and deleted other endpoints starting with 'stock-prediction-endpoint'."
        ),
    }

