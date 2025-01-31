import boto3
import json
import csv
from io import StringIO
import uuid


def lambda_handler(event, context):
    """Handles AWS Lambda events to process data, invoke a SageMaker endpoint, and store results in DynamoDB.

    Args:
        event (dict): Event data passed to the Lambda function. Expected keys:
            - endpoint_name (str): Name of the SageMaker endpoint.
            - processed_data_s3_uri (str): S3 URI of the processed data.
            - dynamodb_table_name (str): Name of the DynamoDB table to store results.
        context (object): Lambda Context runtime methods and attributes.

    Returns:
        dict: Response from the SageMaker endpoint invocation.
    """

    sagemaker_client = boto3.client("sagemaker-runtime")
    dynamodb_client = boto3.client("dynamodb")

    endpoint_name = event["endpoint_name"]
    processed_data_s3_uri = event["processed_data_s3_uri"]
    dynamodb_table_name = event[
        "dynamodb_table_name"
    ]  # Assuming the table name is passed in the event

    # Read the processed data from S3
    s3 = boto3.client("s3")
    bucket, key = processed_data_s3_uri.replace("s3://", "").split("/", 1)
    response = s3.get_object(Bucket=bucket, Key=key)
    payload = response["Body"].read().decode("utf-8")

    # Parse CSV to get the last row
    csv_reader = csv.reader(StringIO(payload))
    rows = list(csv_reader)

    # Get the last row of the CSV data
    last_row = rows[-1] if rows else []

    # Convert the last row to a string (assuming the row is in CSV format)
    last_row_payload = ",".join(last_row)

    # Invoke SageMaker endpoint with the last row as the payload
    response = sagemaker_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="text/csv",
        Body=last_row_payload,
    )

    # Get the predicted result from SageMaker
    prediction = response["Body"].read().decode("utf-8")

    # Remove the square brackets and convert the value to a float
    prediction_value = float(
        prediction.strip("[]")
    )  # Strip the brackets and convert to float
    print("prediction value (as float) =", prediction_value)

    # Apply the threshold
    prediction_class = 1 if prediction_value >= 0.5 else 0
    print("prediction class =", prediction_class)

    # Generate a unique ID for the DynamoDB entry
    prediction_id = str(uuid.uuid4())[:8]

    # Store the prediction in DynamoDB
    dynamodb_client.put_item(
        TableName=dynamodb_table_name,
        Item={
            "id": {"S": prediction_id},  # Unique ID for the entry
            "prediction_value": {"S": prediction},  # Store the prediction
            "prediction_class": {
                "N": str(prediction_class)
            },  # Store the predicted class
        },
    )

    return {
        "statusCode": 200,
        "prediction": prediction,
        "dynamodb_entry_id": prediction_id,  # Return the DynamoDB entry ID
    }
