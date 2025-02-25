import boto3
import json

# Initialize the Step Functions client
sfn_client = boto3.client("stepfunctions")


def lambda_handler(event, context):
    # Extract S3 event information
    bucket_name = event["Records"][0]["s3"]["bucket"]["name"]
    file_name = event["Records"][0]["s3"]["object"]["key"]

    # Set the Step Function input
    input_data = {"S3Bucket": bucket_name, "S3Key": file_name}

    # Start the Step Function execution
    response = sfn_client.start_execution(
        stateMachineArn="arn:aws:states:us-east-2:930627915954:stateMachine:BatchTransformStateMachine",
        name="BatchTransformExecution-" + file_name,
        input=json.dumps(input_data),
    )

    print(f"Started Step Function execution: {response['executionArn']}")

    return {"statusCode": 200, "body": json.dumps(f"Execution started for {file_name}")}
