import boto3
import json

sm_client = boto3.client("sagemaker")
s3_client = boto3.client("s3")
sns_client = boto3.client("sns")


def lambda_handler(event, context):
    print("Lambda function triggered by SageMaker Pipeline.")

    pipeline_name = event["sagemaker_pipeline_name"]
    sns_topic_arn = event["SNS_topic_arn"]

    # S3 path to the constraint violations file
    bucket = "aws-portfolio-projects"
    key = "snp500-data/monitoring_artifacts/data-quality-monitor-reports/constraint_violations.json"

    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        violation_data = json.loads(response["Body"].read().decode("utf-8"))

        if violation_data.get("violations", []):
            print("🚨 Data drift detected! Triggering retraining...")
            send_sns_notification(sns_topic_arn)
            trigger_retraining(pipeline_name)
        else:
            print("✅ No data drift detected. No retraining needed.")
    except s3_client.exceptions.NoSuchKey:
        print("❌ constraint_violations.json not found.")
    except Exception as e:
        print(f"❌ Error checking violations: {e}")


def send_sns_notification(sns_topic_arn):
    message = {
        "Subject": "🚨 Data Drift Alert - Retraining Initiated 🚀",
        "Message": "SageMaker Model Monitor detected data drift. A retraining job has been triggered.",
    }

    sns_client.publish(
        TopicArn=sns_topic_arn,
        Message=json.dumps(message),
        Subject="AWS SageMaker Retraining Triggered",
    )
    print("✅ SNS Notification sent.")


def trigger_retraining(pipeline_name):
    response = sm_client.start_pipeline_execution(PipelineName=pipeline_name)
    print(f"🚀 Training pipeline triggered: {response['PipelineExecutionArn']}")
