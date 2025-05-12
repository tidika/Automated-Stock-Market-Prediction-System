import boto3
import pytz
from datetime import datetime, timedelta
from botocore.exceptions import ClientError

# ============ Configuration ============
ACCOUNT_ID = boto3.client("sts").get_caller_identity().get("Account")
REGION = "us-east-2"
CENTRAL_TZ = pytz.timezone("US/Central")
SCHEDULER_CLIENT = boto3.client("scheduler", region_name=REGION)

# ============ Helper Functions ============
def get_cron_expression(hour_central, minute=0, days="*"):
    """
    Convert Central Time to UTC and return cron expression.
    """
    utc_hour = (hour_central + 6) % 24  # 8 AM CT -> 14 UTC
    return f"cron({minute} {utc_hour} ? * {days} *)"


def schedule_sagemaker_pipeline(name, description, pipeline_name, cron_expression):
    sagemaker_pipeline_arn = (
        f"arn:aws:sagemaker:{REGION}:{ACCOUNT_ID}:pipeline/{pipeline_name}"
    )
    role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/eventbridge-scheduler-role"

    try:
        response = SCHEDULER_CLIENT.create_schedule(
            Name=name,
            ScheduleExpression=cron_expression,
            FlexibleTimeWindow={"Mode": "OFF"},
            Target={
                "Arn": sagemaker_pipeline_arn,
                "RoleArn": role_arn,
                "SageMakerPipelineParameters": {
                    "PipelineParameterList": []  # Add your parameters here if needed
                },
            },
            Description=description,
        )
        print(f"[✔] Scheduled with Scheduler: {name} -> {cron_expression}")
        return response

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_message = e.response["Error"]["Message"]

        if error_code == "ValidationException" and "already exists" in error_message:
            print(f"[!] Schedule '{name}' already exists. Skipping creation.")
        else:
            print(f"[✘] Failed to create schedule '{name}': {error_message}")


# ============ Schedule Definitions ============

# 1. Training pipeline: every Sunday at 8 AM CT
training_cron = get_cron_expression(hour_central=8, days="1")  # Sunday
schedule_sagemaker_pipeline(
    name="SageMakerTrainingPipelineSchedule",
    description="Trigger SageMaker training pipeline every Sunday at 8 AM CT",
    pipeline_name="StockTrainingPipeline",
    cron_expression=training_cron,
)

# 2. Inference pipeline: every Mon–Fri at 9 PM CT
inference_cron = get_cron_expression(hour_central=21, days="MON-FRI")
schedule_sagemaker_pipeline(
    name="SageMakerInferencePipelineSchedule",
    description="Trigger SageMaker inference pipeline every weekday at 9 PM CT",
    pipeline_name="StockInferencePipeline",
    cron_expression=inference_cron,
)
