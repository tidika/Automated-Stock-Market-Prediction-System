import boto3
import pytz
from datetime import datetime, timedelta

# ============ Configuration ============
ACCOUNT_ID = boto3.client("sts").get_caller_identity().get("Account")
REGION = "us-east-2"
CENTRAL_TZ = pytz.timezone("US/Central")
EVENTS_CLIENT = boto3.client("events", region_name=REGION)

# ============ Helper Functions ============


def get_cron_expression(hour_central, minute=0, days="*"):
    """
    Convert Central Time to UTC and return cron expression.
    """
    utc_hour = (hour_central + 6) % 24  # 8 AM CT -> 14 UTC
    return f"cron({minute} {utc_hour} ? * {days} *)"

def schedule_sagemaker_pipeline(name, description, pipeline_name, cron_expression):
    rule_response = EVENTS_CLIENT.put_rule(
        Name=name,
        ScheduleExpression=cron_expression,
        State="ENABLED",
        Description=description,
    )

    sagemaker_pipeline_arn = (
        f"arn:aws:sagemaker:{REGION}:{ACCOUNT_ID}:pipeline/{pipeline_name}"
    )
    role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/Amazon_EventBridge_Scheduler_SAGEMAKER_5ec175fff8"

    target_response = EVENTS_CLIENT.put_targets(
        Rule=name,
        Targets=[
            {"Id": name + "Target", "Arn": sagemaker_pipeline_arn, "RoleArn": role_arn}
        ],
    )

    print(f"Scheduled: {name} -> {cron_expression}")
    return target_response


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
