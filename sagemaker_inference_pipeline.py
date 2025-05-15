import re
import json

import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.sklearn.processing import SKLearnProcessor, ScriptProcessor
from sagemaker.workflow.steps import TransformStep, ProcessingStep, CacheConfig
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.transformer import Transformer
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.lambda_step import (
    Lambda,
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)
from sagemaker.workflow.monitor_batch_transform_step import MonitorBatchTransformStep
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.model_monitor import DatasetFormat

############################### Set up session and role ###############################
region = "us-east-2"
sagemaker_session = sagemaker.Session()
pipeline_session = PipelineSession()
sagemaker_client = boto3.client("sagemaker")
s3 = boto3.client("s3")
account = boto3.client("sts").get_caller_identity().get("Account")
role = f"arn:aws:iam::{account}:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole"

############################## Define pipeline parameters #############################
years_to_filter = ParameterString(name="Historical_Years", default_value="10")
instance_type = ParameterString(name="InstanceType", default_value="ml.m4.xlarge")
instance_count = ParameterInteger(name="InstanceCount", default_value=1)
input_data = ParameterString(
    name="InputData",
    default_value="s3://aws-portfolio-projects/snp500-data/inference_data/processed/sp500_processed.csv",
)
output_data = ParameterString(
    name="OutputData",
    default_value="s3://aws-portfolio-projects/snp500-data/batch-predictions/",
)

# Enable step caching to avoid reprocessing unchanged steps
cache_config = CacheConfig(enable_caching=True, expire_after="T3h")

######################### Step 1: Data Ingestion #########################
image_uri = "930627915954.dkr.ecr.us-east-2.amazonaws.com/stockmodel-image:latest"
data_ingestion_processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
)

ingestion_step = ProcessingStep(
    name="DataIngestion",
    processor=data_ingestion_processor,
    inputs=[],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="ingested",
            source="/opt/ml/processing/output",
            destination="s3://aws-portfolio-projects/snp500-data/inference_data/input/",
        )
    ],
    code="inference_scripts/data_ingestion.py",
    cache_config=cache_config,
    job_arguments=["--years-to-filter", years_to_filter],
)

######################### Step 2: Data Preprocessing #########################
data_preprocessor = SKLearnProcessor(
    framework_version="1.0-1", role=role, instance_type="ml.m5.large", instance_count=1
)

processing_step = ProcessingStep(
    name="DataPreprocessing",
    processor=data_preprocessor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=ingestion_step.properties.ProcessingOutputConfig.Outputs[
                "ingested"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="processed",
            source="/opt/ml/processing/output/train",
            destination="s3://aws-portfolio-projects/snp500-data/inference_data/processed/",
        )
    ],
    code="inference_scripts/data_processing.py",
    cache_config=cache_config,
    job_arguments=[
        "--input_dir",
        "/opt/ml/processing/input/",
        "--output_dir",
        "/opt/ml/processing/output/train",
    ],
)

######################### Step 3: Fetch Approved Model from Lambda #########################
lambda_role = "arn:aws:iam::930627915954:role/sagemaker-pipeline-lambda-role"
function_name = "getapprovedmodelname-sagemaker-step"
model_package_group_name = "StockPredictionModels"

func = Lambda(
    function_name=function_name,
    execution_role_arn=lambda_role,
    script="inference_scripts/lambda_getapproved_model.py",
    handler="lambda_getapproved_model.lambda_handler",
    timeout=600,
    memory_size=128,
)

# Lambda Outputs
output_param_1 = LambdaOutput(
    output_name="statusCode", output_type=LambdaOutputTypeEnum.String
)
output_param_2 = LambdaOutput(
    output_name="modelArn", output_type=LambdaOutputTypeEnum.String
)
output_param_3 = LambdaOutput(
    output_name="s3uriConstraints", output_type=LambdaOutputTypeEnum.String
)
output_param_4 = LambdaOutput(
    output_name="s3uriStatistics", output_type=LambdaOutputTypeEnum.String
)
output_param_5 = LambdaOutput(
    output_name="modelName", output_type=LambdaOutputTypeEnum.String
)

lambda_getmodel_step = LambdaStep(
    name="LambdaStepGetApprovedModel",
    lambda_func=func,
    inputs={
        "model_package_group_name": model_package_group_name,
        "lambda_execution_role": lambda_role,
    },
    outputs=[
        output_param_1,
        output_param_2,
        output_param_3,
        output_param_4,
        output_param_5,
    ],
)

######################### Step 4: Batch Transform with Monitoring #########################
transformer = Transformer(
    model_name=lambda_getmodel_step.properties.Outputs["modelName"],
    instance_count=instance_count.default_value,
    instance_type=instance_type.default_value,
    output_path=output_data.default_value,
    sagemaker_session=pipeline_session,
    strategy="SingleRecord",
    assemble_with="Line",
)

transform_arg = transformer.transform(
    input_data.default_value, content_type="text/csv", split_type="Line"
)

batch_monitor_reports_output_path = "s3://aws-portfolio-projects/snp500-data/monitoring_artifacts/data-quality-monitor-reports/"

job_config = CheckJobConfig(role=role)
data_quality_config = DataQualityCheckConfig(
    baseline_dataset=processing_step.properties.ProcessingOutputConfig.Outputs[
        "processed"
    ].S3Output.S3Uri,
    dataset_format=DatasetFormat.csv(header=False),
    output_s3_uri=batch_monitor_reports_output_path,
)

transform_and_monitor_step = MonitorBatchTransformStep(
    name="MonitorStockDataQuality",
    transform_step_args=transform_arg,
    monitor_configuration=data_quality_config,
    check_job_configuration=job_config,
    monitor_before_transform=True,
    fail_on_violation=False,
    supplied_baseline_statistics=lambda_getmodel_step.properties.Outputs[
        "s3uriStatistics"
    ],
    supplied_baseline_constraints=lambda_getmodel_step.properties.Outputs[
        "s3uriConstraints"
    ],
)

######################### Step 5: Trigger Retraining if Violation Detected #########################
function_name = "trigger-retraining-sagemaker-step"
training_pipeline_name = "StockTrainingPipeline"
topic_arn = "arn:aws:sns:us-east-2:930627915954:StockModelAlarmTopic"

func = Lambda(
    function_name=function_name,
    execution_role_arn=lambda_role,
    script="inference_scripts/lambda_detectviolation.py",
    handler="lambda_detectviolation.lambda_handler",
    timeout=600,
    memory_size=128,
)

lambda_retrainmodel_step = LambdaStep(
    name="LambdaStepStockRetrainModel",
    lambda_func=func,
    inputs={
        "sagemaker_pipeline_name": training_pipeline_name,
        "SNS_topic_arn": topic_arn,
    },
    depends_on=["MonitorStockDataQuality"],
)

######################### Execute Pipeline #########################
pipeline = Pipeline(
    name="StockInferencePipeline",
    parameters=[
        years_to_filter,
        instance_type,
        instance_count,
        input_data,
        output_data,
    ],
    steps=[
        ingestion_step,
        processing_step,
        lambda_getmodel_step,
        transform_and_monitor_step,
        lambda_retrainmodel_step,
    ],
)

# Register or update the pipeline
pipeline.upsert(role_arn=role)

# Start pipeline execution
# execution = pipeline.start()
# print(f"Pipeline Execution ARN: {execution.arn}")
# execution.wait()
