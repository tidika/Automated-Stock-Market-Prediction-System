import boto3
import sagemaker
import logging

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.sklearn.processing import SKLearnProcessor, ScriptProcessor
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.model_monitor import DatasetFormat
from sagemaker.workflow.functions import Join
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.workflow.properties import PropertyFile
from sagemaker.processing import ScriptProcessor as EvalScriptProcessor
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep, JsonGet
from sagemaker.workflow.fail_step import FailStep

############################### Set logging level to reduce verbosity ####################
logging.getLogger("sagemaker").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)

############################### Set up session and role #################################
sagemaker_session = sagemaker.Session()
pipeline_session = PipelineSession()
account = boto3.client("sts").get_caller_identity().get("Account")
role = f"arn:aws:iam::{account}:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole"

############################## Define pipeline parameters ###############################
years_to_filter = ParameterString(name="Historical_Years", default_value="30")
train_instance_type = ParameterString(
    name="TrainingInstanceType", default_value="ml.m5.large"
)
model_approval_status = ParameterString(
    name="ModelApprovalStatus", default_value="Approved"
)
cache_config = CacheConfig(enable_caching=True, expire_after="T3h")

######################### Step 1: Data Ingestion ######################################
image_uri = "930627915954.dkr.ecr.us-east-2.amazonaws.com/stockmodel-image:latest"
data_ingestion_processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
)
step_data_ingestion = ProcessingStep(
    name="DataIngestion",
    processor=data_ingestion_processor,
    inputs=[],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="snp500",
            source="/opt/ml/processing/output",
            destination="s3://aws-portfolio-projects/snp500-data/input_data/",
        ),
    ],
    code="training_scripts/data_ingestion.py",
    cache_config=cache_config,
    job_arguments=["--years-to-filter", years_to_filter],
)

######################### Step 2: Data Preprocessing ####################################
preprocessor = SKLearnProcessor(
    framework_version="1.0-1", role=role, instance_type="ml.m5.large", instance_count=1
)
step_data_processing = ProcessingStep(
    name="DataPreprocessing",
    processor=preprocessor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=step_data_ingestion.properties.ProcessingOutputConfig.Outputs[
                "snp500"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="train_data",
            source="/opt/ml/processing/output/train",
            destination="s3://aws-portfolio-projects/snp500-data/train_data/",
        )
    ],
    code="training_scripts/data_processing.py",
    cache_config=cache_config,
    job_arguments=[
        "--input_path",
        "/opt/ml/processing/input/snp500.csv",
        "--output_dir",
        "/opt/ml/processing/output/train",
    ],
)

######################### Step 3: Model Training #####################################
xgboost_estimator = XGBoost(
    entry_point="training_scripts/train_model.py",
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="1.5-1",
    py_version="py3",
    output_path="s3://aws-portfolio-projects/snp500-data/model_artifacts/",
    base_job_name="xgboost-stockmarket-training-job",
    disable_profiler=True,
)
step_train = TrainingStep(
    name="ModelTrainingStep",
    estimator=xgboost_estimator,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=step_data_processing.properties.ProcessingOutputConfig.Outputs[
                "train_data"
            ].S3Output.S3Uri,
            content_type="text/csv",
        )
    },
    cache_config=cache_config,
)

######################### Step 4: Model Evaluation ######################################
evaluation_processor = EvalScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve(
        framework="xgboost", region=sagemaker_session.boto_region_name, version="1.5-1"
    ),
    command=["python3"],
    instance_type="ml.m5.large",
    instance_count=1,
    base_job_name="model-evaluation",
    role=role,
    sagemaker_session=sagemaker_session,
)
evaluation_report = PropertyFile(
    name="EvaluationReport", output_name="evaluation", path="evaluation.json"
)
step_evaluation = ProcessingStep(
    name="ModelEvaluation",
    processor=evaluation_processor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model",
        ),
        sagemaker.processing.ProcessingInput(
            source=step_data_processing.properties.ProcessingOutputConfig.Outputs[
                "train_data"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/input",
        ),
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination="s3://aws-portfolio-projects/snp500-data/evaluation_results/",
        )
    ],
    code="training_scripts/evaluate_model.py",
    property_files=[evaluation_report],
    cache_config=cache_config,
    job_arguments=[
        "--input-path",
        "/opt/ml/processing/input/train.csv",
        "--model-path",
        "/opt/ml/processing/model",
        "--output-path",
        "/opt/ml/processing/evaluation",
    ],
)

######################### Step 5: Data quality baseline for drift detection ##########################
check_job_config = CheckJobConfig(
    role=role,
    instance_count=1,
    instance_type="ml.c5.xlarge",
    volume_size_in_gb=120,
    sagemaker_session=sagemaker_session,
)
data_quality_check_config = DataQualityCheckConfig(
    baseline_dataset="s3://aws-portfolio-projects/snp500-data/train_data/features.csv",
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=Join(
        on="/",
        values=[
            "s3:/",
            "aws-portfolio-projects",
            "snp500-data/monitoring_artifacts",
            "data-quality-monitor-reports",
        ],
    ),
)
baseline_model_data_step = QualityCheckStep(
    name="DataQualityCheckStep",
    skip_check=True,
    register_new_baseline=True,
    quality_check_config=data_quality_check_config,
    check_job_config=check_job_config,
    model_package_group_name="StockPredictionModels",
    depends_on=["ModelEvaluation"],
)

######################### Step 6: Conditional model registration based on evaluation ##########################
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=step_evaluation.outputs[0].destination + "/evaluation.json",
        content_type="application/json",
    ),
    model_data_statistics=MetricsSource(
        s3_uri=baseline_model_data_step.properties.CalculatedBaselineStatistics,
        content_type="application/json",
    ),
    model_data_constraints=MetricsSource(
        s3_uri=baseline_model_data_step.properties.CalculatedBaselineConstraints,
        content_type="application/json",
    ),
)
drift_check_baselines = DriftCheckBaselines(
    model_data_statistics=MetricsSource(
        s3_uri=baseline_model_data_step.properties.BaselineUsedForDriftCheckStatistics,
        content_type="application/json",
    ),
    model_data_constraints=MetricsSource(
        s3_uri=baseline_model_data_step.properties.BaselineUsedForDriftCheckConstraints,
        content_type="application/json",
    ),
)
step_register = RegisterModel(
    name="StockRegisterModel",
    estimator=xgboost_estimator,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="StockPredictionModels",
    approval_status="Approved",
    model_metrics=model_metrics,
    drift_check_baselines=drift_check_baselines,
)

# Define precision threshold condition
precision_threshold = 0.52
condition = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step=step_evaluation, property_file=evaluation_report, json_path="precision"
    ),
    right=precision_threshold,
)
step_fail = FailStep(
    name="StockModelFail",
    error_message=Join(
        on=" ",
        values=["Execution failed due to Precision score <", str(precision_threshold)],
    ),
)
step_conditional_register = ConditionStep(
    name="stockmarket_precision_cond",
    conditions=[condition],
    if_steps=[step_register],
    else_steps=[step_fail],
)


################################################ Execute Pipeline ##################################
pipeline = Pipeline(
    name="StockTrainingPipeline",
    parameters=[years_to_filter, train_instance_type, model_approval_status],
    steps=[
        step_data_ingestion,
        step_data_processing,
        step_train,
        step_evaluation,
        step_conditional_register,
    ],
    sagemaker_session=sagemaker_session,
)
pipeline.upsert(role_arn=role)
execution = pipeline.start()
execution.wait()
