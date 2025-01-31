# Automated-Stock-Market-Prediction-System

This project demonstrates how to build an automated stock market (SP 500) prediction system using AWS SageMaker. The components of this system are:

1. [**SageMaker Training Pipeline**](sagemaker-training-pipeline.ipynb)
2. [**SageMaker Inference Pipeline**](sagemaker-inference-pipeline.ipynb)
3. [**SageMaker Model Monitoring**](#sagemaker-model-monitoring-upcoming) (to be implemented)

## SageMaker Training Pipeline

![Sagemaker training pipeline](/images/Training_pipeline.jpeg)

This pipeline, as shown in the above diagram, is used to build and deploy the machine learning model. The training pipeline consists of the following steps:

- **Data Ingestion**: Fetches the stock market data (SP 500) from Yahoo Finance API and stores it in an S3 bucket for future reference.
- **Data Processing**: Retrieves the ingested data from the data ingestion phase and processes it into features ready for machine learning training.
- **Model Training**: Retrieves the features from the data processing stage, trains a machine learning model using the XGBoost algorithm, and stores the artifacts in an S3 bucket.
- **Model Evaluation**: Evaluates the trained model using the precision_score metric.
- **Model Registry**: Registers the model to the SageMaker model registry when the precision_score is above 0.5.
- **Model Deployment**: Uses LambdaStep to deploy the trained model to a SageMaker real-time endpoint.

## SageMaker Inference Pipeline

![Sagemaker inference pipeline](/images/Inference_pipeline.jpeg)

The inference pipeline consists of the following steps:
- **Data Ingestion**: Uses an `SKLearnProcessor` to ingest data.
- **Data Preprocessing**: Uses an `SKLearnProcessor` to preprocess the data.
- **Model Inference**: Uses a `ModelStep` to make predictions.

## SageMaker Model Monitoring (Upcoming)

The model monitoring will include:
- **Data Capture**: Enabling data capture for the endpoint to monitor input and output data.
- **Baseline Data and Constraints**: Setting up baseline data and constraints for monitoring.
- **Monitoring Schedule**: Creating a monitoring schedule to regularly check for data drift and model quality.
- **CloudWatch Alarms**: Setting up CloudWatch alarms to notify when data drift or model quality issues are detected.


## Pipeline Scheduling

- **Training Pipeline Schedule**: This schedules the SageMaker training pipeline using EventBridge. It is scheduled to run once every week.
- **Inference Pipeline Schedule**: This runs every weekday (Mon - Fri) and predicts whether the SP 500 will increase or decrease the next day.

## Conclusion

This project provides a comprehensive solution for automated stock market prediction using AWS SageMaker. It includes training, inference, and monitoring pipelines to ensure the model remains accurate and reliable over time.