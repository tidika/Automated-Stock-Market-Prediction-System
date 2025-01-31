# Automated-Stock-Market-Prediction-System

This project demonstrates how to build an automated stock market (SP 500) prediction system using AWS SageMaker. The components of this system include:

1. [**SageMaker Training Pipeline**](sagemaker-training-pipeline.ipynb): This pipeline is used to build and deploy the machine learning model. It includes steps for data ingestion, data processing, model training, model evaluation and model deployment. 
![Sagemaker training pipeline](/images/training_pipeline.jpeg)
<!-- Add a blank line here -->

2. [**SageMaker Inference Pipeline**](sagemaker-inference-pipeline.ipynb): This pipeline is used for making predictions. It includes steps for data ingestion, data preprocessing, and model inference.
![Sagemaker training pipeline](/images/training_pipeline.jpeg)
<!-- Add a blank line here -->

3. **SageMaker Model Monitoring** (to be implemented): This will be used to monitor data quality and model performance. It will trigger retraining when there is a significant shift in the data and send notifications to the product owner.

## Training Pipeline

The training pipeline consists of the following steps:
- **Data Ingestion**: Using an `SKLearnProcessor` to ingest data.
- **Data Processing**: Using an `SKLearnProcessor` to preprocess the data.
- **Model Training**: Using an `Estimator` to train the model.
- **Model Evaluation**: Using a `ProcessingStep` to evaluate the model.
- **Model Deployment**: Using a `LambdaStep` to deploy the model.


## Inference Pipeline:

The inference pipeline consists of the following steps:
- **Data Ingestion**: Using an `SKLearnProcessor` to ingest data.
- **Data Preprocessing**: Using an `SKLearnProcessor` to preprocess the data.
- **Model Inference**: Using a `ModelStep` to make predictions.

## Model Monitoring (Upcoming)

The model monitoring will include:
- **Data Capture**: Enabling data capture for the endpoint to monitor input and output data.
- **Baseline Data and Constraints**: Setting up baseline data and constraints for monitoring.
- **Monitoring Schedule**: Creating a monitoring schedule to regularly check for data drift and model quality.
- **CloudWatch Alarms**: Setting up CloudWatch alarms to notify when data drift or model quality issues are detected.

## Images

![Training Pipeline](images/training_pipeline.png)
![Inference Pipeline](images/inference_pipeline.png)
![Model Monitoring](images/model_monitoring.png)

For more details, refer to the [images directory](images/).

## Conclusion

This project provides a comprehensive solution for automated stock market prediction using AWS SageMaker. It includes training, inference, and monitoring pipelines to ensure the model remains accurate and reliable over time.