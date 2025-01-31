# Automated-Stock-Market-Prediction-System

This project demonstrates how to build an automated stock market (SP 500) prediction system using AWS SageMaker. The components of this system are:

1. [**SageMaker Training Pipeline**](sagemaker-training-pipeline.ipynb) 
2. [**SageMaker Inference Pipeline**](sagemaker-inference-pipeline.ipynb)
3. **SageMaker Model Monitoring** (to be implemented): 

## Training Pipeline
![Sagemaker training pipeline](/images/Training_pipeline.jpeg)
<!-- Add a blank line here -->

This pipeline as shown in the above diagram is used to build and deploy the machine learning model. The training pipeline consists of the following steps:

- **Data Ingestion**: This step fetches the stock market data (sp500) from yahoo finance api and store it in s3 bucket for future reference.
- **Data Processing**: This step retrieves the ingested data from the data ingestion phase and processes it into a feature ready for machine learning training. 
- **Model Training**: Using an `Estimator` to train the model.
- **Model Evaluation**: Using a `ProcessingStep` to evaluate the model.
- **Model Deployment**: Using a `LambdaStep` to deploy the model.


## Inference Pipeline:
![Sagemaker training pipeline](/images/Inference_pipeline.jpeg)
<!-- Add a blank line here -->
This pipeline is used for making predictions.The inference pipeline consists of the following steps:
- **Data Ingestion**: Using an `SKLearnProcessor` to ingest data.
- **Data Preprocessing**: Using an `SKLearnProcessor` to preprocess the data.
- **Model Inference**: Using a `ModelStep` to make predictions.

## Model Monitoring (Upcoming)

This will be used to monitor data quality and model performance. It will trigger retraining when there is a significant shift in the data and send notifications to the product owner.
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