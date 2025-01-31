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
- **Model Training**: it retrieves the features from the data processing stage and trains a machine learning model using xgboost algorithm and stores the artifacts in an s3 bucket.
- **Model Evaluation**: It evaluates the trained model using precision_score metric. 
- **Model Registry**: It registers the model to sagemaker model registry when the percision_score is above 0.5
- **Model Deployment**: It uses LambdaStep to deploy the trained model to a sagmaker realtime endpoint


## Inference Pipeline:
![Sagemaker training pipeline](/images/Inference_pipeline.jpeg)
<!-- Add a blank line here -->
This pipeline is used for making predictions.The inference pipeline consists of the following steps:
- **Data Ingestion**: Similar to data ingestion in the training pipeline, it fetches inference data from the yahoo finance api.
- **Data Preprocessing**: It retrieves the inference data from the ingestion step and processes it and store it in an s3 bucket.
- **Model Inference**: This step uses lambda function to retrieve the inference data from s3 bucket and pass it through the deployed model endpoint and get predicted data which is stored in dynamodb.

## Model Monitoring (Upcoming)

This will be used to monitor data quality and model performance. It will trigger retraining when there is a significant shift in the data and send notifications to the product owner.
The model monitoring will include:
- **Data Capture**: Enabling data capture for the endpoint to monitor input and output data.
- **Baseline Data and Constraints**: Setting up baseline data and constraints for monitoring.
- **Monitoring Schedule**: Creating a monitoring schedule to regularly check for data drift and model quality.
- **CloudWatch Alarms**: Setting up CloudWatch alarms to notify when data drift or model quality issues are detected.

## Pipeline Scheduling
- Training pipeline schedule: This schedules the sagemaker training pipeline using eventbridge. It is scheduled to run once every week.
- Inference pipeline schedule: This runs every weekday (Mon - Fri) and predicts whether the sp500 will increase or decrease the next day. 

## Conclusion

This project provides a comprehensive solution for automated stock market prediction using AWS SageMaker. It includes training, inference, and monitoring pipelines to ensure the model remains accurate and reliable over time.