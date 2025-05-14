# -----------------------------------------------------------------------------
# Dockerfile for a lightweight Python 3.11 environment with essential ML tools
#
# Base Image: python:3.11-slim-buster
# - A minimal Python image based on Debian Buster for faster builds and smaller size
#
# Installed Libraries:
# - pandas==2.2.2         : For data manipulation and analysis
# - yfinance==0.2.54      : For downloading historical stock market data
# - scikit-learn==1.5.1   : For machine learning models and utilities
#
# Environment Variables:
# - PYTHONUNBUFFERED=TRUE : Ensures real-time logging by disabling output buffering
#
# Entry Point:
# - Starts the container with Python 3 interpreter by default

FROM python:3.11-slim-buster

RUN pip3 install pandas==2.2.2 yfinance==0.2.60 scikit-learn==1.5.1 curl_cffi==0.10.0
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]