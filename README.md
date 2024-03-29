# README for Smart Battery Monitoring System Project

## Overview
This repository contains the code and models developed for a smart battery monitoring system. The system is controlled via voice commands to start/stop monitoring the battery status of IoT devices. Utilizing TensorFlow Lite, the project involves various stages, including voice activity detection (VAD), intent detection, and the integration of multiple cloud computing resources. It features a resource-driven design approach, optimizing for memory and latency constraints on IoT devices.

## PART 1

### TimeSeries_isolated.py
This isolated Python script monitors your PC's battery status (percentage and power supply connection) and uploads the data to Redis Cloud as a time series.

### VAD_isolated.py
An isolated script that records audio continuously. It saves the data on the disk only if speech is detected using finely-tuned VAD parameters.

## PART 2

### preprocessing.py
This script includes pre-processing methods utilized in the accompanying Jupyter notebook `training.ipynb` for data preparation before model training.

### training.ipynb
Jupyter notebook script for training and generating a TensorFlow Lite model, optimized for energy efficiency, latency reduction, and minimal memory usage.

### model.tflite.zip
The compressed TensorFlow Lite model file, ready to be deployed on the IoT device after training in the cloud.

### stages_integration.py
A Python script that combines a Voice User Interface (VUI) based on VAD and Keyword Spotting (KWS). The script enables:
- Disabled monitoring initially.
- Background running VUI that continuously checks for speech in the audio input.
- Voice command detection ("yes" to start and "no" to stop monitoring) with high accuracy.
- Data collection for battery status every second and storage on Redis.
- Command-line execution with arguments for device, host, port, user, and password specifications for Redis Cloud.

## PART 3

### publisher.py (devices)
This Python script monitors your PC's battery status, publishing the data to an MQTT broker every second. It batches and transmits the last 10 records as a single JSON-formatted message.

### subscriber.ipynb (server)
A Jupyter notebook that functions as an MQTT subscriber. It receives and processes battery status messages and stores them for further analysis.

### rest_server.ipynb
A Jupyter notebook designed to serve as a REST Server, which provides APIs for retrieving and managing the battery status data collected from IoT devices.

### rest_client.ipynb
This notebook acts as a REST Client for Data Visualization. It retrieves data from the REST Server and visualizes the battery status information for further insights.

