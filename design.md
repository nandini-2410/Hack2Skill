System Design – SepsisGuard
1. High-Level Architecture
Input Layer
ICU Monitors

Wearable Devices (HR, BP, SpO2)

Historical ICU Dataset

Data Ingestion Layer
AWS IoT Core (real-time vitals ingestion)

Amazon Kinesis (data streaming)

Processing Layer
AWS Lambda (data preprocessing & validation)

Feature engineering pipeline

ML Layer
Amazon SageMaker (Model Training)

SageMaker Endpoint (Real-time inference)

SHAP (Explainability)

Model Drift Monitoring

Storage Layer
Amazon S3 (datasets & model artifacts)

Amazon RDS / DynamoDB (patient & risk logs)

Application Layer
Backend API (FastAPI/Flask on EC2)

Real-time Dashboard UI

Alert & Escalation Layer
Amazon SNS (alerts)

Escalation workflow logic

Acknowledgment tracking system

2. System Workflow
Patient vitals collected from wearable/ICU.

Data sent to AWS IoT Core.

Lambda preprocesses data.

SageMaker endpoint predicts mortality percentage.

SHAP explains prediction.

Dashboard updates in real-time.

Alert triggered if sustained high risk detected.

Escalation initiated if no acknowledgment.

Delay detection logic analyzes treatment timeline.

3. Model Design
Model Type: XGBoost (Mortality Prediction)

Time-Series Feature Engineering

Organ Deterioration Speed Calculation

Threshold-based alert validation

SHAP for feature attribution

4. Database Design
Tables:

Patients

Vital Logs

Risk Predictions

Alert Logs

Acknowledgment Records

Delay Analysis Records

5. Security Design
IAM role-based access control

Encrypted S3 storage

Secure API Gateway

HTTPS communication

Audit logging