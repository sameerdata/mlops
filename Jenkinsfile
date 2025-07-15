pipeline {
    agent any

    parameters {
        string(name: 'AWS_ACCESS_KEY_ID', description: 'Your AWS Access Key ID')
        string(name: 'AWS_SECRET_ACCESS_KEY', description: 'Your AWS Secret Access Key')
        string(name: 'AWS_REGION', defaultValue: 'us-east-1', description: 'AWS region')
        string(name: 'DATASET_S3_BUCKET', defaultValue: 'mlops-project-datas', description: 'S3 bucket name')
        string(name: 'DATA_FILE', defaultValue: 'customer_churn_100.csv', description: 'CSV dataset filename')
        string(name: 'SAGEMAKER_ROLE', description: 'SageMaker execution role ARN (optional)')
    }

    environment {
        AWS_ACCESS_KEY_ID = "${params.AWS_ACCESS_KEY_ID}"
        AWS_SECRET_ACCESS_KEY = "${params.AWS_SECRET_ACCESS_KEY}"
        AWS_DEFAULT_REGION = "${params.AWS_REGION}"
        BUCKET = "${params.DATASET_S3_BUCKET}"
        DATA_FILE = "${params.DATA_FILE}"
        SAGEMAKER_ROLE = "${params.SAGEMAKER_ROLE}"
    }

    stages {
        stage('Clone Repository') {
            steps {
                checkout scm
            }
        }

        stage('Set up Python Environment') {
            steps {
                sh '''
                    python3 -m venv venv
                    bash -c "source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt"
                '''
            }
        }

        stage('Download Dataset from S3') {
            steps {
                sh '''
                    aws s3 cp s3://$BUCKET/$DATA_FILE dataset.csv
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                    bash -c "source venv/bin/activate && python3 train.py"
                '''
            }
        }

        stage('Upload Model to S3') {
            steps {
                sh '''
                    aws s3 cp model.pkl s3://$BUCKET/models/model.pkl
                '''
            }
        }
         stage('Deploy to SageMaker') {
            steps {
                sh '''
                    bash -c "source venv/bin/activate && python3 deploy_to_sagemaker.py"
                '''
            }
        }


    }

    post {
        always {
            echo '🚀 Pipeline completed.'
        }
    }
}
