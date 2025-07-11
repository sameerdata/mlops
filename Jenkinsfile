pipeline {
    agent any

    parameters {
        string(name: 'AWS_ACCESS_KEY_ID', description: 'Your AWS Access Key ID')
        string(name: 'AWS_SECRET_ACCESS_KEY', description: 'Your AWS Secret Access Key')
        string(name: 'AWS_REGION', defaultValue: 'us-east-1', description: 'AWS region')
        string(name: 'DATASET_S3_BUCKET', defaultValue: 'mlops-project-datas', description: 'S3 bucket name')
        string(name: 'DATA_FILE', defaultValue: 'customer_churn_100.csv', description: 'CSV dataset filename')
        string(name: 'SAGEMAKER_ROLE', defaultValue: 'arn:aws:iam::981986258943:role/SageMakerExecutionRole', description: 'SageMaker execution role ARN')
    }

    environment {
        AWS_ACCESS_KEY_ID     = "${params.AWS_ACCESS_KEY_ID}"
        AWS_SECRET_ACCESS_KEY = "${params.AWS_SECRET_ACCESS_KEY}"
        AWS_DEFAULT_REGION    = "${params.AWS_REGION}"
    }

    stages {
        stage('Install dependencies') {
            steps {
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Run pipeline script') {
            steps {
                sh '''
                    . venv/bin/activate
                    python pipeline.py \
                        --bucket ${DATASET_S3_BUCKET} \
                        --data_file ${DATA_FILE} \
                        --region ${AWS_REGION} \
                        --role ${SAGEMAKER_ROLE}
                '''
            }
        }
    }

    post {
        success {
            echo '✅ SageMaker pipeline executed successfully.'
        }
        failure {
            echo '❌ Pipeline execution failed. Check logs for details.'
        }
    }
}

