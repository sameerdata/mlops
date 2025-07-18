pipeline {
    agent any

    environment {
        AZURE_SUBSCRIPTION_ID = credentials('AZ_SUBSCRIPTION_ID')
        AZURE_CLIENT_ID = credentials('AZ_CLIENT_ID')
        AZURE_CLIENT_SECRET = credentials('AZ_CLIENT_SECRET')
        AZURE_TENANT_ID = credentials('AZ_TENANT_ID')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Env') {
            steps {
                sh '''
                    python3 -m venv venv
                    source venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Login to Azure') {
            steps {
                sh '''
                    az login --service-principal \
                      -u $AZURE_CLIENT_ID \
                      -p $AZURE_CLIENT_SECRET \
                      --tenant $AZURE_TENANT_ID
                    az account set --subscription $AZURE_SUBSCRIPTION_ID
                '''
            }
        }

        stage('Submit Azure ML Job') {
            steps {
                sh '''
                    az ml job create --file azure-job.yml
                '''
            }
        }

        stage('Register Model (Optional)') {
            steps {
                sh '''
                    az ml model create --name churn-model --path outputs/sklearn_model.pkl
                '''
            }
        }

        stage('Deploy Endpoint') {
            steps {
                sh '''
                    az ml online-endpoint create --file endpoint.yml
                '''
            }
        }
    }

    post {
        always {
            echo '✅ Azure ML Pipeline Completed!'
        }
    }
}
