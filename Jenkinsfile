pipeline {
    agent any

    parameters {
        string(name: 'AZURE_CLIENT_ID', defaultValue: '', description: 'Azure Service Principal Client ID')
        string(name: 'AZURE_CLIENT_SECRET', defaultValue: '', description: 'Azure Service Principal Client Secret')
        string(name: 'AZURE_SUBSCRIPTION_ID', defaultValue: '', description: 'Azure Subscription ID')
        string(name: 'AZURE_TENANT_ID', defaultValue: '', description: 'Azure Tenant ID')
    }

    environment {
        AZURE_CLIENT_ID = "${params.AZURE_CLIENT_ID}"
        AZURE_CLIENT_SECRET = "${params.AZURE_CLIENT_SECRET}"
        AZURE_SUBSCRIPTION_ID = "${params.AZURE_SUBSCRIPTION_ID}"
        AZURE_TENANT_ID = "${params.AZURE_TENANT_ID}"
    }

    stages {

        stage('Install Dependencies') {
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
                      --username "$AZURE_CLIENT_ID" \
                      --password "$AZURE_CLIENT_SECRET" \
                      --tenant "$AZURE_TENANT_ID"
                      
                    az account set --subscription "$AZURE_SUBSCRIPTION_ID"
                '''
            }
        }

        stage('Upload Dataset to Azure') {
            steps {
                sh '''
                    source venv/bin/activate
                    python upload_dataset.py
                '''
            }
        }

        stage('Train Model on Azure') {
            steps {
                sh '''
                    source venv/bin/activate
                    python train_on_azure.py
                '''
            }
        }
    }

    post {
        always {
            echo "📦 Jenkins Azure ML pipeline complete!"
        }
    }
}
