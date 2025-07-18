pipeline {
    agent any

    parameters {
        string(name: 'AZURE_CLIENT_ID', description: 'Azure Service Principal Client ID')
        string(name: 'AZURE_CLIENT_SECRET', description: 'Azure Service Principal Client Secret')
        string(name: 'AZURE_TENANT_ID', description: 'Azure Tenant ID')
        string(name: 'AZURE_SUBSCRIPTION_ID', description: 'Azure Subscription ID')
        string(name: 'AZURE_RESOURCE_GROUP', defaultValue: 'mlops-rg', description: 'Resource Group Name')
        string(name: 'AZURE_WORKSPACE_NAME', defaultValue: 'mlops-workspace', description: 'Azure ML Workspace Name')
        string(name: 'AZURE_REGION', defaultValue: 'eastus', description: 'Azure Region')
        string(name: 'DATA_FILE', defaultValue: 'customer_churn_100.csv', description: 'Dataset CSV File')
    }

    environment {
        AZURE_CLIENT_ID       = "${params.AZURE_CLIENT_ID}"
        AZURE_CLIENT_SECRET   = "${params.AZURE_CLIENT_SECRET}"
        AZURE_TENANT_ID       = "${params.AZURE_TENANT_ID}"
        AZURE_SUBSCRIPTION_ID = "${params.AZURE_SUBSCRIPTION_ID}"
        AZURE_RESOURCE_GROUP  = "${params.AZURE_RESOURCE_GROUP}"
        AZURE_WORKSPACE_NAME  = "${params.AZURE_WORKSPACE_NAME}"
        AZURE_REGION          = "${params.AZURE_REGION}"
        DATA_FILE             = "${params.DATA_FILE}"
    }

    stages {

        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Set Up Python Environment') {
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
                      --username $AZURE_CLIENT_ID \
                      --password $AZURE_CLIENT_SECRET \
                      --tenant $AZURE_TENANT_ID

                    az account set --subscription $AZURE_SUBSCRIPTION_ID
                '''
            }
        }

        stage('Upload Dataset to Azure ML') {
            steps {
                sh '''
                    echo "Uploading dataset to Azure ML default datastore..."

                    source venv/bin/activate
                    python3 upload_dataset.py \
                      --resource_group $AZURE_RESOURCE_GROUP \
                      --workspace_name $AZURE_WORKSPACE_NAME \
                      --region $AZURE_REGION \
                      --dataset $DATA_FILE
                '''
            }
        }

        stage('Train Model on Azure ML') {
            steps {
                sh '''
                    echo "Submitting training job to Azure ML..."

                    source venv/bin/activate
                    python3 train_on_azure.py \
                      --resource_group $AZURE_RESOURCE_GROUP \
                      --workspace_name $AZURE_WORKSPACE_NAME \
                      --region $AZURE_REGION
                '''
            }
        }
    }

    post {
        always {
            echo '✅ Jenkins pipeline execution complete.'
        }
    }
}
