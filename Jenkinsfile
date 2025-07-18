pipeline {
    agent any

    environment {
        AZ_CRED = credentials('AZURE_CREDENTIALS') // from Jenkins secrets
        AZ_SUBSCRIPTION_ID = 'your-subscription-id'
        AZ_RG = 'mlops-rg'
        AZ_WS = 'mlops-workspace'
    }

    stages {
        stage('Checkout Code') {
            steps {
                git 'https://github.com/your-username/azure-mlops-project.git'
            }
        }

        stage('Azure Login') {
            steps {
                sh '''
                    echo $AZ_CRED > azcreds.json
                    az login --service-principal -u $(jq -r .clientId azcreds.json) \
                        -p $(jq -r .clientSecret azcreds.json) \
                        --tenant $(jq -r .tenantId azcreds.json)
                    az account set --subscription $AZ_SUBSCRIPTION_ID
                '''
            }
        }

        stage('Upload Dataset') {
            steps {
                sh '''
                    az storage blob upload-batch \
                      --account-name mlopsstorageaccount \
                      --destination datasets \
                      --source dataset/
                '''
            }
        }

        stage('Submit Azure ML Training') {
            steps {
                sh '''
                    az ml job create --file azureml/job.yml \
                        --resource-group $AZ_RG \
                        --workspace-name $AZ_WS
                '''
            }
        }

        stage('Deploy Model') {
            steps {
                sh '''
                    az ml online-endpoint create --name churn-endpoint --file azureml/deploy.yml \
                        --resource-group $AZ_RG \
                        --workspace-name $AZ_WS
                '''
            }
        }
    }
}
