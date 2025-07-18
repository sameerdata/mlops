from azureml.core import Workspace, Dataset
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# 1. Load dataset from Azure ML workspace
print("📥 Loading dataset from Azure ML...")
ws = Workspace.from_config()
dataset = Dataset.get_by_name(ws, 'customer_churn_dataset')
df = dataset.to_pandas_dataframe()

# 2. Preprocess
print("🔧 Preprocessing dataset...")
df.drop("CustomerID", axis=1, inplace=True)
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 3. Train/test split
print("🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
print("🧠 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate model
print("📊 Classification Report:")
print(classification_report(y_test, model.predict(X_test)))

# 6. Save model
model_filename = "sklearn_model.pkl"
joblib.dump(model, model_filename)
print(f"✅ Model saved as {model_filename}")
