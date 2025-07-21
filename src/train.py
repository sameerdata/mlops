import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='Path to input CSV file')
args = parser.parse_args()

# Load dataset
print(f"📥 Loading dataset from {args.data_path}...")
df = pd.read_csv(args.data_path)

# Preprocess
print("🔧 Preprocessing dataset...")
df.drop("CustomerID", axis=1, inplace=True)
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split
print("🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
print("🧠 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("📊 Classification Report:")
print(classification_report(y_test, model.predict(X_test)))

# Save to AzureML outputs directory
output_dir = os.getenv("AZUREML_OUTPUT_DIR", "outputs")
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "sklearn_model.pkl")
joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")
