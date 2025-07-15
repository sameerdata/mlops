import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import tarfile
import os

# 1. Load and preprocess dataset
print("📥 Loading dataset...")
df = pd.read_csv("dataset.csv")
df.drop("CustomerID", axis=1, inplace=True)
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
print("🧠 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
print("📊 Classification Report:")
print(classification_report(y_test, model.predict(X_test)))

# 5. Save model as required by SageMaker
model_filename = "sklearn_model.pkl"
joblib.dump(model, model_filename)
print(f"✅ Model saved as {model_filename}")

# 6. Remove old archive if exists
archive_path = "model.tar.gz"
if os.path.exists(archive_path):
    os.remove(archive_path)

# 7. Package model and inference script into model.tar.gz
print("📦 Packaging model and inference.py into model.tar.gz...")
with tarfile.open(archive_path, "w:gz") as tar:
    tar.add(model_filename, arcname="sklearn_model.pkl")
    tar.add("inference.py", arcname="code/inference.py")

print("✅ model.tar.gz ready for SageMaker deployment.")
