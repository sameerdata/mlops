import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import tarfile

# Load dataset
df = pd.read_csv("dataset.csv")
df.drop("CustomerID", axis=1, inplace=True)
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split & Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Report
print("Classification Report:\n", classification_report(y_test, model.predict(X_test)))

# ✅ Save with exact name required by SageMaker
joblib.dump(model, "sklearn_model.pkl")

# ✅ Package correctly
with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("sklearn_model.pkl", arcname="sklearn_model.pkl")

print("✅ Model packaged as model.tar.gz")
