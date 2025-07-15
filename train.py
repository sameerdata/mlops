import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import tarfile

# 1. Load Dataset
df = pd.read_csv("dataset.csv")
df.drop("CustomerID", axis=1, inplace=True)
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])

X = df.drop("Churn", axis=1)
y = df["Churn"]

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
print("Classification Report:\n", classification_report(y_test, model.predict(X_test)))

# 5. Save as required file name
joblib.dump(model, "sklearn_model.pkl")

# 6. Package to model.tar.gz
with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("sklearn_model.pkl", arcname="sklearn_model.pkl")

print("✅ Model packaged as model.tar.gz")
