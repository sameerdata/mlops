import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import tarfile

# 1. Load Dataset
print("📥 Loading dataset...")
df = pd.read_csv("dataset.csv")

# 2. Preprocessing
print("🧹 Preprocessing...")
df.drop("CustomerID", axis=1, inplace=True)
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])  # Male=1, Female=0

X = df.drop("Churn", axis=1)
y = df["Churn"]

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training
print("🧠 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluation
print("📊 Classification Report:")
print(classification_report(y_test, model.predict(X_test)))

# 6. Save Model as .pkl
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")

# 7. Package as .tar.gz for SageMaker
with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("model.pkl", arcname="model.pkl")
print("📦 Packaged model as model.tar.gz (SageMaker compatible)")
