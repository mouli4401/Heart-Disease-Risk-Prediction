# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("heart_data.csv")

# Encode categorical
df["sex"] = df["sex"].map({"M": 1, "F": 0})
df["is_smoking"] = df["is_smoking"].map({"YES": 1, "NO": 0})

# Fill missing values
df.fillna(df.median(), inplace=True)

# Features & Target
X = df.drop(columns=["id", "TenYearCHD"])
y = df["TenYearCHD"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Save model & scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model trained and saved ✅")
