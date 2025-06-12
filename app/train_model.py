import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load data
df = pd.read_csv("data/bank_additional_full.csv", sep=";")

# Target variable
df["y"] = df["y"].map({"yes": 1, "no": 0})

# Encode categorical variables
cat_cols = df.select_dtypes(include="object").columns
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders
with open("app/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Train/test split
X = df.drop(columns=["y"])
y = df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("app/bank_model.pkl", "wb") as f:
    pickle.dump(model, f)
