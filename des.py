# =========================
# 📦 IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# 📂 LOAD DATA
# =========================
df = pd.read_csv('/content/drive/MyDrive/loan_data.csv')  # change path if needed

# =========================
# 🧹 DATA PREPROCESSING
# =========================

# Drop unnecessary column
if 'Loan_ID' in df.columns:
    df.drop('Loan_ID', axis=1, inplace=True)

# Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# =========================
# 🎯 SPLIT DATA
# =========================
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 🔍 MODEL + GRID SEARCH (CV)
# =========================
model = LogisticRegression(
    penalty='l2',
    class_weight='balanced',
    random_state=42
)

param_grid = {
    'C': [0.1, 0.5, 1],
    'max_iter': [100, 200, 300]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    model,
    param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1
)

grid.fit(X_train, y_train)

# =========================
# 🏆 BEST MODEL
# =========================
best_model = grid.best_estimator_

# =========================
# 📊 EVALUATION
# =========================
y_pred = best_model.predict(X_test)

print("Best Parameters:", grid.best_params_)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
