# ==============================
# IMPORT LIBRARIES
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(
    '/kaggle/input/datasets/ninzaami/loan-predication/train_u6lujuX_CVtuZ9i (1).csv'
)

# ==============================
# DATA PREPROCESSING
# ==============================

# Drop Loan_ID column
if 'Loan_ID' in df.columns:
    df.drop('Loan_ID', axis=1, inplace=True)

# Handle missing values
for col in df.columns:

    # For categorical columns
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])

    # For numerical columns
    else:
        df[col] = df[col].fillna(df[col].median())

# Encode categorical columns
for col in df.columns:

    if df[col].dtype == 'object':

        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# ==============================
# FEATURES & TARGET
# ==============================

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# MODELS
# ==============================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=500,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )
}

# ==============================
# TRAIN & EVALUATE MODELS
# ==============================

for name, model in models.items():

    print("\n" + "="*60)
    print(f"MODEL : {name}")
    print("="*60)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Macro F1 Score
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\nAccuracy Score : {accuracy:.4f}")
    print(f"Macro F1 Score : {f1:.4f}")

    # Classification Report
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:\n")
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Rejected', 'Approved'],
        yticklabels=['Rejected', 'Approved']
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")

    plt.show()

# ==============================
# FINAL MODEL - XGBOOST
# ==============================

print("\n" + "="*60)
print("FINAL MODEL : XGBOOST WITH CROSS VALIDATION")
print("="*60)

final_model = XGBClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

# Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    final_model,
    X,
    y,
    cv=cv,
    scoring='f1_macro'
)

print("\nF1 Macro Scores:\n")
print(scores)

print(f"\nAverage F1 Macro Score : {scores.mean():.4f}")

# ==============================
# FEATURE IMPORTANCE
# ==============================

# Train final model
final_model.fit(X_train, y_train)

# Create importance dataframe
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': final_model.feature_importances_
})

# Sort values
importance = importance.sort_values(
    by='Importance',
    ascending=False
)

# Plot Feature Importance
plt.figure(figsize=(8, 5))

sns.barplot(
    data=importance,
    x='Importance',
    y='Feature'
)

plt.title("Feature Importance - XGBoost")

plt.show()
