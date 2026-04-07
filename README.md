# 🏦 Loan Prediction ML Project

## 📌 Overview

This project focuses on building a Machine Learning model to predict whether a loan application will be **approved or rejected** based on applicant details.

The dataset is sourced from Kaggle and includes features such as income, loan amount, credit history, and more.

---

## 🎯 Problem Statement

Predict the **Loan Status**:

* `1` → Approved
* `0` → Rejected

This is a **binary classification problem**.

---

## 📂 Dataset Features

Some important features used:

* ApplicantIncome
* CoapplicantIncome
* LoanAmount
* Credit_History
* Gender, Education, Property_Area

---

## 🛠️ Tech Stack

* Python 🐍
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn

---

## 🔍 Approach

### 1️⃣ Data Preprocessing

* Handled missing values
* Encoded categorical variables
* Removed unnecessary columns (e.g., Loan_ID)

---

### 2️⃣ Exploratory Data Analysis (EDA)

* Analyzed feature distributions
* Checked class imbalance
* Studied relationship with target variable

---

### 3️⃣ Handling Imbalance

* Used `class_weight='balanced'` in Logistic Regression

---

### 4️⃣ Model Building
Models experimented with:
- Logistic Regression ✅ (Final Model)
- Random Forest
- Gradient Boosting
- XGBoost

All models were evaluated using accuracy, precision, recall, and macro F1-score.
---

### 5️⃣ Model Tuning

* Applied **L2 Regularization**
* Tuned hyperparameter `C`
* Optimized `max_iter`
* Used **Cross-Validation (Stratified K-Fold)**

---

## 🏆 Final Model

**Logistic Regression (Best Model)**

```python
LogisticRegression(
    penalty='l2',
    C=0.5,
    class_weight='balanced',
    random_state=42,
    max_iter=best_value
)
```

---

## 📊 Final Results

| Metric         | Score    |
| -------------- | -------- |
| Accuracy       | **0.77** |
| Macro F1 Score | **0.73** |

### Classification Report:

```
Class 0 (Rejected): Precision = 0.58, Recall = 0.70, F1 = 0.64  
Class 1 (Approved): Precision = 0.87, Recall = 0.79, F1 = 0.83  
```

---

## 📈 Key Insights
- Multiple models were tested including Logistic Regression, Random Forest, Gradient Boosting, and XGBoost  
- Logistic Regression performed best due to the dataset’s small size and mostly linear relationships  
- Tree-based models (RF, GB, XGBoost) showed lower or unstable performance, likely due to overfitting  
- Handling class imbalance significantly improved model fairness  
- Regularization (C=0.5) improved generalization  
- Cross-validation ensured model stability  
---

## ⚠️ Challenges

* Imbalanced dataset
* Small dataset size
* Overfitting in tree-based models

---

## 🚀 Future Improvements

* Apply SMOTE for better imbalance handling
* Try advanced models like XGBoost
* Perform feature engineering (income ratios, etc.)
* Deploy model using Flask/Streamlit

---

## 💡 Conclusion

A well-tuned Logistic Regression model provided the best balance between accuracy and fairness, making it suitable for real-world loan approval prediction.

---

## 🙌 Author

**Keerthi**
GitHub: https://github.com/allianceprokeerthi-cmd
