# Telco Customer Churn Prediction

## 1. Project Overview

Customer churn occurs when a customer stops using a company's services. Predicting churn helps telecommunications companies identify customers who may leave and develop suitable customer-retention strategies.

The objective of this project is to build and evaluate machine learning models that predict whether a telecommunications customer will churn.

## 2. Dataset

- **Dataset:** Telco Customer Churn
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Number of records:** 7,043
- **Original number of columns:** 21
- **Target variable:** `Churn`

The target variable is encoded as:
- `No = 0`
- `Yes = 1`

The `customerID` column is removed because it is an identifier rather than a meaningful predictive feature. The `TotalCharges` column is converted to numeric format, and missing values are handled during data cleaning.

## 3. Machine Learning Models

Four classification algorithms are used:

1. **Logistic Regression** — A linear classification algorithm used to predict the probability of customer churn.
2. **Decision Tree** — A model that learns decision rules from customer data.
3. **Random Forest** — An ensemble model that combines multiple decision trees.
4. **XGBoost** — A gradient-boosting algorithm that builds decision trees sequentially to improve predictions.

Class weighting is used to address the imbalance between customers who churn and those who remain.

## 4. Project Workflow

The project follows these steps:

1. Import the required Python libraries.
2. Load and inspect the dataset.
3. Examine missing values, duplicates, and the churn distribution.
4. Clean the data and encode the target variable.
5. Perform Exploratory Data Analysis (EDA).
6. Split the dataset into training and testing sets.
7. Preprocess numerical and categorical features.
8. Train the four machine learning models.
9. Evaluate and compare model performance.
10. Visualise evaluation metrics using charts.
11. Analyse confusion matrices and ROC curves.
12. Investigate feature importance using Random Forest permutation importance.
13. Summarise the results and limitations.

## 5. Exploratory Data Analysis

EDA is used to understand the dataset and investigate relationships between customer characteristics and churn.

The analysis includes:
- Customer churn distribution.
- Churn rate by contract type.
- Customer tenure by churn status.
- Monthly charges by churn status.

These visualisations help identify patterns associated with customer churn. However, observed relationships do not necessarily establish causation.

## 6. Data Preprocessing

The following preprocessing techniques are applied:

- **Missing-value handling:** Missing numerical values are imputed using the median, while missing categorical values are imputed using the most frequent category.
- **Feature scaling:** `StandardScaler` standardises numerical features.
- **One-Hot Encoding:** Converts categorical variables into numerical representations.
- **Train-test split:** The dataset is divided into 80% training data and 20% testing data.
- **Stratification:** Maintains approximately the same churn-class proportions in both sets.

Preprocessing is incorporated into machine learning pipelines to ensure that transformations are fitted using the training data.

## 7. Model Evaluation

The models are evaluated using the following metrics:

| Metric | Description |
|---|---|
| Accuracy | Proportion of predictions that are correct |
| Precision | Proportion of predicted churners who actually churn |
| Recall | Proportion of actual churners correctly identified |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Measures how well the model distinguishes churners from non-churners across classification thresholds |

Confusion matrices and ROC curves are also used to compare the models.

Since the dataset is imbalanced, accuracy is not considered the only measure of model performance.

## 8. Feature Importance

Random Forest permutation importance is used to investigate which features contribute most to predictive performance.

Permutation importance measures how much a model's performance changes when a feature's values are shuffled.

Features with larger positive importance scores generally provide more useful predictive information to the fitted model.

Feature importance indicates predictive usefulness, not proof of causation.

## 9. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook
