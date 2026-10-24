# Telco Customer Churn Prediction

## Project Overview

This project uses machine learning to predict customer churn in a telecommunications company. The objective is to analyze customer data, identify patterns associated with churn, and compare different machine learning models.

## Dataset

- **Dataset:** Telco Customer Churn
- **Total Customers:** 7,043
- **Original Columns:** 21
- **Features Used:** 19
- **Target Variable:** `Churn`

The target variable is encoded as:
- `0` — Customer stayed
- `1` — Customer churned

## Project Workflow

1. Data loading and initial exploration
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Train-test split
5. Feature preprocessing
6. Model training
7. Model evaluation and comparison
8. ROC curve analysis
9. Conclusion

## Data Cleaning and Preprocessing

The following preprocessing steps were performed:

- Removed `customerID` because it is an identifier rather than a useful predictive feature.
- Converted `TotalCharges` from text to numeric.
- Identified 11 missing `TotalCharges` values associated with customers whose tenure was zero and replaced them with `0`.
- Encoded the target variable as `No = 0` and `Yes = 1`.
- Retained all 7,043 customer records.
- Split the dataset into training and testing sets using an 80:20 ratio.
- Used stratified sampling to preserve the churn distribution.

### Train-Test Split

- **Training Records:** 5,634
- **Testing Records:** 1,409
- **Random State:** 42

## Exploratory Data Analysis

### 1. Churn Rate by Contract Type

| Contract Type | Churn Rate |
|---|---:|
| Month-to-month | 42.71% |
| One year | 11.27% |
| Two year | 2.83% |

Customers with month-to-month contracts had a substantially higher churn rate than customers with longer-term contracts.

### 2. Average Tenure by Churn Status

| Churn Status | Average Tenure |
|---|---:|
| Stayed | 37.57 months |
| Churned | 17.98 months |

Customers who churned had a shorter average tenure than customers who stayed.

### 3. Average Monthly Charges by Churn Status

| Churn Status | Average Monthly Charges |
|---|---:|
| Stayed | 61.27 |
| Churned | 74.44 |

Customers who churned had higher average monthly charges than customers who stayed.

These findings show associations in the dataset and do not establish causation.

## Feature Preprocessing

The dataset contains numerical and categorical features.

### Numerical Features
- `SeniorCitizen`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

Numerical features were processed using median imputation and standardization.

### Categorical Features

Categorical features were processed using most-frequent-value imputation and one-hot encoding.

The preprocessing steps were incorporated into scikit-learn pipelines to help prevent data leakage from the test set.

## Machine Learning Models

Four classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

All models were evaluated using the same train-test split.

## Model Evaluation

The models were evaluated using the following metrics:

- **Accuracy:** Proportion of all predictions that were correct.
- **Precision:** Proportion of predicted churners who actually churned.
- **Recall:** Proportion of actual churners correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between customers who churn and customers who stay.

## Model Comparison Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 60.40% | 84.21% |
| Decision Tree | 72.11% | 47.55% | 49.20% | 48.36% | 64.77% |
| Random Forest | 78.64% | 62.54% | 48.66% | 54.74% | 81.85% |
| XGBoost | 80.34% | 66.33% | 52.67% | 58.72% | 84.45% |

## Results Analysis

- **Logistic Regression** achieved the highest accuracy, recall, and F1-score.
- **XGBoost** achieved the highest precision and ROC-AUC.
- **Random Forest** performed better than the single Decision Tree across all five evaluation metrics.
- **Decision Tree** recorded the lowest ROC-AUC among the four models.

Logistic Regression and XGBoost were the strongest models in this baseline comparison, with different strengths.

The results are based on a single stratified train-test split. They do not establish that one model will always perform best on other datasets.

## Conclusion

This project demonstrated a machine learning workflow for customer churn prediction, including data cleaning, exploratory analysis, feature preprocessing, model training, and evaluation.

The exploratory analysis identified relationships between churn and contract type, customer tenure, and monthly charges.

Logistic Regression achieved the highest accuracy, recall, and F1-score, while XGBoost achieved the highest precision and ROC-AUC. The most appropriate model depends on the business objective and the trade-off between missing potential churners and incorrectly flagging customers who would stay.

## Future Improvements

Possible future improvements include:

- Hyperparameter tuning
- Classification threshold optimization
- Cross-validation
- Further analysis of factors associated with churn
- Evaluation of potential business costs associated with false positives and false negatives

## Tools and Technologies

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost
- Jupyter Notebook

## Project Structure

```text
ML_Case_Study/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── Telco_Customer_Churn.ipynb
└── README.md
