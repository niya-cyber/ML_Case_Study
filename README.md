# Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

Customer churn prediction is a machine learning classification problem that aims to identify customers who are likely to discontinue a service.

This project uses the Telco Customer Churn dataset to implement and compare four machine learning algorithms: Logistic Regression, Decision Tree, Random Forest, and XGBoost.

The project focuses on data preprocessing, exploratory data analysis, model training, performance evaluation, and feature interpretation to understand how different machine learning approaches perform in predicting customer churn.

## 🎯 Problem Statement

Telecommunication companies face challenges in retaining customers who may discontinue their services. Identifying customers who are likely to churn can help businesses understand customer behaviour and develop appropriate retention strategies.

This project investigates how effectively different machine learning algorithms can predict customer churn using customer demographics, subscribed services, contract information, and account details.

## 📂 Dataset

**Dataset:** Telco Customer Churn
**Source:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
**File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

| Property          | Description           |
| ----------------- | --------------------- |
| Domain            | Telecommunications    |
| Number of records | 7,043                 |
| Number of columns | 21                    |
| Target variable   | `Churn`               |
| Problem type      | Binary classification |
| Target classes    | Yes / No              |

The dataset contains customer information, including demographics, subscribed services, contract types, tenure, and billing details.

The target variable, `Churn`, indicates whether a customer has discontinued the service.

## 🎯 Objectives

* Perform exploratory data analysis (EDA) to understand customer churn patterns.
* Identify missing values, duplicates, and data-quality issues.
* Preprocess numerical and categorical features appropriately.
* Implement four machine learning classification algorithms.
* Evaluate model performance using multiple classification metrics.
* Compare the strengths and limitations of the selected algorithms.
* Analyse feature importance and model coefficients.
* Identify the algorithm that performs best according to the evaluation results.

## 🤖 Machine Learning Algorithms

Four supervised machine learning algorithms are implemented in this project.

### 1. Logistic Regression

Logistic Regression is used as the baseline classification model. It estimates the probability of customer churn based on the available customer features.

**Reason for selection:**

* Computationally efficient.
* Provides probability estimates.
* Relatively easy to interpret.
* Establishes a baseline for comparing more complex models.

### 2. Decision Tree

A Decision Tree predicts customer churn by learning a sequence of feature-based decision rules.

**Reason for selection:**

* Captures nonlinear relationships.
* Produces interpretable decision rules.
* Helps identify patterns associated with customer churn.
* Provides a useful comparison with ensemble learning algorithms.

### 3. Random Forest

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

**Reason for selection:**

* Captures complex relationships between customer features.
* Can generalise better than an individual decision tree.
* Provides feature importance estimates.
* Is suitable for structured datasets containing numerical and categorical features after appropriate preprocessing.

### 4. XGBoost

XGBoost is a gradient-boosting algorithm that builds decision trees sequentially to improve predictive performance.

**Reason for selection:**

* Models complex nonlinear relationships.
* Captures interactions between customer features.
* Includes regularisation mechanisms.
* Provides a strong candidate for comparison with other classification algorithms.

## ⚙️ Project Workflow

The project follows these stages:

1. Dataset loading and inspection.
2. Statistical summary and data-quality analysis.
3. Exploratory Data Analysis (EDA).
4. Missing-value and duplicate analysis.
5. Data type conversion and feature preparation.
6. Target variable encoding.
7. Numerical feature scaling where appropriate.
8. Categorical feature encoding.
9. Stratified training and testing split.
10. Model training using four algorithms.
11. Evaluation using classification metrics.
12. Confusion matrix visualisation.
13. ROC curve comparison.
14. Feature importance and coefficient analysis.
15. Saving model comparison results and trained models.

## 🧹 Data Preprocessing

The following preprocessing techniques are implemented.

* **Missing-value handling:** Numerical missing values are handled using median imputation, while categorical missing values are handled using the most frequent category.
* **Data type conversion:** The `TotalCharges` column is converted to numeric format because it is stored as text in the original dataset.
* **Identifier removal:** The `customerID` column is excluded from the model inputs because it is an identifier rather than a meaningful customer-behaviour feature.
* **Target encoding:** The `Churn` variable is converted into binary values, with `No = 0` and `Yes = 1`.
* **Categorical encoding:** One-hot encoding is used to convert categorical features into a numerical representation.
* **Feature scaling:** Standard scaling is applied to numerical features for Logistic Regression. Tree-based models use unscaled numerical features.
* **Data splitting:** Stratified sampling is used to maintain approximately the same class proportions in the training and testing sets.

Preprocessing is implemented using Scikit-learn pipelines and column transformers. This ensures that preprocessing transformations are fitted on the training data and subsequently applied to the test data, reducing the risk of data leakage.

## 🔬 Experimental Setup

The models are trained and evaluated using the following configuration.

| Parameter            | Configuration               |
| -------------------- | --------------------------- |
| Training set         | 80%                         |
| Testing set          | 20%                         |
| Random state         | 42                          |
| Splitting strategy   | Stratified train-test split |
| Classification type  | Binary classification       |
| Number of algorithms | 4                           |

The models use the following configurations:

* **Logistic Regression:** Maximum iterations set to 2,000.
* **Decision Tree:** Default classifier settings with a fixed random state.
* **Random Forest:** 300 decision trees.
* **XGBoost:** 300 estimators, maximum depth of 4, learning rate of 0.05, and subsampling parameters of 0.9.

These are the initial model configurations implemented in the notebook. Hyperparameter tuning and cross-validation can be introduced as further improvements.

## 📊 Model Evaluation

The four algorithms are evaluated using the following classification metrics.

| Metric           | Purpose                                                                                                  |
| ---------------- | -------------------------------------------------------------------------------------------------------- |
| Accuracy         | Measures the proportion of correctly classified customers.                                               |
| Precision        | Measures the proportion of predicted churners who actually churned.                                      |
| Recall           | Measures the proportion of actual churners correctly identified.                                         |
| F1-score         | Balances precision and recall.                                                                           |
| ROC-AUC          | Measures the model's ability to distinguish churners from non-churners across classification thresholds. |
| Confusion Matrix | Summarises correct and incorrect classifications.                                                        |

Since the dataset contains more non-churners than churners, accuracy alone is insufficient to evaluate model effectiveness.

Recall and F1-score are particularly important for assessing how effectively the models identify customers who discontinue their services.

The results from all four models are collected into a comparison table to facilitate performance analysis.

## 📈 Exploratory Data Analysis

The notebook includes visualisations to investigate customer churn patterns.

The analysis includes:

* **Customer Churn Distribution:** Examines the distribution of churners and non-churners.
* **Churn by Contract Type:** Compares customer churn counts across contract categories.
* **Monthly Charges by Churn Status:** Examines differences in monthly charges between churners and non-churners.
* **Customer Tenure by Churn Status:** Compares customer tenure across the two churn classes.

These visualisations help identify patterns and relationships that may be useful for understanding customer churn.

## 📉 Model Performance Visualisation

The notebook generates visualisations to compare the four trained algorithms.

### Confusion Matrices

A confusion matrix is generated for each algorithm to examine:

* True positives.
* True negatives.
* False positives.
* False negatives.

These results help identify the types of classification errors made by each model.

### ROC Curves

ROC curves are plotted for all four algorithms to compare their ability to distinguish between churners and non-churners across different classification thresholds.

The corresponding ROC-AUC values are included in the model comparison table.

## 🔍 Feature Interpretation

Feature interpretation is performed to investigate which customer characteristics contribute to the models' predictions.

The notebook examines:

* Logistic Regression coefficients.
* Decision Tree feature importance.
* Random Forest feature importance.
* XGBoost feature importance.

These analyses help identify features that influence predictions.

Feature importance indicates predictive association rather than causation, so the findings must be interpreted carefully.

## 💾 Saving Results and Trained Models

The project saves the model comparison results and trained models for future use.

### Model Comparison Results

The evaluation metrics are exported to:

`Telco_Churn_Model_Comparison_Results.csv`

This file can be opened in spreadsheet software for further analysis and visualisation.

### Trained Models

The four trained models are saved as Joblib files:

* `logistic_regression_model.joblib`
* `decision_tree_model.joblib`
* `random_forest_model.joblib`
* `xgboost_model.joblib`

Each saved pipeline contains its fitted preprocessing steps and classifier.

These files can be loaded later to reuse the trained models without repeating the entire training process, provided the necessary Python libraries and compatible environment are available.

Only load model files from trusted sources.

## 🛠️ Technologies Used

* **Python** - Implementation language.
* **Pandas** - Data loading and manipulation.
* **NumPy** - Numerical operations.
* **Matplotlib** - Data visualisation.
* **Seaborn** -Statistical visualisation.
* **Scikit-learn** - Preprocessing, model training, and evaluation.
* **XGBoost** -Gradient-boosting classification.
* **Joblib** -Saving trained models.
* **Jupyter Notebook** - Development and experimentation environment.

## 🚀 Future Enhancements

Potential improvements include:

* Hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`.
* Stratified cross-validation for more robust model evaluation.
* Class weighting and classification-threshold optimisation to improve churn detection.
* Comparison with additional algorithms such as LightGBM or CatBoost.
* Explainable AI techniques such as SHAP.
* Evaluation using business-oriented measures, such as the cost of missed churners.
* Validation on additional customer churn datasets.

Any oversampling technique, such as SMOTE, should be applied only within the training process to avoid data leakage.

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── Telco_Customer_Churn_ML_Case_Study.ipynb
├── Telco_Churn_Model_Comparison_Results.csv
├── logistic_regression_model.joblib
├── decision_tree_model.joblib
├── random_forest_model.joblib
├── xgboost_model.joblib
└── README.md
```
## 📊 Results

All four machine learning models were evaluated on the same stratified test set using accuracy, precision, recall, F1-score, and ROC-AUC.

### Model Performance Comparison

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   80.55% |    65.72% | 55.88% |   60.40% |  84.19% |
| Decision Tree       |   72.89% |    48.96% | 50.53% |   49.74% |  65.73% |
| Random Forest       |   78.07% |    61.09% | 47.86% |   53.67% |  81.79% |
| XGBoost             |   80.13% |    65.56% | 52.94% |   58.58% |  84.15% |

*Note: The values above are taken from the notebook's evaluation results. Precision, recall, and F1-score refer to the positive class, `Churn = Yes`.*

### Key Findings

* **Logistic Regression** achieved the highest accuracy, F1-score, and ROC-AUC among the four models.
* **XGBoost** achieved performance close to Logistic Regression, with a ROC-AUC of 84.15%.
* **Random Forest** performed better than the Decision Tree across the reported metrics, demonstrating the benefit of combining multiple trees in an ensemble.
* **Decision Tree** recorded the lowest accuracy and ROC-AUC, suggesting weaker predictive performance on this test set.
* Although Logistic Regression achieved the best overall results, its recall indicates that it still misses a considerable proportion of customers who actually churn.

## 🔍 Analysis and Discussion

### 1. Best Overall Model: Logistic Regression

Logistic Regression achieved an accuracy of **80.55%**, an F1-score of **60.40%**, and a ROC-AUC of **84.19%**.

It performed particularly well compared with the other evaluated models. Its ROC-AUC indicates a relatively strong ability to distinguish between customers who churn and those who remain.

Despite being a comparatively simple algorithm, Logistic Regression performed slightly better than XGBoost on this particular test set. This demonstrates that a more complex model does not necessarily produce better results for every dataset.

### 2. XGBoost: Competitive Performance

XGBoost achieved an accuracy of **80.13%** and a ROC-AUC of **84.15%**, making it a close competitor to Logistic Regression.

Its performance suggests that gradient boosting can capture useful patterns in customer data. However, with the configurations used in this experiment, it did not outperform Logistic Regression on the reported metrics.

Further hyperparameter tuning and cross-validation could help determine whether XGBoost's performance can be improved.

### 3. Random Forest: Moderate Performance

Random Forest achieved an accuracy of **78.07%** and a ROC-AUC of **81.79%**.

It outperformed the individual Decision Tree on the reported evaluation metrics. However, its recall of **47.86%** indicates that it identified fewer than half of the customers who actually churned.

This limitation matters in a customer-retention setting, where failing to identify a customer who is likely to leave may represent a missed opportunity for intervention.

### 4. Decision Tree: Lowest Predictive Performance

The Decision Tree achieved an accuracy of **72.89%** and a ROC-AUC of **65.73%**, the lowest reported values among the four models.

Its F1-score of **49.74%** also indicates a weaker balance between precision and recall compared with the other models.

Although Decision Trees can be useful for interpreting decision rules, this particular model configuration was less effective at predicting churn on the test set.

### 5. Why Accuracy Alone Is Not Enough

The dataset contains more customers who did not churn than customers who did. As a result, accuracy alone does not fully describe how effectively a model identifies churners.

For example, Logistic Regression achieved approximately **80.55% accuracy**, but its recall was **55.88%**. This means that it identified approximately 56% of the customers who actually churned and missed the remainder at the default classification threshold.

For a business focused on customer retention, recall may be particularly important because missed churners are potential customers the company fails to identify in advance. However, increasing recall may also increase false positives and the cost of contacting customers who would have stayed.

The appropriate model therefore depends not only on predictive performance but also on the business cost of missed churners, unnecessary retention offers, and other classification errors.

### 6. Overall Conclusion from the Results

Based on the reported test-set metrics, **Logistic Regression is the strongest overall model in this experiment**, with XGBoost performing very closely.

However, none of the evaluated models identifies every customer who churns. Future work could investigate classification-threshold optimisation, class weighting, cross-validation, and hyperparameter tuning to improve churn detection.

These findings apply to the current train-test split and model configurations. They should not be treated as proof that Logistic Regression will always outperform the other algorithms on different datasets or unseen customer populations.

## 🏁 Conclusion

This project implements and compares four machine learning algorithms—Logistic Regression, Decision Tree, Random Forest, and XGBoost—for predicting customer churn using the Telco Customer Churn dataset.

The workflow includes data inspection, exploratory data analysis, preprocessing, model training, performance evaluation, confusion matrix analysis, ROC curve comparison, and feature interpretation.

The comparative analysis provides a structured way to investigate the strengths and limitations of different classification approaches and determine which algorithm performs best according to the selected evaluation metrics.

The project demonstrates the application of machine learning techniques to customer churn prediction and provides a foundation for further work in model optimisation, explainability, and customer retention analysis.
