# Customer Churn Prediction Using Machine Learning

## 1. Project Overview

Customer churn prediction is a machine learning classification problem that aims to identify customers who are likely to discontinue a service.

This project uses the Telco Customer Churn dataset to develop and compare four machine learning algorithms. The objective is to evaluate their predictive performance and determine which algorithm is most effective at identifying customers who are likely to churn.

The project also explores customer characteristics that may be associated with churn through exploratory data analysis and feature importance analysis.

## 2. Problem Statement

Telecommunication companies face the challenge of retaining customers who may discontinue their services. Identifying customers at risk of churn can help businesses understand customer behaviour and develop appropriate retention strategies.

This project investigates how effectively different machine learning algorithms can predict customer churn using customer demographics, service subscriptions, contract information, and account details.

## 3. Dataset Description

**Dataset:** Telco Customer Churn

**Dataset file:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

**Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property          | Description                   |
| ----------------- | ----------------------------- |
| Domain            | Telecommunications            |
| Number of records | 7,043                         |
| Number of columns | 21                            |
| Target variable   | `Churn`                       |
| Problem type      | Binary classification         |
| Positive class    | `Yes` — customer churned      |
| Negative class    | `No` — customer did not churn |

### Target Variable Distribution

| Churn Status | Number of Customers | Percentage |
| ------------ | ------------------: | ---------: |
| No           |               5,174 |     73.46% |
| Yes          |               1,869 |     26.54% |
| **Total**    |           **7,043** |   **100%** |

The dataset contains a mixture of numerical and categorical features describing customer demographics, subscribed services, contract details, and billing information.

## 4. Objectives

The main objectives of this project are:

* To understand customer churn patterns through exploratory data analysis.
* To clean and preprocess the dataset appropriately.
* To implement four machine learning classification algorithms.
* To evaluate model performance using multiple classification metrics.
* To compare the strengths and limitations of the selected algorithms.
* To identify features associated with churn predictions.
* To determine which algorithm performs best on the selected dataset.

## 5. Machine Learning Algorithms

Four supervised machine learning algorithms are used in this study.

### 5.1 Logistic Regression

Logistic Regression is used as the baseline classification model. It estimates the probability of customer churn based on the available features.

**Reason for selection:**

* Computationally efficient.
* Provides probability estimates.
* Relatively easy to interpret.
* Establishes a baseline for comparison with more complex models.

### 5.2 Decision Tree

A Decision Tree predicts customer churn by learning a sequence of feature-based decision rules.

**Reason for selection:**

* Captures nonlinear relationships.
* Produces interpretable decision rules.
* Handles numerical features and appropriately encoded categorical features.
* Provides a useful comparison against ensemble methods.

### 5.3 Random Forest

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

**Reason for selection:**

* Captures complex relationships between customer characteristics.
* Often generalises better than an individual decision tree.
* Provides feature importance estimates.
* Performs well on many structured datasets.

### 5.4 XGBoost

XGBoost is a gradient-boosting algorithm that builds decision trees sequentially to improve predictive performance.

**Reason for selection:**

* Models complex nonlinear relationships.
* Can capture interactions between customer features.
* Includes regularisation mechanisms.
* Provides a strong candidate for comparison with the other classifiers.

## 6. Project Workflow

The project follows these stages:

1. Dataset loading and inspection.
2. Data cleaning and validation.
3. Exploratory Data Analysis (EDA).
4. Missing-value and duplicate analysis.
5. Feature preparation and target encoding.
6. Categorical feature encoding.
7. Numerical feature scaling where appropriate.
8. Stratified training and testing split.
9. Model training.
10. Model evaluation and comparison.
11. Confusion matrix and ROC curve analysis.
12. Feature interpretation.
13. Conclusions and limitations.

## 7. Data Preprocessing

The following preprocessing techniques are applied where appropriate:

* **Missing values:** Numerical missing values are handled using median imputation, while categorical missing values are handled using the most frequent category.
* **Data type conversion:** The `TotalCharges` column is converted to numeric format because it is stored as text in the original dataset.
* **Identifier removal:** The `customerID` column is excluded from the model inputs because it is an identifier rather than a meaningful customer-behaviour feature.
* **Categorical encoding:** One-hot encoding is used to represent categorical variables numerically.
* **Feature scaling:** Standard scaling is applied to numerical features for Logistic Regression. Tree-based models use unscaled numerical features.
* **Target encoding:** `Churn` is converted to binary values, with `No = 0` and `Yes = 1`.
* **Data splitting:** The dataset is divided into training and testing sets using stratified sampling.

Preprocessing is implemented using pipelines so that transformations are fitted on the training data rather than the entire dataset, reducing the risk of data leakage.

## 8. Experimental Setup

The initial experiment uses the following configuration:

| Parameter            | Value                       |
| -------------------- | --------------------------- |
| Training set         | 80%                         |
| Testing set          | 20%                         |
| Random state         | 42                          |
| Splitting strategy   | Stratified train-test split |
| Classification type  | Binary classification       |
| Number of algorithms | 4                           |

The initial experiment uses baseline model configurations. Hyperparameter tuning and cross-validation can be conducted as additional experiments.

## 9. Evaluation Metrics

The models are evaluated using the following metrics:

| Metric           | Purpose                                                                                                  |
| ---------------- | -------------------------------------------------------------------------------------------------------- |
| Accuracy         | Measures the proportion of correctly classified customers.                                               |
| Precision        | Measures the proportion of predicted churners who actually churned.                                      |
| Recall           | Measures the proportion of actual churners correctly identified.                                         |
| F1-score         | Balances precision and recall.                                                                           |
| ROC-AUC          | Measures the model's ability to distinguish churners from non-churners across classification thresholds. |
| Confusion Matrix | Summarises correct and incorrect classifications.                                                        |

Since the dataset is imbalanced, accuracy alone is insufficient to evaluate model effectiveness. Recall and F1-score are particularly useful for understanding how effectively the models identify customers who churn.

## 10. Initial Experimental Results

The following results were obtained from the initial experiment using the specified train-test split and baseline model configurations.

| Algorithm           | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   80.55% |    65.72% | 55.88% |   60.40% |  84.19% |
| Decision Tree       |   72.89% |    48.96% | 50.53% |   49.74% |  65.73% |
| Random Forest       |   78.07% |    61.09% | 47.86% |   53.67% |  81.79% |
| XGBoost             |   80.13% |    65.56% | 52.94% |   58.58% |  84.15% |

**Note:** These are baseline results from one stratified train-test split. They are not hyperparameter-tuned results, and the rankings may change under different configurations or validation procedures.

### Initial Observations

* Logistic Regression achieved the highest accuracy, recall, F1-score, and ROC-AUC in the initial experiment.
* XGBoost produced results close to Logistic Regression, particularly in ROC-AUC.
* Random Forest achieved reasonable discrimination performance but had lower churn recall.
* Decision Tree had the lowest accuracy and ROC-AUC among the four models in this experiment.

These observations suggest that a more complex model does not necessarily outperform a simpler model on every dataset.

Further tuning and validation are needed before drawing stronger conclusions.

## 11. Exploratory Data Analysis

The following visualisations are included or planned for the analysis:

* Customer churn distribution.
* Churn rate by contract type.
* Monthly charges by churn status.
* Customer tenure by churn status.
* Confusion matrices for all four algorithms.
* ROC curves for model comparison.
* Feature importance plots.

These visualisations help explain the data, assess model performance, and identify patterns associated with churn.

## 12. Feature Interpretation

Feature interpretation is used to investigate which customer characteristics contribute to the model's predictions.

The analysis includes:

* Logistic Regression coefficients.
* Decision Tree feature importance.
* Random Forest feature importance.
* XGBoost feature importance.

Feature importance indicates predictive relationships rather than causation. The results should therefore be interpreted carefully, especially when multiple features are correlated.

## 13. Limitations

The project has several limitations:

* The initial results are based on a single train-test split.
* Model performance may vary with different hyperparameters and data splits.
* The dataset contains an imbalance between churners and non-churners.
* Feature importance does not establish causal relationships.
* Findings from one telecommunications dataset may not generalise to other businesses or customer populations.

## 14. Future Enhancements

Potential extensions include:

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
* Stratified cross-validation.
* Class-weighting and threshold optimisation to improve churn detection.
* Comparison with additional algorithms such as LightGBM or CatBoost.
* Explainable AI techniques such as SHAP.
* Evaluation using business-oriented measures, such as the cost of missed churners.
* Validation on additional customer churn datasets.

Any oversampling method, such as SMOTE, should be applied only within the training process to prevent data leakage.

## 15. Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook

## 16. Project Files

The project can be organised as follows:

```text
Customer-Churn-Prediction/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── Telco_Customer_Churn_ML_Case_Study.ipynb
├── Telco_Churn_Model_Comparison_Results.csv
└── README.md
```

## 17. Conclusion

This project investigates customer churn prediction using four machine learning algorithms: Logistic Regression, Decision Tree, Random Forest, and XGBoost.

The initial experimental results show that Logistic Regression achieved the strongest overall performance on the selected test split, while XGBoost produced comparable ROC-AUC results. Decision Tree and Random Forest demonstrated different performance characteristics, highlighting the importance of evaluating multiple algorithms using appropriate metrics.

Further hyperparameter tuning, validation, and feature analysis can provide a more comprehensive understanding of the models' strengths and limitations.

The study demonstrates how machine learning can support customer churn analysis and provide insights that may help businesses understand customer retention challenges.
