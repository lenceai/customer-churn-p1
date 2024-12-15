"""
constants.py

This module contains all constant values used across the project for predicting customer churn.
These constants include file paths, column names, model parameters, and other configuration values.

Author: Claude
Date: December 15, 2024
"""

# File paths
DATA_PATH = "./data/bank_data.csv"
MODEL_PATH = "./models/"
EDA_IMAGES_PATH = "./images/eda/"
RESULTS_IMAGES_PATH = "./images/results/"
LOG_PATH = "./logs/churn_library.log"

# Model file names
RFC_MODEL_NAME = "rfc_model.pkl"
LOGISTIC_MODEL_NAME = "logistic_model.pkl"

# EDA plot file names
CHURN_HISTOGRAM = "churn_histogram.png"
CUSTOMER_AGE_HISTOGRAM = "customer_age_histogram.png"
MARITAL_STATUS_BAR = "marital_status_counts.png"
TOTAL_TRANSACTION_HISTOGRAM = "total_transaction_histogram.png"
CORRELATION_HEATMAP = "heatmap.png"

# Results plot file names
RFC_CLASSIFICATION_REPORT = "rf_classification_report.png"
LR_CLASSIFICATION_REPORT = "lr_classification_report.png"
FEATURE_IMPORTANCE_PLOT = "cv_feature_importance.png"
LRC_ROC_CURVE = "lrc_roc_curve.png"
ROC_CURVES_COMPARISON = "lrc_rfc_roc_curves.png"

# Column names
CATEGORICAL_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status', 
    'Income_Category',
    'Card_Category'
]

QUANT_COLUMNS = [
    'Customer_Age',
    'Dependent_count', 
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

KEEP_COLS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count', 
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio',
    'Gender_Churn',
    'Education_Level_Churn',
    'Marital_Status_Churn', 
    'Income_Category_Churn',
    'Card_Category_Churn'
]

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Random Forest parameters for GridSearchCV
RF_PARAM_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
}

# Logistic Regression parameters
LR_SOLVER = 'lbfgs'
LR_MAX_ITER = 3000

# Plot settings
PLT_FIGURE_SIZE = (20, 10)
FEATURE_IMPORTANCE_FIG_SIZE = (20, 5)
CLASSIFICATION_REPORT_FIG_SIZE = (5, 5)
