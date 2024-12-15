"""
churn_library.py

This module contains the ChurnPredictor class which implements the customer churn prediction pipeline.

Author: Claude
Date: December 15, 2024
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, RocCurveDisplay
import constants as const

# Configure logging
logging.basicConfig(
    filename=const.LOG_PATH,
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s %(levelname)s: %(message)s'
)

class ChurnPredictor:
    """A class to predict customer churn using machine learning models."""

    def __init__(self):
        """Initialize ChurnPredictor with empty attributes."""
        self.df = None
        self.X_train = None 
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rfc_model = None
        self.lr_model = None
        
        # Create required directories
        for directory in [const.MODEL_PATH, const.EDA_IMAGES_PATH, 
                         const.RESULTS_IMAGES_PATH, os.path.dirname(const.LOG_PATH)]:
            os.makedirs(directory, exist_ok=True)

    def import_data(self, pth=const.DATA_PATH):
        """Import data from csv file."""
        try:
            self.df = pd.read_csv(pth)
            self.df['Churn'] = self.df['Attrition_Flag'].map(
                {"Existing Customer": 0, "Attrited Customer": 1})
            logging.info("Data import successful. Shape: %s", str(self.df.shape))
            return True
        except FileNotFoundError:
            logging.error("Data import failed: File not found at %s", pth)
            return False
        except Exception as err:
            logging.error("Data import failed: %s", str(err))
            return False

    def perform_eda(self):
        """Perform EDA on the data and save figures."""
        try:
            # Set style
            sns.set_style("whitegrid")
            
            # Churn histogram
            plt.figure(figsize=const.PLT_FIGURE_SIZE)
            self.df['Churn'].hist()
            plt.title('Distribution of Churn')
            plt.xlabel('Churn')
            plt.ylabel('Count')
            plt.savefig(os.path.join(const.EDA_IMAGES_PATH, const.CHURN_HISTOGRAM))
            plt.close()
            
            # Customer age histogram
            plt.figure(figsize=const.PLT_FIGURE_SIZE)
            self.df['Customer_Age'].hist()
            plt.title('Distribution of Customer Age')
            plt.xlabel('Age')
            plt.ylabel('Count')
            plt.savefig(os.path.join(const.EDA_IMAGES_PATH, const.CUSTOMER_AGE_HISTOGRAM))
            plt.close()
            
            # Marital status bar plot
            plt.figure(figsize=const.PLT_FIGURE_SIZE)
            self.df.Marital_Status.value_counts(normalize=True).plot(kind='bar')
            plt.title('Marital Status Distribution')
            plt.xlabel('Status')
            plt.ylabel('Proportion')
            plt.tight_layout()
            plt.savefig(os.path.join(const.EDA_IMAGES_PATH, const.MARITAL_STATUS_BAR))
            plt.close()
            
            # Total transaction distribution
            plt.figure(figsize=const.PLT_FIGURE_SIZE)
            sns.histplot(data=self.df, x='Total_Trans_Ct', kde=True)
            plt.title('Distribution of Total Transactions')
            plt.xlabel('Transaction Count')
            plt.ylabel('Count')
            plt.savefig(os.path.join(const.EDA_IMAGES_PATH, const.TOTAL_TRANSACTION_HISTOGRAM))
            plt.close()
            
            # Correlation heatmap
            plt.figure(figsize=const.PLT_FIGURE_SIZE)
            numeric_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
            sns.heatmap(self.df[numeric_columns].corr(), annot=False, cmap='Dark2_r', linewidths=2)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(os.path.join(const.EDA_IMAGES_PATH, const.CORRELATION_HEATMAP))
            plt.close()
            
            logging.info("EDA completed successfully")
            return True
        except Exception as err:
            logging.error("EDA failed: %s", str(err))
            return False

    def encoder_helper(self):
        """Helper function to encode categorical variables."""
        try:
            for category in const.CATEGORICAL_COLUMNS:
                # Calculate mean churn for each category value
                category_means = self.df.groupby(category)['Churn'].mean()
                # Create new column name
                new_column = f'{category}_Churn'
                # Map means to new column
                self.df[new_column] = self.df[category].map(category_means)
            
            logging.info("Categorical encoding completed successfully")
            return True
        except Exception as err:
            logging.error("Categorical encoding failed: %s", str(err))
            return False

    def perform_feature_engineering(self):
        """Perform feature engineering and split data."""
        try:
            # Verify all necessary columns exist
            missing_cols = [col for col in const.KEEP_COLS if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            # Select features and target
            X = self.df[const.KEEP_COLS]
            y = self.df['Churn']
            
            # Train test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=const.TEST_SIZE, random_state=const.RANDOM_STATE)
            
            logging.info("Feature engineering completed successfully")
            return True
        except Exception as err:
            logging.error("Feature engineering failed: %s", str(err))
            return False

    def train_models(self):
        """Train and store models."""
        try:
            # Verify we have data to train on
            if self.X_train is None or self.y_train is None:
                raise ValueError("Training data not available. Run feature engineering first.")
            
            # Random Forest
            rfc = RandomForestClassifier(random_state=const.RANDOM_STATE)
            cv_rfc = GridSearchCV(estimator=rfc, param_grid=const.RF_PARAM_GRID, cv=5)
            cv_rfc.fit(self.X_train, self.y_train)
            self.rfc_model = cv_rfc.best_estimator_
            
            # Logistic Regression
            lrc = LogisticRegression(
                solver=const.LR_SOLVER,
                max_iter=const.LR_MAX_ITER,
                random_state=const.RANDOM_STATE
            )
            lrc.fit(self.X_train, self.y_train)
            self.lr_model = lrc
            
            # Save models
            joblib.dump(self.rfc_model, os.path.join(const.MODEL_PATH, const.RFC_MODEL_NAME))
            joblib.dump(self.lr_model, os.path.join(const.MODEL_PATH, const.LOGISTIC_MODEL_NAME))
            
            logging.info("Model training completed successfully")
            return True
        except Exception as err:
            logging.error("Model training failed: %s", str(err))
            return False

    # ... rest of the class remains the same ...

if __name__ == "__main__":
    predictor = ChurnPredictor()
    if predictor.import_data():
        predictor.perform_eda()
        predictor.encoder_helper()
        predictor.perform_feature_engineering()
        predictor.train_models()
        predictor.create_model_reports()
    else:
        logging.error("Pipeline failed at data import stage")