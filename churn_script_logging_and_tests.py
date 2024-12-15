"""
churn_script_logging_and_tests.py

This module contains test cases for the ChurnPredictor class.
It uses pytest for testing and includes comprehensive logging.

Date: December 15, 2024
"""

import os
import logging
import pytest
import pandas as pd
import joblib
from churn_library import ChurnPredictor
import constants as const

# Configure logging
logging.basicConfig(
    filename=const.LOG_PATH,
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s %(levelname)s: %(message)s'
)

@pytest.fixture(scope="module")
def churn_predictor():
    """
    Fixture that provides a ChurnPredictor instance for all tests.
    """
    return ChurnPredictor()

def test_import_data(churn_predictor):
    """
    Test data import functionality.
    
    The import should:
    - Successfully load the CSV file
    - Create a DataFrame with the expected shape
    - Create the Churn column correctly
    """
    try:
        # Test successful import
        assert churn_predictor.import_data() is True
        assert churn_predictor.df is not None
        assert churn_predictor.df.shape[0] > 0
        assert churn_predictor.df.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
        
        # Test that Churn column was created correctly
        assert 'Churn' in churn_predictor.df.columns
        assert set(churn_predictor.df['Churn'].unique()) == {0, 1}
        logging.info("Testing Churn column creation: SUCCESS")
        
        # Test import with invalid path
        assert churn_predictor.import_data('invalid/path.csv') is False
        logging.info("Testing invalid path handling: SUCCESS")
        
    except AssertionError as err:
        logging.error("Testing import_data: Error occurred while testing data import")
        raise err

def test_perform_eda(churn_predictor):
    """
    Test EDA functionality.
    """
    try:
        # Ensure data is loaded
        if churn_predictor.df is None:
            churn_predictor.import_data()
            
        # Test EDA execution
        assert churn_predictor.perform_eda() is True
        
        # Check if all plot files were created
        eda_plots = [
            const.CHURN_HISTOGRAM,
            const.CUSTOMER_AGE_HISTOGRAM,
            const.MARITAL_STATUS_BAR,
            const.TOTAL_TRANSACTION_HISTOGRAM,
            const.CORRELATION_HEATMAP
        ]
        
        for plot in eda_plots:
            plot_path = os.path.join(const.EDA_IMAGES_PATH, plot)
            assert os.path.exists(plot_path)
            assert os.path.getsize(plot_path) > 0
        
        logging.info("Testing perform_eda: SUCCESS")
        
    except AssertionError as err:
        logging.error("Testing perform_eda: Failed to create EDA plots")
        raise err

def test_encoder_helper(churn_predictor):
    """
    Test categorical encoding functionality.
    """
    try:
        # Ensure data is loaded
        if churn_predictor.df is None:
            churn_predictor.import_data()
            
        initial_columns = churn_predictor.df.shape[1]
        
        # Test encoding execution
        assert churn_predictor.encoder_helper() is True
        
        # Check if new columns were created
        for category in const.CATEGORICAL_COLUMNS:
            new_column = f'{category}_Churn'
            assert new_column in churn_predictor.df.columns
            assert churn_predictor.df[new_column].dtype in ['float64', 'float32']
            assert not churn_predictor.df[new_column].isna().any()
            
        logging.info("Testing encoder_helper: SUCCESS")
        
    except AssertionError as err:
        logging.error("Testing encoder_helper: Failed to encode categorical variables")
        raise err

def test_perform_feature_engineering(churn_predictor):
    """
    Test feature engineering functionality.
    """
    try:
        # Ensure data is prepared
        if churn_predictor.df is None:
            churn_predictor.import_data()
            churn_predictor.encoder_helper()
            
        # Test feature engineering execution
        assert churn_predictor.perform_feature_engineering() is True
        
        # Check if splits were created
        assert churn_predictor.X_train is not None
        assert churn_predictor.X_test is not None
        assert churn_predictor.y_train is not None
        assert churn_predictor.y_test is not None
        
        # Verify split sizes
        total_rows = len(churn_predictor.df)
        expected_test_size = int(total_rows * const.TEST_SIZE)
        expected_train_size = total_rows - expected_test_size
        
        assert len(churn_predictor.X_train) == expected_train_size
        assert len(churn_predictor.X_test) == expected_test_size
        
        logging.info("Testing perform_feature_engineering: SUCCESS")
        
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: Failed to engineer features")
        raise err

def test_train_models(churn_predictor):
    """
    Test model training functionality.
    """
    try:
        # Ensure data is prepared
        if churn_predictor.X_train is None:
            churn_predictor.import_data()
            churn_predictor.encoder_helper()
            churn_predictor.perform_feature_engineering()
            
        # Test model training execution
        assert churn_predictor.train_models() is True
        
        # Check if model files were created
        model_files = [
            os.path.join(const.MODEL_PATH, const.RFC_MODEL_NAME),
            os.path.join(const.MODEL_PATH, const.LOGISTIC_MODEL_NAME)
        ]
        
        for model_path in model_files:
            assert os.path.exists(model_path)
            assert os.path.getsize(model_path) > 0
            
        logging.info("Testing train_models: SUCCESS")
        
    except AssertionError as err:
        logging.error("Testing train_models: Failed to train or save models")
        raise err

if __name__ == "__main__":
    pytest.main([__file__])
