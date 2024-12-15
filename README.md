# customer-churn-p1
Udacity ML DevOps Project 1

# Customer Churn Prediction Project

## Project Overview
This project implements a machine learning solution to predict customer churn for a bank. It uses a dataset of credit card customers to identify those most likely to churn, helping the bank take proactive retention measures.

The project follows software engineering best practices including:
- Modular, object-oriented design
- PEP 8 coding standards
- Comprehensive testing
- Proper logging
- Clear documentation

## Project Structure
```
.
├── README.md                     # Project documentation
├── requirements.txt              # Project dependencies
├── constants.py                  # Project constants and configuration
├── churn_library.py             # Main churn prediction implementation
├── churn_script_logging_and_tests.py  # Testing suite
├── data/
│   └── bank_data.csv            # Input dataset
├── images/
│   ├── eda/                     # Exploratory Data Analysis plots
│   └── results/                 # Model performance plots
├── logs/
│   └── churn_library.log        # Logging output
└── models/                      # Saved model files
```

## Dependencies
The project requires Python 3.8+ and the following key libraries:
```
scikit-learn==0.24.1
pandas==1.2.4
numpy==1.20.1
matplotlib==3.3.4
seaborn==0.11.2
pytest==7.1.2
pylint==2.7.4
autopep8==1.5.6
```

To install all dependencies:
```bash
pip install -r requirements.txt
```

## Installation & Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd customer-churn-prediction
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure the data file is in place:
```bash
mkdir -p data
# Place bank_data.csv in the data directory
```

## Usage

### Running the Churn Prediction Pipeline

The main script performs the complete churn prediction pipeline:
```bash
python churn_library.py
```

This will:
1. Load and prepare the data
2. Perform EDA and generate visualizations
3. Engineer features
4. Train Random Forest and Logistic Regression models
5. Generate performance reports and plots
6. Save trained models

### Running Tests

To run the test suite with logging:
```bash
pytest churn_script_logging_and_tests.py -v
```

To check code quality:
```bash
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```

To format code according to PEP 8:
```bash
autopep8 --in-place --aggressive --aggressive churn_library.py
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
```

## Project Features

### Data Processing
- Automated data import and validation
- Creation of churn labels from attrition flags
- Encoding of categorical variables
- Feature engineering pipeline

### EDA
The project generates various exploratory plots including:
- Churn distribution
- Customer age distribution
- Marital status analysis
- Transaction patterns
- Feature correlation heatmap

### Model Training
Implements two models:
1. Random Forest Classifier with GridSearchCV
2. Logistic Regression

### Model Evaluation
Generates comprehensive performance metrics:
- Classification reports
- ROC curves
- Feature importance plots
- Model comparison visualizations

## Output Files

### EDA Visualizations
- `images/eda/churn_histogram.png`
- `images/eda/customer_age_histogram.png`
- `images/eda/marital_status_counts.png`
- `images/eda/total_transaction_histogram.png`
- `images/eda/heatmap.png`

### Results
- `images/results/rf_classification_report.png`
- `images/results/lr_classification_report.png`
- `images/results/feature_importance.png`
- `images/results/roc_curves.png`

### Models
- `models/rfc_model.pkl`
- `models/logistic_model.pkl`

## Logging
The project maintains detailed logs in `logs/churn_library.log`, including:
- Data import status
- EDA completion
- Model training progress
- Error messages
- Test results

## Contributing
To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure they pass
5. Submit a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
Created by Ryan Lence
Date: December 15, 2024