# Credit Card Customer Churn Prediction

## Project Overview
This project implements a machine learning solution to predict customer churn for credit card services. It uses supervised learning to identify customers most likely to churn, helping financial institutions take proactive retention measures.

## Project Features
- Data preprocessing and exploratory data analysis (EDA)
- Feature engineering with categorical variable encoding
- Model training with Random Forest and Logistic Regression
- Model evaluation with classification reports and ROC curves
- Comprehensive logging and testing suite
- Modular, object-oriented design following PEP 8 standards

## Directory Structure
```
.
├── README.md                              # Project documentation
├── requirements.txt                       # Project dependencies
├── constants.py                          # Configuration constants
├── churn_library.py                      # Main implementation
├── churn_script_logging_and_tests.py     # Testing suite
├── data/
│   └── bank_data.csv                    # Input dataset
├── images/
│   ├── eda/                            # EDA visualizations
│   │   ├── churn_histogram.png
│   │   ├── customer_age_histogram.png
│   │   ├── marital_status_counts.png
│   │   ├── total_transaction_histogram.png
│   │   └── heatmap.png
│   └── results/                        # Model results
│       ├── feature_importance.png
│       ├── lrc_roc_curve.png
│       ├── roc_curves.png
│       ├── lr_classification_report.png
│       └── rf_classification_report.png
├── logs/
│   └── churn_library.log              # Execution logs
└── models/                            # Saved model files
    ├── rfc_model.pkl
    └── logistic_model.pkl
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Steps
1. Clone the repository:
```bash
git clone <repository-url>
cd credit-card-churn
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Prepare the data:
- Place your `bank_data.csv` file in the `data/` directory
- Ensure the data file follows the expected schema with required columns

## Usage

### Running the Churn Prediction Pipeline
Execute the main script to run the complete pipeline:
```bash
python churn_library.py
```

This will:
1. Load and preprocess the data
2. Perform EDA and generate visualizations
3. Engineer features
4. Train Random Forest and Logistic Regression models
5. Generate and save performance reports
6. Save trained models for future use

### Running Tests
Execute the test suite:
```bash
pytest churn_script_logging_and_tests.py -v
```

Check code quality:
```bash
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```

Format code:
```bash
autopep8 --in-place --aggressive --aggressive churn_library.py
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
```

## Project Components

### Data Processing (`churn_library.py`)
The `ChurnPredictor` class implements:
- Data import and validation
- EDA with visualization generation
- Categorical variable encoding
- Feature engineering pipeline
- Model training and evaluation
- Results visualization and storage

### Configuration (`constants.py`)
Contains:
- File paths and names
- Model parameters
- Column definitions
- Visualization settings

### Testing (`churn_script_logging_and_tests.py`)
Includes:
- Unit tests for all major functions
- Logging configuration
- Data validation checks
- Performance metric verification

## Model Details

### Features Used
- Customer demographics (age, gender, etc.)
- Account information (credit limit, balance, etc.)
- Transaction history
- Relationship metrics

### Models Implemented
1. Random Forest Classifier
   - Optimized with GridSearchCV
   - Feature importance analysis
   
2. Logistic Regression
   - Baseline model for comparison
   - ROC curve analysis

## Output Files

### EDA Visualizations
- Customer age distribution
- Churn distribution
- Marital status analysis
- Transaction patterns
- Feature correlation heatmap

### Model Results
- Classification reports
- Feature importance rankings
- ROC curves
- Model comparison metrics

## Logging
- Comprehensive logging in `logs/churn_library.log`
- Tracks execution progress
- Records errors and warnings
- Documents model performance

## Contributing
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## Code Quality Standards
- PEP 8 compliance
- Comprehensive docstrings
- Unit test coverage
- Error handling
- Logging implementation

## License
This project is licensed under the MIT License - see LICENSE file for details.

## Author
Created by: [Your Name]
Date: December 15, 2024

## Acknowledgments
- Scikit-learn documentation and community
- Python data science community
- Original data providers

## Support
For issues and questions:
1. Check existing issues on GitHub
2. Create a new issue with:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
