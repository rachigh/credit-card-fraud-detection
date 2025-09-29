#  Credit Card Fraud Detection

An end-to-end machine learning project for credit card fraud detection using classical ML algorithms and deep learning techniques. This project includes comprehensive data analysis, model training, and deployment as a REST API.

##  Project Overview

This project implements a complete fraud detection system with:
- **Exploratory Data Analysis (EDA)** with detailed insights
- **Advanced preprocessing** with feature engineering and SMOTE balancing
- **Classical ML models** comparison (10+ algorithms)
- **Deep Learning** implementation with TensorFlow/Keras


##  Key Results

| Model | F1-Score | ROC-AUC | Precision | Recall |
|-------|----------|---------|-----------|--------|
| XGBoost (Tuned) | 0.87 | 0.94 | 0.88 | 0.86 |
| Neural Network | 0.85 | 0.92 | 0.87 | 0.83 |
| Random Forest | 0.84 | 0.91 | 0.86 | 0.82 |
| Ensemble Model | 0.88 | 0.95 | 0.89 | 0.87 |

##  Quick Start

### Option 1: Google Colab (Recommended)
1. Open any notebook in Google Colab
2. Upload your `creditcard.csv` dataset
3. Run all cells sequentially


### Option 2: Local Setup
```bash
# Clone the repository
git clone https://github.com/rachigh/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create virtual environment
python -m venv fraud_env
source fraud_env/bin/activate  # On Windows: fraud_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# Place creditcard.csv in data/raw/

# Run notebooks in order
jupyter notebook
```

##  Project Structure

```
credit-card-fraud-detection/
├── notebooks/
│   ├── 01_EDA_Fraud_Detection.ipynb           # Exploratory Data Analysis
│   ├── 02_Preprocessing_Fraud_Detection.ipynb # Data preprocessing & feature engineering
│   ├── 03_Modeling_Fraud_Detection.ipynb      # Classical ML models comparison
│   └── 04_Deep_Learning_Fraud_Detection.ipynb # Neural networks with TensorFlow
├── models/
│   ├── best_fraud_model.joblib   # Best classical ML model
│   ├── deployment_artifacts.pkl  # Complete preprocessing pipeline
│   └── best_fraud_detection_nn.h5 # Best neural network model
├── data/
│   ├── raw/                      # Original dataset 
│   └── processed/                # Processed datasets
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

##  Methodology

### 1. Exploratory Data Analysis
- **Dataset**: 284,807 transactions (0.17% fraud rate)
- **Features**: 28 PCA-transformed features + Time + Amount
- **Key insights**: Time-based fraud patterns, amount distribution analysis
- **Class imbalance**: 1:577 ratio (Normal:Fraud)

### 2. Data Preprocessing
- **Duplicate removal**: 260 duplicates cleaned
- **Feature engineering**: Time-based features, amount transformations
- **Scaling**: StandardScaler for Amount/Time features
- **Balancing**: SMOTE with multiple strategies tested
- **Feature selection**: Top 10 features based on correlation analysis

### 3. Model Development
#### Classical Machine Learning
- **Models tested**: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, etc.
- **Hyperparameter tuning**: GridSearchCV with stratified cross-validation
- **Evaluation metrics**: F1-score, ROC-AUC, Precision, Recall (fraud-focused)

#### Deep Learning
- **Architecture**: 4 different neural network designs tested
- **Features**: Dropout, BatchNormalization, Early Stopping
- **Optimization**: Adam, RMSprop optimizers with learning rate scheduling
- **Class weights**: Handled imbalanced data with weighted loss

### 4. Model Evaluation
- **Business metrics**: Cost-benefit analysis with real-world impact
- **Confusion matrix**: Focus on minimizing false negatives
- **Threshold optimization**: Business-oriented threshold tuning


##  Technologies Used

### Machine Learning & Data Science
- **Python 3.8+**: Primary programming language
- **pandas & NumPy**: Data manipulation and analysis
- **scikit-learn**: Classical ML algorithms and preprocessing
- **TensorFlow/Keras**: Deep learning implementation
- **imbalanced-learn**: SMOTE and resampling techniques

### Visualization & Analysis
- **matplotlib & seaborn**: Data visualization
- **plotly**: Interactive charts

##  Dataset

The project uses the **Credit Card Fraud Detection** dataset from Kaggle:
- **Source**: [Kaggle Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**: 30 (V1-V28 PCA features + Time + Amount)
- **Target**: Binary classification (0=Normal, 1=Fraud)
- **Imbalance**: 0.17% fraud rate

##  Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or Google Colab
- Git (for cloning)

### Step-by-step Installation

1. **Clone the repository**
```bash
git clone https://github.com/rachigh/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. **Create virtual environment**
```bash
python -m venv fraud_env
source fraud_env/bin/activate  # Linux/Mac
# fraud_env\Scripts\activate    # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
   - Register on [Kaggle](https://www.kaggle.com/)
   - Download [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - Place `creditcard.csv` in `data/raw/`

5. **Run notebooks**
```bash
jupyter notebook
# Open and run notebooks in order: 01 → 02 → 03 → 04 
```


##  Model Performance Details

### Cross-Validation Results
```
Stratified 5-Fold Cross-Validation:
├── XGBoost:        F1: 0.87 ± 0.02
├── Random Forest:  F1: 0.84 ± 0.03
├── Neural Network: F1: 0.85 ± 0.02
└── Ensemble:       F1: 0.88 ± 0.01
```

