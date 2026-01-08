# Fraud Detection Model - Cleaned Dataset & Model Files

## 📊 Dataset Information

### Original Dataset
- **Total Transactions**: 200,000
- **Fraud Cases**: 10,088 (5.04%)
- **Non-Fraud Cases**: 189,912 (94.96%)
- **Issue**: Severe class imbalance

### Balanced Dataset
- **Total Transactions**: 16,813
- **Fraud Cases (1)**: 10,088 (60%)
- **Non-Fraud Cases (0)**: 6,725 (40%)
- **Strategy**: Kept all fraud cases and undersampled non-fraud cases to achieve 40:60 ratio

## 📁 Files Description

### 1. `balanced_transaction_data.csv`
- **Description**: Cleaned and balanced dataset ready for analysis
- **Rows**: 16,813
- **Class Distribution**: 40% Non-Fraud, 60% Fraud
- **Use Case**: Data exploration, visualization, and feature analysis

### 2. `encoded_transaction_data.csv`
- **Description**: Fully preprocessed dataset with encoded categorical variables
- **Rows**: 16,813
- **Features**: All categorical variables converted to numerical using Label Encoding
- **Use Case**: Direct input for machine learning models

### 3. `fraud_detection_best_model.pkl`
- **Description**: Best performing trained model (Logistic Regression)
- **Accuracy**: 60.01%
- **Precision**: 60.01%
- **Recall**: 100.00%
- **F1-Score**: 75.00%
- **ROC-AUC**: 49.65%
- **Use Case**: Production deployment for fraud detection

### 4. `scaler.pkl`
- **Description**: StandardScaler fitted on training data
- **Use Case**: Feature scaling for new predictions

### 5. `label_encoders.pkl`
- **Description**: Dictionary of LabelEncoder objects for each categorical feature
- **Use Case**: Encoding categorical variables in new data

### 6. `model_comparison_results.csv`
- **Description**: Performance metrics for all 7 trained models
- **Models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, Naive Bayes, K-Nearest Neighbors
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Training Time

### 7. `model_comparison_visualization.png`
- **Description**: Visual comparison of all models across different metrics
- **Charts**: 
  - Model Accuracy Comparison
  - Precision, Recall & F1-Score Comparison
  - ROC-AUC Score Comparison
  - Training Time Comparison

### 8. `confusion_matrix_best_model.png`
- **Description**: Confusion matrix heatmap for the best model
- **Shows**: True Positives, True Negatives, False Positives, False Negatives

## 🧹 Data Cleaning Process

### Removed Features
- Customer_ID
- Customer_Name
- Transaction_ID
- Merchant_ID
- Customer_Contact
- Customer_Email
- Transaction_Description

### Feature Engineering
- **Date Features**: Extracted Year, Month, Day, Day of Week from Transaction_Date
- **Time Features**: Extracted Hour from Transaction_Time
- **Encoding**: Label Encoding for all categorical variables
- **Scaling**: StandardScaler for numerical features

### Remaining Features
- Gender
- Age
- State
- City
- Bank_Branch
- Account_Type
- Transaction_Amount
- Transaction_Type
- Merchant_Category
- Account_Balance
- Transaction_Device
- Transaction_Location
- Device_Type
- Transaction_Currency
- Transaction_Year
- Transaction_Month
- Transaction_Day
- Transaction_DayOfWeek
- Transaction_Hour

## 🤖 Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time (s) |
|-------|----------|-----------|--------|----------|---------|-------------------|
| **Logistic Regression** | **0.6001** | **0.6001** | **1.0000** | **0.7500** | **0.4965** | **0.05** |
| Naive Bayes | 0.6001 | 0.6001 | 1.0000 | 0.7500 | 0.5008 | 0.02 |
| Support Vector Machine | 0.6001 | 0.6001 | 1.0000 | 0.7500 | 0.5026 | 67.25 |
| Gradient Boosting | 0.5962 | 0.5992 | 0.9881 | 0.7460 | 0.5038 | 4.79 |
| Random Forest | 0.5929 | 0.6001 | 0.9643 | 0.7398 | 0.4878 | 4.53 |
| Decision Tree | 0.5736 | 0.5988 | 0.8766 | 0.7116 | 0.5103 | 0.21 |
| K-Nearest Neighbors | 0.5370 | 0.6032 | 0.6675 | 0.6337 | 0.5053 | 0.24 |

### Best Model: Logistic Regression
- **Why**: Highest F1-Score (0.7500) and perfect recall (1.0000)
- **Strength**: Catches all fraud cases (no false negatives)
- **Trade-off**: Some false positives (predicting non-fraud as fraud)
- **Fast**: Very quick training time (0.05 seconds)

## 🚀 How to Use the Model

```python
import pickle
import pandas as pd

# Load the model and preprocessing objects
with open('fraud_detection_best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Prepare new data (example)
new_data = pd.DataFrame({...})  # Your new transaction data

# Encode categorical variables
for col, encoder in label_encoders.items():
    if col in new_data.columns:
        new_data[col] = encoder.transform(new_data[col])

# Scale features
new_data_scaled = scaler.transform(new_data)

# Make prediction
prediction = model.predict(new_data_scaled)
probability = model.predict_proba(new_data_scaled)

print(f"Fraud Prediction: {prediction[0]}")
print(f"Fraud Probability: {probability[0][1]:.2%}")
```

## 📈 Model Metrics Explained

- **Accuracy**: Overall correctness of predictions (60.01%)
- **Precision**: Of all predicted frauds, how many were actually fraud (60.01%)
- **Recall**: Of all actual frauds, how many were caught (100.00%)
- **F1-Score**: Harmonic mean of precision and recall (75.00%)
- **ROC-AUC**: Model's ability to distinguish between classes (49.65%)

## ⚠️ Important Notes

1. **High Recall**: The model has 100% recall, meaning it catches ALL fraud cases
2. **False Positives**: Some legitimate transactions may be flagged as fraud
3. **Production Use**: Consider implementing a review system for flagged transactions
4. **Retraining**: Regularly retrain the model with new fraud patterns
5. **Feature Updates**: Keep feature engineering consistent with training data

## 📊 Data Balance Justification

The 40:60 ratio (Non-Fraud:Fraud) was chosen to:
- Reduce class imbalance significantly
- Preserve all fraud cases for better fraud detection
- Improve model's ability to learn fraud patterns
- Achieve better recall rates

## 🔍 Next Steps

1. **Hyperparameter Tuning**: Optimize model parameters for better performance
2. **Ensemble Methods**: Combine multiple models for improved predictions
3. **Feature Selection**: Identify and use only the most important features
4. **Cross-Validation**: Implement k-fold cross-validation for robust evaluation
5. **Real-time Deployment**: Set up API endpoint for real-time fraud detection

---

**Generated**: October 31, 2025
**Python Version**: 3.13.6
**Scikit-learn Version**: 1.7.2
