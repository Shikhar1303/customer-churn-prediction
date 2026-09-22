# Customer Churn Prediction

## 📌 Project Overview

Customer churn refers to customers who stop using a company's services.

This project analyzes customer data and uses Machine Learning and SQL to understand customer churn and predict whether a customer is likely to leave the service.

The project uses Python, Pandas, Matplotlib, Seaborn, Scikit-learn, and MySQL.

---

## 🎯 Objectives

- Analyze customer churn patterns.
- Clean and preprocess customer data.
- Perform Exploratory Data Analysis (EDA).
- Identify important factors affecting customer churn.
- Build Machine Learning models for churn prediction.
- Compare Logistic Regression and Random Forest.
- Perform customer churn analysis using SQL.
- Predict the probability of customer churn.

---

## 📊 Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains information about customers such as:

- Customer ID
- Gender
- Senior Citizen
- Tenure
- Internet Service
- Contract
- Payment Method
- Monthly Charges
- Total Charges
- Churn

The dataset contains 7043 customer records.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- MySQL
- VS Code

---

## 🔄 Project Workflow

1. Load the customer dataset.
2. Inspect the dataset.
3. Check missing values.
4. Convert data types.
5. Handle missing values.
6. Perform Exploratory Data Analysis.
7. Prepare data for Machine Learning.
8. Train Logistic Regression model.
9. Train Random Forest model.
10. Evaluate the models.
11. Analyze feature importance.
12. Perform SQL-based customer analysis.
13. Predict churn probability for a customer.

---

## 🤖 Machine Learning Models

### Logistic Regression

Logistic Regression is used as the primary classification model to predict whether a customer will churn.

### Random Forest

Random Forest is used as a second classification model and its performance is compared with Logistic Regression.

---

## 📈 Model Performance

The Logistic Regression model achieved approximately **80% accuracy**.

The Random Forest model achieved approximately **78% accuracy**.

The models were evaluated using:

- Accuracy
- Classification Report
- Confusion Matrix

---

## 🔍 Feature Importance

Feature importance analysis was performed to identify the factors that have the greatest influence on customer churn.

The project also visualizes the Top 15 factors influencing churn.

---

## 🗄️ SQL Analysis

MySQL was used to perform additional customer churn analysis.

The analysis includes:

- Churn by contract type
- Churn by payment method
- Churn by internet service
- Churn among senior citizens
- Average monthly charges of churned customers
- Customer-level churn statistics

### Example Finding

Month-to-month customers have a significantly higher churn rate than customers with one-year or two-year contracts.

---

## 📁 Project Structure

```text
Customer_Churn-Project/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebook/
│   └── 01_customer_churn_analysis.py
│
├── sql/
│   └── churn_analysis.sql
│
├── README.md
│
└── hello.py
