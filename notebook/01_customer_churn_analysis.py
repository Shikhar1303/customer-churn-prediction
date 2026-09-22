import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())
print("\nTotalCharges Data Type Before:")
print(df["TotalCharges"].dtype)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nTotalCharges Data Type After:")
print(df["TotalCharges"].dtype)

print("\nMissing TotalCharges After Conversion:")
print(df["TotalCharges"].isnull().sum())
# Fill missing TotalCharges values with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("\nMissing TotalCharges After Filling:")
print(df["TotalCharges"].isnull().sum())
# Remove customer ID because it is not useful for prediction
df = df.drop("customerID", axis=1)

print("\nDataset after removing customerID:")
print(df.head())

print("\nDataset shape:")
print(df.shape)
# Convert Churn into 0 and 1
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

print("\nChurn values after conversion:")
print(df["Churn"].value_counts())
# Convert categorical columns into numerical columns
df = pd.get_dummies(df, drop_first=True)

print("\nDataset after encoding:")
print(df.head())

print("\nFinal dataset shape:")
print(df.shape)

# ================================
# MACHINE LEARNING
# ================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL RESULTS")
print("==============================")

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ================================
# RANDOM FOREST MODEL
# ================================

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Evaluate Random Forest
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\n==============================")
print("RANDOM FOREST RESULTS")
print("==============================")

print("Random Forest Accuracy:", rf_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
# ================================
# CHURN ANALYSIS
# ================================

import matplotlib.pyplot as plt
import seaborn as sns

# Churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.show()
# Churn by contract type
original_df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

plt.figure(figsize=(8, 5))
sns.countplot(x="Contract", hue="Churn", data=original_df)
plt.title("Churn by Contract Type")
plt.xticks(rotation=15)
plt.show()


# Churn by payment method
plt.figure(figsize=(10, 5))
sns.countplot(x="PaymentMethod", hue="Churn", data=original_df)
plt.title("Churn by Payment Method")
plt.xticks(rotation=30)
plt.show()


# Monthly charges vs churn
plt.figure(figsize=(7, 5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=original_df)
plt.title("Monthly Charges vs Churn")
plt.show()
# ================================
# FEATURE IMPORTANCE
# ================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": abs(model.coef_[0])
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Factors Influencing Churn:")
print(feature_importance.head(15))
# ================================
# ROC-AUC EVALUATION
# ================================

from sklearn.metrics import roc_auc_score

# Probability of churn (class 1)
y_prob = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)

print("\n==============================")
print("ROC-AUC SCORE")
print("==============================")
print("ROC-AUC:", roc_auc)
# ================================
# SAMPLE CUSTOMER PREDICTION
# ================================

# Take one customer from the test dataset
sample_customer = X_test.iloc[[0]]

# Predict churn
prediction = model.predict(sample_customer)[0]

# Get probability of churn
probability = model.predict_proba(sample_customer)[0][1]

print("\n==============================")
print("CUSTOMER CHURN PREDICTION")
print("==============================")

if prediction == 1:
    print("Prediction: Customer is likely to CHURN")
else:
    print("Prediction: Customer is likely to STAY")

print("Churn Probability:", round(probability * 100, 2), "%")