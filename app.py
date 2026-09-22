import mysql.connector
import streamlit as st
import pandas as pd

def get_mysql_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="shikharkhare03@#$",
        database="customer_churn"
    )

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to leave the service.")


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove missing values
    df = df.dropna()

    # Remove customer ID
    df = df.drop("customerID", axis=1)

    return df


df = load_data()


# ==========================================
# PREPARE DATA
# ==========================================

X = df.drop("Churn", axis=1)
y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_columns
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ]
)


model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# ==========================================
# TRAIN MODEL
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Customer Information")


gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=840.0
)


# ==========================================
# PREDICTION
# ==========================================

st.subheader("🔮 Churn Prediction")

st.write(
    "Enter the customer information in the sidebar "
    "and click the button below."
)


if st.button(
    "🚀 Predict Customer Churn",
    use_container_width=True
):

    customer = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [
            1 if senior_citizen == "Yes" else 0
        ],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })


    prediction = model.predict(customer)[0]

    probability = model.predict_proba(
        customer
    )[0][1] * 100


    # ======================================
    # RESULT
    # ======================================

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

         # ======================================
        # PROFESSIONAL PREDICTION RESULT
        # ======================================
        
        st.divider()
        st.subheader("🎯 Prediction Result")
        
        result_col, probability_col = st.columns(2)
        
        with result_col:
        
            if prediction == 1:
                st.error("⚠️ HIGH CHURN RISK")
                st.write(
                    "This customer is predicted to be likely to leave the service."
                )
            else:
                st.success("✅ LOW CHURN RISK")
                st.write(
                    "This customer is predicted to be likely to stay with the service."
                )
        
        with probability_col:
        
            st.metric(
                "Churn Probability",
                f"{probability:.2f}%"
            )
        
            st.progress(
                min(int(probability), 100)
            )
        
        
        # Risk classification
        if probability < 30:
            risk_level = "🟢 Low Risk"
        elif probability < 60:
            risk_level = "🟡 Medium Risk"
        else:
            risk_level = "🔴 High Risk"
        
        st.info(
            f"Customer Risk Level: **{risk_level}**"
        )
        
        
        # Recommendation
        st.subheader("💡 Recommended Action")
        
        if probability >= 60:
        
            st.write(
                "Consider offering a retention incentive, "
                "personalized support, or a suitable contract plan."
            )
        
        elif probability >= 30:
        
            st.write(
                "Monitor this customer and consider proactive "
                "engagement to reduce the possibility of churn."
            )
        
        else:
        
            st.write(
                "The customer currently has a relatively low "
                "predicted churn risk. Continue regular engagement."
            )
        

       
    with col2:

        st.metric(
            "Churn Probability",
            f"{probability:.2f}%"
        )


    st.progress(
        int(probability)
    )


    if probability >= 50:

        st.warning(
            "This customer has a relatively high "
            "probability of leaving the service."
        )

    else:

        st.info(
            "This customer has a relatively low "
            "probability of leaving the service."
        )


# ==========================================
# DATASET INFORMATION
# ==========================================

st.divider()

st.subheader("📊 Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Customers",
        len(df)
    )

with col2:
    st.metric(
        "Features",
        len(X.columns)
    )

with col3:
    st.metric(
        "Churned Customers",
        int(y.sum())
    )

def get_mysql_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="shikharkhare03@#$",
        database="customer_churn"
    )
try:
    connection = get_mysql_connection()

    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM customers")

    customer_count = cursor.fetchone()[0]

    st.success("✅ MySQL Database Connected")
    st.metric("Total Customers in Database", customer_count)

    cursor.close()
    connection.close()

except mysql.connector.Error as e:
    st.error(f"❌ Database connection failed: {e}")

# ==============================
# MYSQL DATABASE DASHBOARD
# ==============================

st.divider()

st.header("📊 MySQL Database Dashboard")

try:
    connection = get_mysql_connection()

    # Load customer data from MySQL
    query = "SELECT * FROM customers"
    db_df = pd.read_sql(query, connection)

    connection.close()

    # Basic statistics
    total_customers = len(db_df)

    churned_customers = len(
        db_df[db_df["Churn"] == "Yes"]
    )

    active_customers = total_customers - churned_customers

    churn_rate = (churned_customers / total_customers) * 100

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "👥 Total Customers",
        total_customers
    )

    col2.metric(
        "🔴 Churned Customers",
        churned_customers
    )

    col3.metric(
        "🟢 Active Customers",
        active_customers
    )

    col4.metric(
        "📈 Churn Rate",
        f"{churn_rate:.2f}%"
    )

    # Churn Rate by Contract
    st.subheader("📄 Churn Rate by Contract")
    
    contract_analysis = (
        db_df.groupby("Contract")
        .agg(
            total_customers=("Churn", "count"),
            churned_customers=("Churn", lambda x: (x == "Yes").sum())
        )
        .reset_index()
    )
    
    contract_analysis["churn_rate"] = (
        contract_analysis["churned_customers"]
        / contract_analysis["total_customers"]
        * 100
    )
    
    contract_analysis["churn_rate"] = contract_analysis["churn_rate"].round(2)
    
    st.dataframe(
        contract_analysis,
        use_container_width=True,
        hide_index=True
    )
    
    st.bar_chart(
        contract_analysis.set_index("Contract")["churn_rate"]
    )
    # Churn Rate by Internet Service
    st.subheader("🌐 Churn Rate by Internet Service")
    
    internet_analysis = (
        db_df.groupby("InternetService")
        .agg(
            total_customers=("Churn", "count"),
            churned_customers=("Churn", lambda x: (x == "Yes").sum())
        )
        .reset_index()
    )
    
    internet_analysis["churn_rate"] = (
        internet_analysis["churned_customers"]
        / internet_analysis["total_customers"]
        * 100
    )
    
    internet_analysis["churn_rate"] = internet_analysis["churn_rate"].round(2)
    
    st.dataframe(
        internet_analysis,
        use_container_width=True,
        hide_index=True
    )
    
    st.bar_chart(
        internet_analysis.set_index("InternetService")["churn_rate"]
    )
     # Churn Rate by Payment Method
    st.subheader("💳 Churn Rate by Payment Method")
    
    payment_analysis = (
        db_df.groupby("PaymentMethod")
        .agg(
            total_customers=("Churn", "count"),
            churned_customers=("Churn", lambda x: (x == "Yes").sum())
        )
        .reset_index()
    )
    
    payment_analysis["churn_rate"] = (
        payment_analysis["churned_customers"]
        / payment_analysis["total_customers"]
        * 100
    )
    
    payment_analysis["churn_rate"] = payment_analysis["churn_rate"].round(2)
    
    st.dataframe(
        payment_analysis,
        use_container_width=True,
        hide_index=True
    )
    
    st.bar_chart(
        payment_analysis.set_index("PaymentMethod")["churn_rate"]
    )

    # ==============================
    # CUSTOMER SEARCH & FILTER
    # ==============================
    
    st.subheader("🔍 Customer Database")
    
    search_text = st.text_input(
        "Search Customer ID",
        placeholder="Enter Customer ID..."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        churn_filter = st.selectbox(
            "Churn Status",
            ["All", "Yes", "No"]
        )
    
    with col2:
        contract_filter = st.selectbox(
            "Contract",
            ["All"] + sorted(db_df["Contract"].dropna().unique().tolist())
        )
    
    filtered_df = db_df.copy()
    
    # Customer ID search
    if search_text:
        filtered_df = filtered_df[
            filtered_df["customerID"]
            .astype(str)
            .str.contains(search_text, case=False, na=False)
        ]
    
    # Churn filter
    if churn_filter != "All":
        filtered_df = filtered_df[
            filtered_df["Churn"] == churn_filter
        ]
    
    # Contract filter
    if contract_filter != "All":
        filtered_df = filtered_df[
            filtered_df["Contract"] == contract_filter
        ]
    
    st.write(
        f"Showing **{len(filtered_df)}** customer(s)"
    )
    
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )
    
       

except Exception as e:
    st.error(f"Database Error: {e}")
