-- ==========================================
-- CUSTOMER CHURN ANALYSIS
-- ==========================================

-- 1. View all customers
SELECT *
FROM customers;


-- 2. Total number of customers
SELECT COUNT(*) AS total_customers
FROM customers;


-- 3. Number of churned customers
SELECT COUNT(*) AS churned_customers
FROM customers
WHERE Churn = 'Yes';


-- 4. Churn by contract type
SELECT
    Contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers
FROM customers
GROUP BY Contract
ORDER BY churned_customers DESC;


-- 5. Churn by payment method
SELECT
    PaymentMethod,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers
FROM customers
GROUP BY PaymentMethod
ORDER BY churned_customers DESC;


-- 6. Average monthly charges of churned customers
SELECT
    AVG(MonthlyCharges) AS average_monthly_charge
FROM customers
WHERE Churn = 'Yes';


-- 7. Churn by internet service
SELECT
    InternetService,
    COUNT(*) AS total_customers
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers
FROM customers
GROUP BY InternetService
ORDER BY churned_customers DESC;
