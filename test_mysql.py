import mysql.connector

try:
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="shikharkhare03@#$",
        database="customer_churn"
    )

    print("✅ MySQL connected successfully!")

    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM customers")

    result = cursor.fetchone()
    print("Customers in database:", result[0])

    cursor.close()
    connection.close()

except mysql.connector.Error as e:
    print("❌ MySQL connection failed:")
    print(e)
