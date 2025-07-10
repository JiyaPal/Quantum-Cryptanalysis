import mysql.connector

try:
    mydb = mysql.connector.connect(
        host="localhost",
        user="yourusername",
        password="yourpassword",
        database="yourdatabase"
    )
    print("MySQL Connector successfully installed!")
    mydb.close()

except mysql.connector.Error as err:
    print("Error connecting to database:", err)