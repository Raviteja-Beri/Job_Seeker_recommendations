from dotenv import load_dotenv
import mysql.connector
import os

load_dotenv()

def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

def fetch_all_jobs():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM jobs")
        rows = cursor.fetchall()

        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        print(f"Error fetching jobs from database: {e}")
        return []
