import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()

try:
  

    conn = psycopg2.connect(
    dbname = os.getenv("POSTGRESS_DB_NAME"),
    user = os.getenv("POSTGRESS_DB_USER"),
    password = os.getenv("POSTGRESS_DB_PASSWORD"),
    host = os.getenv("POSTGRESS_DB_HOST"),
    port = os.getenv("POSTGRESS_DB_PORT")

)
    print("Connection successful!")
except Exception as e:
    print(f"Error: {e}")
