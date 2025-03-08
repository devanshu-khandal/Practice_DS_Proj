
import os
import pandas as pd
import sys
from src.AI_ML.logger import logging
from src.AI_ML.exception import CustomException
import pymysql
from dotenv import load_dotenv
import pickle

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
database = os.getenv("database")
port = os.getenv("port")
# print(str(database))
def read_sql_data():
    # logging.info("Reading data from MySQL database")

    try:
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            # port=int(port)
        )
        logging.info("Connection Established", conn)

        query = f"select * from studentsperformance"

        df=pd.read_sql(query, conn)
        print(df.head())
        logging.info("Data read successfully")

        return df
    
    except Exception as ex:
        logging.error("Error in connecting to MySQL database")
        raise CustomException(ex,sys)
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
            logging.info(f"Object saved at {file_path}")

    except Exception as ex:
        
        logging.error("Error in saving object")
        raise CustomException(ex, sys)
