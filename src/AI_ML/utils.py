
import os
import pandas as pd
import sys
from src.AI_ML.logger import logging
from src.AI_ML.exception import CustomException
import pymysql
from dotenv import load_dotenv
import pickle
import dill

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

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

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            logging.info(f"Best Params are {gs.best_params_}")

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        logging.error("Error in Evaluation of model")
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
