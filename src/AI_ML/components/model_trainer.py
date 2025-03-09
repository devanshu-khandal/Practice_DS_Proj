import os
import sys
import numpy as np
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn
# import dagshub #open source lib
from urllib.parse import urlparse

from src.AI_ML.exception import CustomException
from src.AI_ML.logger import logging
from src.AI_ML.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "trained_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self, true, predicted):
        mae = mean_absolute_error(true, predicted)
        # mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mean_squared_error(true, predicted))
        r2_square = r2_score(true, predicted)
        return rmse,mae, r2_square

    def initiate_train_model(self,train_array,test_array):
        """
        This function is responsible for training the model
        """
        try:
            logging.info("Train test split started")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "Linear Regression":{},

                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report : dict = evaluate_models(X_train,y_train,X_test,y_test,
                                                  models,params)
            
            # To get the best model_score from dict
            best_model_score = max(sorted(model_report.values()))

            # to get the name of the best model from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"This is our best Model : {best_model_name}")

            model_names = list(params.keys())
            
            actual_model = ""
            for model in model_names:
                if best_model_name==model:
                    actual_model = actual_model + model

            best_params = params[actual_model]

            # mlflow (tracking/logging in our dagshub)
            logging.info("Tracking of Model started")
            # dagshub.init(repo_owner='devanshu-khandal', repo_name='Practice_DS_Proj', mlflow=True)
            mlflow.set_registry_uri("https://dagshub.com/devanshu-khandal/Practice_DS_Proj.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            with mlflow.start_run():

                predicted_qualities = best_model.predict(X_test)

                (rmse,mae,r2) = self.evaluate_model(y_test,predicted_qualities)

                mlflow.log_params(best_params)

                mlflow.log_metric("rmse",rmse) # If you want to give metrices then use .log_metrics (dict)
                mlflow.log_metric("mae",mae)
                mlflow.log_metric("r2", r2)

                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(best_model, "trained_model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "trained_model")

            logging.info("Tracking of Model started")

            # Threshold for our score, if it falls below this then we will end our training
            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found on both training and testing dataset : {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_scoring = r2_score(y_test,predicted)
            return r2_scoring
        
        except Exception as e:
            raise CustomException(e, sys)

