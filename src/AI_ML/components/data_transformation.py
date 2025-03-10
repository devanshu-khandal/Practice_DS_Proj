import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.AI_ML.exception import CustomException
from src.AI_ML.logger import logging
from src.AI_ML.utils import save_object

@dataclass

class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_obj(self):
        """
        This function responsible for data transformation
        """
        try:
            # pass
            logging.info("Data Transformation started")

            # Feature Engineering
            num_features = ['writing_score', 'reading_score']
            
            #train_df.select_dtypes(exclude="object").columns
            cat_features = ["gender","race_ethnicity","parental_level_of_education",
                            "lunch",
                            "test_preparation_course"] #test_df.select_dtypes(include="object").columns

            # To handle missing values
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')), # fill missing values with median, imputer is used to fill missing values
                ('std_scaler', StandardScaler()) # standardize the data, it is z = (x - u) / s where u is mean and s is standard deviation
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')), # fill missing values with most frequent or mode value
                ('onehot', OneHotEncoder()), # one hot encoding, it is used to convert categorical data into binary format
                ('std_scaler', StandardScaler(with_mean=False)) # standardize the data, we are doing this because to make the data normally distributed
                # although values will be between 0 and 1
            ])

            logging.info(f"Numerical Features: {num_features}")
            logging.info(f"Categorical Features: {cat_features}")
            
            # Preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, num_features),
                    ('cat_pipeline', cat_pipeline, cat_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            #changing the column names in the standard format
            train_df.columns = train_df.columns.str.replace('/', '_').str.replace(' ', '_')
            test_df.columns = test_df.columns.str.replace('/', '_').str.replace(' ', '_')

            preprocessing_obj = self.get_transformer_obj()

            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']
            #train_df.select_dtypes(exclude=["object",'math score']).columns

            # Divide the train data into dependent and independent variables
            input_features_train_df = train_df.drop([target_column_name], axis=1)
            target_train_df = train_df[target_column_name]

            # Divide the test data into dependent and independent variables
            input_features_test_df = test_df.drop([target_column_name], axis=1)
            target_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on training and testing data")

            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            # Combining the transformed data with target column
            train_arr = np.c_[input_features_train_arr, np.array(target_train_df)]# np.c_ is used to concatenate the arrays along the second axis (axis=1)
            test_arr = np.c_[input_features_test_arr, np.array(target_test_df)]

            logging.info("Save the preprocessor object")

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (train_arr, test_arr, 
                    self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e, sys)