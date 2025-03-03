#%%
# from flask import Flask, request, jsonify
from src.AI_ML.logger import logging
from src.AI_ML.exception import CustomException
import sys
from src.AI_ML.components.data_ingestion import DataIngestion, DataIngestionConfig

#%%
if __name__ == '__main__':
    logging.info("Execution has started")

    try:
        # data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
                
    except Exception as e:
        logging.info(f"Custom Exception raised : {CustomException(e, sys)}")
        raise CustomException(e, sys)
    # app.run(debug=True)

