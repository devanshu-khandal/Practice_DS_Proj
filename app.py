
from flask import Flask, request, render_template
from src.AI_ML.pipelines.prediction_pipeline import CustomData, PredictPipeline

# from src.AI_ML.logger import logging
# from src.AI_ML.exception import CustomException
# import sys
 
app = Flask(__name__)

## Rotuing to our landing page (in our case we are creating other wise you'll share the api to the developer)

@app.route('/')
def index():
    return render_template('index.html')

## Routing to the predict data page

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method =='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        pred_df = data.get_data_as_df()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        final_results = predict_pipeline.predict(features=pred_df)

        return render_template('home.html',results= final_results[0])

if __name__=="__main__":
    app.run(host='0.0.0.0',port='8000',debug=True)