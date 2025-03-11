# Subject Score Prediction

### Introduction About the Data :

**The dataset** The goal is to predict `score` of given subject (here we've taken Maths) (Regression Analysis).

There are 10 independent variables :

- **`gender`** : The gender of the student (e.g., Male, Female, Non-binary).  
- **`race_ethnicity`** : The racial or ethnic background of the student, categorized into different groups (e.g., Group A, Group B, etc.).  
- **`parental_level_of_education`** : The highest level of education attained by a student's parent(s), which may influence academic performance (e.g., High school, Associate’s degree, Bachelor’s degree).  
- **`lunch`** : The type of lunch program the student is enrolled in, indicating economic background (e.g., Standard, Free/Reduced-price lunch).  
- **`test_preparation_course`** : Whether the student has completed a test preparation course before the exam (e.g., Completed, Not completed).  
- **`reading_score`** : The student’s score in the reading section of the standardized test, representing comprehension skills (numerical value).  
- **`writing_score`** : The student’s score in the writing section of the standardized test, indicating proficiency in written communication (numerical value).  

Target variable:
* `maths score`: Score of the student.

Dataset Source Link :
[https://www.kaggle.com/datasets/spscientist/students-performance-in-exams/data?select=StudentsPerformance.csv](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams/data?select=StudentsPerformance.csv)

### It is observed that the categorical variables 'gender', 'race_ethnicity' and 'lunch' are cardinal in nature

# Approach for the project 

1. Data Ingestion : 
    * In Data Ingestion phase the data is first read as csv or from local db (through pymysql server)
    * Then the data is split into training and testing and saved as csv file.

2. Data Transformation : 
    * In this phase a ColumnTransformer Pipeline is created.
    * for Numeric Variables first SimpleImputer is applied with strategy median , then Standard Scaling is performed on numeric data.
    * for Categorical Variables SimpleImputer (to cater the null values) is applied with most frequent strategy, then ordinal encoding performed , after this data is scaled with Standard Scaler.
    * This preprocessor is saved as pickle file.

3. Model Training : 
    * In this phase base model is tested . The best model found was Linear regressor.
    * After this hyperparameter tuning is performed on the same.
    * A final LinearRegressor is created.
    * This model is saved as pickle file.

4. Prediction Pipeline : 
    * This pipeline converts given data into dataframe and has various functions to load pickle files and predict the final results in python.

5. Flask App creation : 
    * Flask app is created with User Interface to predict the score inside a simple Web Application.

# Exploratory Data Analysis Notebook

Link : [EDA Notebook](https://github.com/devanshu-khandal/Practice_DS_Proj/blob/main/notebooks/1%20.%20EDA%20STUDENT%20PERFORMANCE%20%20(1).ipynb)

# Model Training Approach Notebook

Link : [Model Training Notebook](https://github.com/devanshu-khandal/Practice_DS_Proj/blob/main/notebooks/2.%20MODEL%20TRAINING.ipynb)
