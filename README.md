# Data-Science-Disaster-Response-Pipeline
This is Project 1 from Udacity Data Science Nano-degree
Term 2 ( Disaster Response Pipeline ).

# How to run the Python scripts and web app
Run the following command in the app's directory to run your web app. python run.py

Next, go to http://0.0.0.0:3001/

# Repository :-
•	Run.py – This Python file contains the following steps:

        •	Initializes a flask app

        •	Tokenizes and normalizes text

        •	Loads data from a database

        •	Loads a model from “Train_classifer”

        •	Returns a website that displays model results

•	Go.html – contains html code for master.html

•	Master.html – allows users to enter messages that are then automatically classified

# Process_data – This Python file contains the following steps:

        •	Loads csv data containing category and messages data

        •	Cleans the data by splitting the category field and dropping duplicates

        •	Saves that data into a database.

•	DisasterResponse.db – this database fie contains the clean data processed in “process_data.py”

•	Disaster_categories.csv – this text file contains a column with concatenates category names

•	Disaster_messages.csv – this text file contains messages typed by disaster victims

# 	Train_classifer – This Python file contains the following steps:

        •	Tokenizes disaster message text, normalizes that text to lower case, and removes stop words

        •	Builds an Adaboost model that uses grid search to optimize it’s hyperparameters

        •	Evaluates the model and predicts the categories of messages

        •	The trained model is saved as “pickle” into the run.py file
