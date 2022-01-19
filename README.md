# Disaster-Response Webb App

### Table of Contents

1. [Installation](#installation)
2. [Project Summary](#summary)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code should run with no issues using Python versions 3.* with the following packages: json, plotly, numpy, pandas, nltk, flask, sklearn, sys, re, pickle and sqlalchem.

## Project Summary<a name="summary"></a>

This project is webb app that uses a machine learning pipeline to categorize these disaster relief messages so that new messages can be sent to the appropriate disaster relief agency. An emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## File Descriptions <a name="files"></a>

process_data.py: This is an ETL pipeline. This takes as its input csv files containing message data and message categories, merges the message data and message categories, cleans the merged data and saves the merged data to a SQLite database.

train_classifier.py: This is an MTL pipeline. This takes the SQLite database produced by process_data.py as input and uses the data to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. The model uses a k-nearest neighbor algorithm to classify the data. A gridsearch was used to find the optimal number of k neighbors. Given the limited amount of computation and time, the gridsearch simply compares 5 and 10 neighbors.

run.py: The web app.

## Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to run the web app.

    - Create a directory called data and save the process_data.py file in it. To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - Create a directory called models and save the train_classifier.py file in it. To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Create an app directory in the current working directory. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Results

The data provided was highly unbalanced. Most of the existing catagories provided were in the related category (related, air_related, weather_related), accounting for roughly 77% of the categorized data. Many of the catagories had only a few positive classifications. Based on the unbalanced data, the accuracy for all catagories is high (above 90% for most categories). The model simply predicts that the messages do not fall into these categories and since the majority of messages do not, this results in high accuracy. However, the proportion of positive examples that were correctly classified (recall) tends to be low since there simply isn't enough positive casses to train the model.

![newplot](https://user-images.githubusercontent.com/91521736/150093300-ab54a06e-9c96-44bd-858f-3f0cbee32e7d.png)

![newplot (1)](https://user-images.githubusercontent.com/91521736/150098670-5269054a-e3b8-49d5-9a94-8adde1a672cf.png)

![newplot (2)](https://user-images.githubusercontent.com/91521736/150099161-f10b9941-c2b0-4d76-ba0d-ded8b99f434b.png)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Code templates and data were provided by Udacity.
