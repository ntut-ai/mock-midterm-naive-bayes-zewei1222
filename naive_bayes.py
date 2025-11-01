#!/usr/bin/env python
# coding: utf-8

# This code refers to https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

import pandas as pd
import numpy as np
import argparse
from csv import reader
from math import sqrt, exp, pi

TRAIN_FILE = './data/IRIS.csv'
TEST_FILE = './data/iris_test.csv'


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
        
# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
        print('[%s] => %d' % (value, i))
 
    for row in dataset:
        row[column] = lookup[row[column]]
    
    return lookup

# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]#row, each flower
        class_value = vector[-1]#last element:species
        if (class_value not in separated):
            separated[class_value] = list()#if not in separated:dict then create a new dict_element
        separated[class_value].append(vector)#add index of dataset in dict
    return separated#looks like {Iris-setosa: list(), Iris-versicolor: list()}

# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers)/float(len(numbers))
 
# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return sqrt(variance)
 
# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]#column is a tuple
    del(summaries[-1])#delete species
    return summaries
 
# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
    #{species: list("row of dataset")}
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
        #summaries = {species: list(mean, stdev, row)}
    return summaries
 
# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent
 
# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    #summaries = [species:list(mean, stdev, row)]
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows) #flower0 / each_flower_species 
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
            #probabilities = [species, probability:float] ex:{0: 0.01, 1: 0.002, 2: 0.5}
    return probabilities



def nb_train(train_data):
    ### Uncomment two lines below and you will pass the train model unit test ###
    model = summarize_by_class(train_data)
    return model
    return None

#######
# Complete this functions:
# Predict the class for a given row
#######
def nb_predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    print(probabilities)
    predict = max(probabilities, key=probabilities.get)
    return predict
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive Bayes Classifier")
    parser.add_argument("--train-csv", help="Training data in CSV format. Labels are stored in the last column.", required=True)
    parser.add_argument("--test-csv", help="Test data in CSV format", required=True)
    args = parser.parse_args()

    # Load training CSV file. The labels are stored in the last column
    train_df = pd.read_csv(args.train_csv)
    train_data = train_df.to_numpy()

    test_df = pd.read_csv(args.test_csv)
    #test_data = test_df.to_numpy()
    test_data = test_df.iloc[:,:-1].to_numpy()
    test_label = test_df.iloc[:,-1:].to_numpy() # Split labels in last column

    # Training label preprocessing
    label_id_dict = str_column_to_int(train_data, len(train_data[0])-1)
    id_label_dict = {value: key for key, value in label_id_dict.items()}

    # Training
    #model = summarize_by_class(train_data)
    model = nb_train(train_data)

    # Validating
    row = [5.7,2.9,4.2,1.3]
    label = nb_predict(model, row)
    print('Validation on Data=%s, Predicted: %s' % (row, label))

    # Make predictions on test dataset
    predictions = []
    rows, columns = test_data.shape
    for i in range(rows):
        y_p = nb_predict(model, test_data[i])
        predictions.append([id_label_dict[y_p]])

    # Calculate accuracy
    result = np.array(predictions) == test_label
    accuracy = sum(result == True) / len(result)
    print('Evaluate Naive Bayes on Iris Flower dataset. Accuracy = %.2f' % accuracy)