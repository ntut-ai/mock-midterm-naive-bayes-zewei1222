import pandas as pd
import numpy as np
import argparse
TRAIN_FILE = './data/IRIS.csv'

def str_column_to_int(dataset, column):
    class_value = [row[column] for row in dataset]
    #print(f"{class_value=}")
    unique = set(class_value)
    #print(f"{unique=}")
    look_up = dict()
    for i, value in enumerate(unique):
        look_up[value] = i
        print(f"{i=} ,{value=}")
    print(f"{look_up=}")

    print(dataset[0])

    for row in dataset:
        row[column] = look_up[row[column]]
        print(row)

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description="Naive Bayes Classifier")
    parser.add_argument("--train-csv", help="Training data in CSV format. Labels are stored in the last column.", required=True)
    args = parser.parse_args()
    train_df = pd.read_csv("./data/IRIS.csv")
    train_data = train_df.to_numpy()
    str_column_to_int(train_data, len(train_data[0])-1)
    #print(f"{train_data=}")