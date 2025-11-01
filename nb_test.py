import pandas as pd
import numpy as np
from naive_bayes import str_column_to_int, nb_train, nb_predict

# Test if KNN accuracy on Iris flower dataset is higher than random-guess (> 0.33).

def test_naive_bayes_classifier():
    # Load training CSV file. The labels are stored in the last column
    train_df = pd.read_csv("./data/IRIS.csv")
    train_data = train_df.to_numpy()

    test_df = pd.read_csv("./data/iris_test.csv")
    test_data = test_df.iloc[:,:-1].to_numpy()
    test_label = test_df.iloc[:,-1:].to_numpy() # Split labels in last column

    # Training label preprocessing
    label_id_dict = str_column_to_int(train_data, len(train_data[0])-1)
    id_label_dict = {value: key for key, value in label_id_dict.items()}

    # Training
    model = nb_train(train_data)

    # Make predictions on test dataset
    predictions = []
    rows, columns = test_data.shape
    for i in range(rows):
        y_p = nb_predict(model, test_data[i])
        predictions.append([id_label_dict[y_p]])

    # Calculate accuracy
    result = predictions == test_label
    accuracy = sum(result == True) / len(result)
    print('Evaluate Naive Bayes classifier on Iris Flower dataset. Accuracy = %.2f' % accuracy)

    assert accuracy > 0.8

def test_naive_bayes_model():
    # Load training CSV file. The labels are stored in the last column
    train_df = pd.read_csv("./data/IRIS.csv")
    train_data = train_df.to_numpy()


    # Training label preprocessing
    label_id_dict = str_column_to_int(train_data, len(train_data[0])-1)
    
    model = nb_train(train_data)

    assert model != None