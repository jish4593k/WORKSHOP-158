

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pickle

def load_data(file_path):
    return pd.read_csv(file_path)

def prepare_data(data):
    features = data[['EXAM1', 'EXAM2', 'EXAM3']]
    target = data['FINAL']
    return features, target

def train_model(features, target, test_size=0.25, random_state=6):
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)
    lr_model = LinearRegression()
    lr_model.fit(x_train, y_train)
    return lr_model, x_test, y_test

def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print("RMSE:", rmse)
    print("R^2:", r2)
    return predictions

def save_model(model, file_path='model.pkl'):
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    # Load data
    data_file_path = 'finalexam.csv'
    exam_data = load_data(data_file_path)

    exam_features, exam_target = prepare_data(exam_data)

    
    trained_model, test_features, test_target = train_model(exam_features, exam_target)


    model_predictions = evaluate_model(trained_model, test_features, test_target)


    model_file_path = 'model2.pkl'
    save_model(trained_model, model_file_path)

    
    new_data = np.array([[66, 69, 65]])
    new_predictions = trained_model.predict(new_data)
    print("Predictions for new data:", new_predictions)
