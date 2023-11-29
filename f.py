import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pickle

def load_data(file_path):
    return pd.read_csv(file_path)

def prepare_data(data):
    X = data['cgpa'].values.reshape(-1, 1)
    Y = data['placement_exam_marks'].values.reshape(-1, 1)
    return X, Y

def split_data(X, Y, test_size=0.25, random_state=21):
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)

def train_model(X_train, Y_train):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, Y_train)
    return lin_reg

def visualize_results(X_test, Y_test, model):
    plt.scatter(X_test, Y_test, color='black')
    plt.plot(X_test, model.predict(X_test), color='red')
    plt.title('Placements (Test set)')
    plt.xlabel('CGPA')
    plt.ylabel('Placement Exam Marks')
    plt.show()

def evaluate_model(Y_test, test_pred):
    score = r2_score(Y_test, test_pred)
    print("R2 Score is =", score)  # printing the accuracy
    print("MSE is =", mean_squared_error(Y_test, test_pred))
    print("RMSE is =", np.sqrt(mean_squared_error(Y_test, test_pred)))

def save_model(model, file_path='model.pkl'):
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
   
    data_file_path = "placement.csv"
    placement_data = load_data(data_file_path)

    
    X_data, Y_data = prepare_data(placement_data)

    X_train, X_test, Y_train, Y_test = split_data(X_data, Y_data)

    trained_model = train_model(X_train, Y_train)

  
    visualize_results(X_test, Y_test, trained_model)

  
    test_pred = trained_model.predict(X_test)

    evaluate_model(Y_test, test_pred)

   
    save_model(trained_model, 'model10.pkl')
