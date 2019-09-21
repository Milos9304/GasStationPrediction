import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from perceptron import Perceptron

SEED=2019

dataset = pd.read_csv("GasStationsDataset.csv")
print("Dataset entries: " + str(dataset.shape[0]))

#stations < 2500, parkings < 2500, traffic < 1500 
X = dataset.iloc[: ,[6, 11, 14]]
#4sq checking < 50
y = dataset.iloc[:, 17]

fig, ax = plt.subplots()

plt.figure(figsize=(15, 10))
sns.heatmap(dataset.corr(), annot=True)
#plt.show()

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
scores = []

colors = ['r', 'g', 'b']

fig, ax = plt.subplots()
for i in range(3):
    ax.scatter(X[:,i], y, color=colors[i])
#plt.show()

scores=[]

cv = KFold(n_splits=5, random_state=SEED, shuffle=True)
for train_index, test_index in cv.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    p=Perceptron(3)
    epochs = 0

    # Instantiate mse for the loop
    mse =999

    while (epochs < 1000):

        # Epoch cumulative error
        error = 0

        # For each set in the training_data
        for i in range(len(X_train)):

            # Calculate the result
            output = p.result(X_train[i,:])

            # Calculate the error
            iter_error = abs(y_train.as_matrix()[i] - output)
            # Add the error to the epoch error
            error += iter_error

            # Adjust the weights based on inputs and the error
            p.weight_adjustment(X_train[i,:], iter_error)
            #print(str(output)+"c")
            #print(p.result(X_train[i,:]))

        # Calculate the MSE - epoch error / number of sets
        mse = float(error/len(X_train))

        # Print the MSE for each epoch
        #print("The MSE of %d epochs is %.10f" + str(epochs) + " " + str(mse))

        # Every 100 epochs show the weight values
        #if epochs % 100 == 0:
        #    print("0: %.10f - 1: %.10f - 2: %.10f - 3: %.10f" % (p.w[0], p.w[1], p.w[2], p.w[3]))

        # Increment the epoch number
        epochs += 1
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)
    #best_svr = SVR(kernel='linear')
    error=0
    for i in range(len(X_test)):
        output = p.result(X_test[i,:])
        error += abs(y_test.as_matrix()[i]-output)
    score = float(error/len(X_test))
    scores.append(score)
    print("Fold score: " + str(score))
print("Mean score: " + str(np.mean(scores)))
