import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#1A 
"""
check = pd.read_csv("SpotifyFeatures.csv", sep= ",")
songs = check['genre']
number_features = check.columns

print(f"number of songs:", len(songs))
print(f"number of features:", len(number_features))
"""
#1B
data = pd.read_csv("SpotifyFeatures.csv", sep= ",", usecols=["genre","liveness", "loudness"])
data1 = np.array(data.drop(data[data["genre"] != "Pop" ].index))
data2 = np.array(data.drop(data[data["genre"] != "Classical"].index))

ener = np.ones((data1.shape[0])).reshape(-1, 1)
nuller= np.zeros((data2.shape[0])).reshape(-1,1)


data1= np.concatenate((data1, ener), axis =1)
data2= np.concatenate((data2, nuller), axis =1)
"""
print(f"antall Pop sanger: ",data1.shape[0])
print(f"antall classical sanger: ",data2.shape[0])
"""
#1C
new_matrix = np.vstack((data1, data2))
fixed_matrix= new_matrix[:,1:] #make new array and remove genre column from the new_matrix array
first_array = fixed_matrix[:, :2] # make new array and keep only the 2 first column of the fixed_matrix array
second_array = fixed_matrix[:, 2:] # make new array and keep only the last column of the fixed_matrix array

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(first_array, second_array, test_size= 0.2, train_size = 0.8, shuffle = True)
#1D
"""
plt.scatter(data1[:,1], data1[:, 2], s = 5)
plt.scatter(data2[:,1], data2[:, 2], s = 5)
plt.xlabel("liveness")
plt.ylabel("loudness")
plt.show()"""

#2A


def sigmoid(x):
    return 1 / (1 + np.exp(-np.array((x), dtype=np.float64)))

def loss_function(y_true, y_pred):


    return -np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))


def logistic_regression(x_train,y_train, learning_rate = 0.01, iterations = 100):

    amount_samples, amount_features = x_train.shape
    w = np.zeros(amount_features)
    bias = 0
    total_loss = 0
    loss_list = []
    iteration_list =[]

    for iteration in range(iterations):
        total_loss = 0
        iteration_list.append(iteration)
        for i in range(amount_samples):


            linear_modell = np.dot(w, x_train[i]) + bias

            y_pred = sigmoid(linear_modell)
            y_true = y_train[i]


            dw = (y_pred - y_true)* x_train[i].T
            db = np.sum(y_pred - y_true)

            w = w - learning_rate* dw.T
            bias = bias - learning_rate* db


            loss = loss_function(y_true, y_pred)
            total_loss += loss
            
        loss_per_sample = total_loss / amount_samples
        loss_list.append(loss_per_sample)
        
        if iteration % (iterations // 10 ) == 0:
            print(f"the loss after {iteration}  iterations: {loss_per_sample}")

    return w, bias, loss_list, iteration_list

    

#endret på learning rate her i fra 0.001 til 0.01 når jeg skulle sammenligne mellom de to learning rates-ene
weight, bias, loss_list, iteration_list = logistic_regression(x_train, y_train, learning_rate = 0.01, iterations= 100)

xpoints = np.array(loss_list)
ypoints = np.array(iteration_list)

plt.plot(xpoints,ypoints)
plt.show()

def f(x):
     linear_modell = np.dot(weight, x.T) + bias

     y_pred = sigmoid(linear_modell)

     y_pred = np.where(y_pred >= 0.5, 1 ,0)
     
     return y_pred

y_train = np.array(y_train.flatten(), dtype= np.int64)
y_hat_train = f(x_train)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train, y_hat_train)*100)

#2B
y_hat = f(x_test)
y_test = np.array(y_test.flatten(), dtype= np.int64)

print(accuracy_score(y_test, y_hat)*100)



#3A

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_hat))

