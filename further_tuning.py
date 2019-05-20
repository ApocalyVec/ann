# Hyper parameters: epochs, batch size, the optimizer, the number of neurons

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydot

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  # we put 13 instead of 12 because the upper bound is excluded from the range
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable

# In this case, we have two independent variables to encode: country and gender
# because ANN can only take numeric values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])  # convert the first encoder feature (country) to numeric values

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# ct = ColumnTransformer(
#     [('one_hot_encoder', OneHotEncoder(), [1])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
#     remainder='passthrough'                         # Leave the rest of the columns untouched
# )

onehotencoder = OneHotEncoder(categorical_features=[1])  # create dummy variable, instead of having one column
# holding the country information, we'll have three column of dummy variables to do so.
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Tuning the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense


# add input arguement optimizer
def build_classifier(optimizer, units_first, units_second):
    from keras.layers import Dropout
    classifier = Sequential()

    # the first hidden layer with dropout
    classifier.add(Dense(units=units_first, kernel_initializer='uniform', activation='relu', input_dim=11))

    # the second hidden layer with dropout
    classifier.add(Dense(units=units_second, kernel_initializer='uniform', activation='relu'))

    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    # changed optimizer from 'adam' to the function input optimizer
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


# build_fn = the function that returns a model of the ANN
# removed batch size and np_epoch
classifier = KerasClassifier(build_fn=build_classifier)

# contains all the combinations that we want to try
parameters = {'batch_size': [25, 32, 64, 128],
              'epochs': [100, 500, 1000],
              'optimizer': ['adam', 'rmsprop', 'SGD'],
              'units_first': [3,6,9,11],
            'units_second': [3,6,9,11]}  # key = the hyper parameters to tune, value = the different value that we
# want to try for these hyper parameters
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)  # cv = 10 for 10 fold cross validation

# fit the grid search to the training set
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
