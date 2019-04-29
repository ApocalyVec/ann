import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Evaluating the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def build_classifier():
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    classifier = Sequential()

    # the first hidden layer with dropout
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dropout(p=0.1))

    # the second hidden layer with dropout
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(p=0.1))

    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


# build_fn = the function that returns a model of the ANN
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
# cv is the number of fold, we are using cv=10: 10-fold cross validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)

mean = accuracies.mean()
variance = accuracies.std()
