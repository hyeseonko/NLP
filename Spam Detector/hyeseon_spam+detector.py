########################################################
# Title: Spam Detector
# Date: 2018-01-08
# Reference: Lazy Programmer
#########################################################

# Multinomial Naive Bayes is for classification with discrete features
# e.g., word counts for text classification

# Use Multinomial naive Bayes
from sklearn.naive_bayes import MultinomialNB
import pandas as pd     # To read csv file
import numpy as np      # To calculate as a matrix, array

data = pd.read_csv('spambase.csv').as_matrix()
np.random.shuffle(data)     # randomly split the data into train and test and it's different at every time

# We are dealing with pre-processed data (spambase.csv)
# Column 1 ~ 48: Word-frequency in the documents
# column 49: Labeling into 0 or 1 (spam=1, 0=not spam)
X = data[:, :48]
Y = data[:, -1]

# Train data are (total-last 100) rows
X_train = X[:-100,]
Y_train = Y[:-100,]

# Test data are last 100 rows
X_test = X[-100:, ]
Y_test = Y[-100:, ]

model = MultinomialNB()
model.fit(X_train, Y_train)
print("Accuracy for NB:", model.score(X_test, Y_test))

# USE AdaBoost
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(X_train, Y_train)
print("Accuracy for AdaBoost: ", model.score(X_test, Y_test))
