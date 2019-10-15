# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('spam.csv',encoding='latin1') 

#Cleanig The Text
# Removing Stop Words 
import re
from nltk.corpus import stopwords # Removing The Stop Words
from nltk.stem.porter import PorterStemmer # Stemming
ps =  PorterStemmer()
corpus= []

for i in range(0,len(dataset['Message'][:])):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag oF Words Model
# How many time a Word appers 
# This Is Very Useful For Classicfication 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=6000)
X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:, 0].values


# Runnig The Classifiers
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Naive Bays Classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=30,random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.externals import joblib
joblib.dump(classifier,'RandomForest.pkl')





