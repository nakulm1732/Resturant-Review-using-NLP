# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 01:56:00 2019

@author: nakul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
nltk.download('stopwords')
from nltk.corpus import stopwords
#To ignore double codes use quoting = 3
data = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#STep 2: CLean the reviews
import re
review = re.sub('[^a-zA-Z]', ' ', data['Review'][0])
#ABove, We are trying to remove all dots, and not remore all the capital and small aplhabets. 
#FUrthrmore, use a space(' ') so that the alphabets dont stick together, and the deleted dots, comma's etc are replaced by spaces

#STep 2: COnvert all captial to small letters
review = review.lower()

#STep 3: Remove unwanted words or irrelavent words
import nltk
nltk.download('stopwords')
#Split the string
review = review.split()

rev = [word for word in review if not word in set(stopwords.words('english'))]

#STep 4: Stemming(Taking the root of the word, meaning taking key words)
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
review2 = re.sub('[^a-zA-Z]', ' ', data['Review'][0])
review2 = review2.lower()

ps = PorterStemmer()

review2 = [ps.stem(word) for word in review2 if not word in set(stopwords.words('english'))]#'This' is removed


#STep 5: COnvert the string into list of 3 words, and join the 3 words together to make a sentence
review2 = ' '.join(review2)

#STep 6 : Do the same for all the 1000 reviews
corpus = []
len(data)
for i in range(0,len(data)):
    review3 = re.sub('[^a-zA-z]', ' ', data['Review'][i])
    review3 = review3.lower()
    review3 = review3.split()
    review3 = [ps.stem(word) for word in review3 if not word in set(stopwords.words('english'))]
    review3 = ' '.join(review3)
    corpus.append(review3)
    #Results are in corpus
    
#Step 7: Creating Bag of Words#SParse Matrix
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)#max_features keeps the most relavent words in the feature
X = cv.fit_transform(corpus).toarray()

#Creating a model to predict whether an individual liked or not
Y = data.iloc[:,1].values

#Most common models used for Natural Language processing are Naive Bayes and Decision Tree classification

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)


#Fitting Naive Bayes onto Training Set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

#Predicting the results
y_pred = classifier.predict(X_test)
#Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

    
