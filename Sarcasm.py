#Importing all the libraries 

import pandas as pd 
import numpy as np
import re
from nltk.stem.porter import PorterStemmer

data = pd.read_json("C:/Users/Dell/Desktop/Yash/Machine Learning/Dataset/Sarcasm/Sarcasm_Headlines_Dataset.json", lines = True)

#Cleaning the data from headline column
data['headline'] = data['headline'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))

#Getting features and labels for our model
features = data['headline']
labels = data['is_sarcastic']

#Stemming the headlines 
ps = PorterStemmer()
features = features.apply(lambda x: x.split())
features = features.apply(lambda x: ' '.join([ps.stem(word) for word in x]))

#Creating a corpus and Document of words 
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(max_features = 6000)
features = list(features)
features = tv.fit_transform(features).toarray()

#Splitting of Data 
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = .15, random_state = 42)

#Training the model using different Machine Learning techniques 

#Techniques 1
#Using Support Vector CLassifier

from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(features_train, labels_train)

#Technique 2
#Using Naive Bayes Algorithn

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(features_train, labels_train)

#Technique 3
#Logistic Regression 

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(features_train, labels_train)

#Tecgnique 4 
#Random Forest Algorithm 

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 1000, random_state = 0, n_jobs = -1)
rfc.fit(features_train, labels_train)

#Finding out the accuracy of each model
print("The Accuracy of Support Vector Classifier is: ", svc.score(features_test, labels_test))
print("The Accuracy of Naive Bayes Algorithm is: ", nb.score(features_test, labels_test))
print("The Accuracy of Logistic Regression Classifier is: ", lr.score(features_test, labels_test))
print("The Accuracy of Random Forest Algorithm is: ", rfc.score(features_test, labels_test))

Sample_input = ['Light travels faster than sound. This is why some people appear bright until they speak']
