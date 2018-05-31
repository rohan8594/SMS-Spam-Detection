# File:        parameter_tuning.py
#
# Author:      Rohan Patel
#
# Date:        05/12/2018
#
# Description: This script uses scikit-learn's GridSearchCV to perform an exhaustive grid search.
#              Exhaustive grid search is a way to select the best model out of a family of models
#              by tuning the model parameters. 

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

def SVM_Tuning(X_train, X_test, y_train, y_test):

	print('\n############### SVM ###############\n')
	param_grid = {'kernel':['sigmoid','rbf','linear'],'gamma':[1,0.1,0.01]}

	model = GridSearchCV(SVC(), param_grid, verbose = 3)
	model.fit(X_train, y_train)

	print('\nBest parameter:', model.best_params_)

	pred = model.predict(X_test)

	print('\nAccuracy Score:', accuracy_score(y_test, pred))
	print('\n')
	print(classification_report(y_test, pred))

def MNB_Tuning(X_train, X_test, y_train, y_test):

	print('\n############### Multinomial NB ###############\n')
	param_grid = {'alpha': np.arange(0.05, 1.05, 0.05)}

	model = GridSearchCV(MultinomialNB(), param_grid, verbose = 1)
	model.fit(X_train, y_train)

	print('\nBest parameter:', model.best_params_)

	pred = model.predict(X_test)

	print('\nAccuracy Score:', accuracy_score(y_test, pred))
	print('\n')
	print(classification_report(y_test, pred))

def DTree_Tuning(X_train, X_test, y_train, y_test):

	print('\n############### Decision Tree ###############\n')
	param_grid = {'min_samples_split': np.arange(2, 21, 1)}

	model = GridSearchCV(DecisionTreeClassifier(), param_grid, verbose = 1)
	model.fit(X_train, y_train)

	print('\nBest parameter:', model.best_params_)

	pred = model.predict(X_test)

	print('\nAccuracy Score:', accuracy_score(y_test, pred))
	print('\n')
	print(classification_report(y_test, pred))

def main():

	tfidf_vect = pickle.load(open("output/tfidf_vector.pickle", "rb")) # load previously generated tf-idf vector from pickle file
	messages = pd.read_csv('output/processed_msgs.csv')

	# append our message length feature to the tfidf vector to produce the final feature vector we fit into our classifiers
	len_feature = messages['length'].as_matrix()
	feat_vect = np.hstack((tfidf_vect.todense(), len_feature[:, None]))

	X_train, X_test, y_train, y_test = train_test_split(feat_vect, messages['label'], test_size = 0.3, random_state = 101)

	MNB_Tuning(X_train, X_test, y_train, y_test)
	#SVM_Tuning(X_train, X_test, y_train, y_test)
	#DTree_Tuning(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()