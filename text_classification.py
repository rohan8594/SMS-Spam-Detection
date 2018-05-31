# File:        text_classification.py
#
# Author:      Rohan Patel
#
# Date:        05/10/2018
#
# Description: This script contains the main sms classification code. We first load the processed 
#              messages, convert each message into a bag of words and then into a tfidf vector 
#              representation. Then we split the tfidf feature vector into training and test sets, 
#              build our classifier on the training set, and test it on the test set.  

import pickle
import numpy as np
import pandas as pd
import text_preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', -1)

def Tfidf_Vectorization(messages):
    '''
    1. Convert word tokens from processed msgs dataframe into a bag of words
    2. Convert bag of words representation into tfidf vectorized representation for each message
    '''
    
    bow_transformer = CountVectorizer(analyzer=text_preprocessing.text_process).fit(messages['message'])
    bow = bow_transformer.transform(messages['message']) # bag of words

    tfidf_transformer = TfidfTransformer().fit(bow)
    tfidf_vect = tfidf_transformer.transform(bow) # tfidf vector representation
    
    pickle.dump(tfidf_vect, open("output/tfidf_vector.pickle", "wb")) # stores tfidf vector in a pickle file so tha it can be be used later in other scripts
    
    return tfidf_vect

def TrainTestSplit(feature_vect, messages):
    '''
    Split dataset into training and test sets. We use a 70/30 split.
    '''
    X_train, X_test, y_train, y_test = train_test_split(feature_vect, messages['label'], test_size = 0.3, random_state = 101)
    
    return X_train, X_test, y_train, y_test

def train_classifier(classifier, X_train, y_train):
    classifier.fit(X_train, y_train)
    
def predict_labels(classifier, X_test):
    return (classifier.predict(X_test))

def main():
    
    messages = pd.read_csv('output/processed_msgs.csv')

    tfidf_vect = Tfidf_Vectorization(messages)

    # append our message length feature to the tfidf vector to produce the final feature vector we fit into our classifiers
    len_feature = messages['length'].as_matrix()
    feat_vect = np.hstack((tfidf_vect.todense(), len_feature[:, None]))

    X_train, X_test, y_train, y_test = TrainTestSplit(feat_vect, messages)
    
    svm = SVC()
    dtree = DecisionTreeClassifier()
    mnb = MultinomialNB()
    knn = KNeighborsClassifier()
    rfc = RandomForestClassifier()
    ada_boost = AdaBoostClassifier()
    bagging_clf = BaggingClassifier()
    
    classifiers = {'SVM': svm, 'Decision Tree': dtree, 'Multinomial NB': mnb, 'KNN': knn, 'Random Forest': rfc, 
                   'AdaBoost': ada_boost, 'Bagging Classifier': bagging_clf}
    
    X_train2, X_test2 = train_test_split(messages['message'], test_size = 0.3, random_state = 101)
    pred_scores = []
    pred = dict()
    file = open('output/misclassified_msgs.txt', 'a', encoding='utf-8') # misclassified messages will be written in this
    for k, v in classifiers.items():
        train_classifier(v, X_train, y_train)
        pred[k] = predict_labels(v, X_test)
        pred_scores.append((k, [accuracy_score(y_test, pred[k])]))
        print('\n############### ' + k + ' ###############\n')
        print(classification_report(y_test, pred[k]))

        # write misclassified messages into a new text file
        file.write('\n#################### ' + k + ' ####################\n')
        file.write('\nMisclassified Spam:\n\n')
        for msg in X_test2[y_test < pred[k]]:
            file.write(msg)
            file.write('\n')
        file.write('\nMisclassified Ham:\n\n')
        for msg in X_test2[y_test > pred[k]]:
            file.write(msg)
            file.write('\n')
    file.close()

    print('\n############### Accuracy Scores ###############')
    accuracy = pd.DataFrame.from_items(pred_scores, orient = 'index', columns = ['Accuracy Rate'])
    print('\n')
    print(accuracy)
    print('\n')

    '''
    #plot accuracy scores in a bar plot
    accuracy.plot(kind =  'bar', ylim=(0.85,1.0), edgecolor='black', figsize=(10,5))
    plt.ylabel('Accuracy Score')
    plt.title('Distribution by Classifier')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    sys.exit(0)
    '''

if __name__ == "__main__":
    main()