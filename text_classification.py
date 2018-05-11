# File:        text_classification.py
#
# Author:      Rohan Patel
#
# Date:        05/10/2018
#
# Description:

import pandas as pd
import pickle
import text_preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def Tfidf_Vectorization(messages):
    '''
    1. Convert word tokens from processed msgs dataframe into a bag of words
    2. Convert bag of words representation into tfidf vectorized representation for each message
    '''
    
    bow_transformer = CountVectorizer(analyzer=text_preprocessing.text_process).fit(messages['message'])
    bow = bow_transformer.transform(messages['message']) # bag of words

    tfidf_transformer = TfidfTransformer().fit(bow)
    tfidf_vect = tfidf_transformer.transform(bow) # tfidf vector representation
    
    pickle.dump(tfidf_vect, open("output/tfidf_vector.pickle", "wb")) # stores tfidf vector in a pickle file so it could be used later in future scripts
    
    return tfidf_vect

def TrainTestSplit(tfidf_vect, messages):
    '''
    Split dataset into training and test sets. We use a 70/30 split.
    '''
    X_train, X_test, y_train, y_test = train_test_split(tfidf_vect, messages['label'], test_size = 0.3, random_state = 101)
    
    return X_train, X_test, y_train, y_test

def main():
    
    messages = pd.read_csv('output/processed_msgs.csv')

    tfidf_vect = Tfidf_Vectorization(messages)
    X_train, X_test, y_train, y_test = TrainTestSplit(tfidf_vect, messages)
    
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    pred = mnb.predict(X_test)
    print(pred)
    print('\n')
    print('Accuracy Score:', accuracy_score(y_test, pred))
    print('\n')
    print(classification_report(y_test, pred))
    
if __name__ == "__main__":
    main()