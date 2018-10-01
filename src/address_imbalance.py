# File:        address_imbalance.py
#
# Author:      Rohan Patel
#
# Date:        05/22/2018
#
# Description: This script cuts down the UCI message dataset to 747 spam messages and 1000 ham messages
#              to address the imbalance of the previously used dataset. The classifiers are re-trained 
#              to check if changing the composition of the dataset affects the final results.

import numpy as np
import pandas as pd
import text_preprocessing
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report

def Generate_balanced_dataset(messages):
    '''
    generate a more balanced dataset containing 1000 ham and 747 spam messages
    '''
    ham_msg = messages[messages['label'] == 'ham'] # ham messages
    spam_msg = messages[messages['label'] == 'spam'] # spam messages
    
    ham_msg = ham_msg.reset_index(drop=True)[:1000] # pick only top 1000 ham messages
    spam_msg = spam_msg.reset_index(drop=True) # pick all 747 spam messages
    
    balanced_data = pd.concat([ham_msg, spam_msg]).sample(frac=1).reset_index(drop=True) # concatenate spam and ham messages to create a more balanced dataset
    
    return balanced_data

def Tfidf_Vectorization(sms_data):
    '''
    1. Convert word tokens from processed msgs dataframe into a bag of words
    2. Convert bag of words representation into tfidf vectorized representation for each message
    '''
    
    bow_transformer = CountVectorizer(analyzer=text_preprocessing.text_process).fit(sms_data['message'])
    bow = bow_transformer.transform(sms_data['message']) # bag of words

    tfidf_transformer = TfidfTransformer().fit(bow)
    tfidf_vect = tfidf_transformer.transform(bow) # tfidf vector representation
    
    return tfidf_vect

def main():
    
    messages = pd.read_csv("output/processed_msgs.csv")
    
    balanced_data = Generate_balanced_dataset(messages)
    tfidf_vect = Tfidf_Vectorization(balanced_data)

    # append our message length feature to the tfidf vector to produce the final feature vector we fit into our classifiers
    len_feature = balanced_data['length'].as_matrix()
    feat_vect = np.hstack((tfidf_vect.todense(), len_feature[:, None]))
    
    X_train, X_test, y_train, y_test = train_test_split(feat_vect, balanced_data['label'], test_size=0.3, random_state=101)
    
    #Multinomial Naive Bayes
    mnb = MultinomialNB(alpha = 0.10000000000000001)
    mnb.fit(X_train, y_train)
    pred = mnb.predict(X_test)
    
    print('\n################# Multinomial NB #################\n')
    print(classification_report(y_test, pred))
    print('\n')
    print(accuracy_score(y_test, pred))
    
    #SVM
    svm = SVC(kernel = 'linear', gamma = 1)
    svm.fit(X_train, y_train)
    pred = svm.predict(X_test)
    
    print('\n################# SVM #################\n')
    print(classification_report(y_test, pred))
    print('\n')
    print(accuracy_score(y_test, pred))
    
if __name__ == "__main__":
    main()