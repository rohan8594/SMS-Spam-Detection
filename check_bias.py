# File:        check_bias.py
#
# Author:      Rohan Patel
#
# Date:        05/17/2018
#
# Description: This script loads a second message dataset that has been scraped from Dublin Institute
#              of Technology's website in order to check the bias of our learned SVM and Multinomial
#              NB classifiers. The classfiers are trained on the UCI dataset and tested on the Dublin 
#              dataset.

import pickle
import numpy as np
import pandas as pd
import text_preprocessing
from xml.dom import minidom
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report

def load_messages2():
    sms_list = []
    
    # parse spam.xml containing 2nd dataset
    xmldoc = minidom.parse('smsspamcollection/spam.xml')
    itemlist = xmldoc.getElementsByTagName('text')
    
    for msg in itemlist:
        sms_list.append(msg.childNodes[0].nodeValue)
    
    messages2 = pd.DataFrame(data=sms_list, columns=['message'])
    messages2['label'] = 'spam'
    messages2['length'] = messages2['message'].apply(len)
    return messages2

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
    messages2 = load_messages2() 
    sms_data = pd.concat([messages, messages2]) # concatenate messages1 and messages2 to create a common tfidf feature vector
    
    tfidf_vect = Tfidf_Vectorization(sms_data) # create a large sparse tfidf feature vector

    # append our message length feature to the tfidf vector to produce the final feature vector we fit into our classifiers
    len_feature = sms_data['length'].as_matrix()
    feat_vect = np.hstack((tfidf_vect.todense(), len_feature[:, None]))
    
    X_train = feat_vect[:5572] # training set is comprised of entire UCI message dataset
    y_train = messages['label']
    X_test = feat_vect[5572:] # test set is comprised of entire Dublin spam message dataset
    y_test = messages2['label']
    
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