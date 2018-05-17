# File:        check_bias.py
#
# Author:      Rohan Patel
#
# Date:        05/17/2018
#
# Description: This script tests the learned classifier on a different dataset that can be found 
#              in smsspamcollection/spam.xml in order to check for bias.

import pickle
import pandas as pd
from xml.dom import minidom
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
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
    return messages2

def Tfidf_Vectorization(sms_data):
    '''
    1. Convert word tokens from processed msgs dataframe into a bag of words
    2. Convert bag of words representation into tfidf vectorized representation for each message
    '''
    
    bow_transformer = CountVectorizer(analyzer=text_process).fit(sms_data['message'])
    bow = bow_transformer.transform(sms_data['message']) # bag of words

    tfidf_transformer = TfidfTransformer().fit(bow)
    tfidf_vect = tfidf_transformer.transform(bow) # tfidf vector representation
    
    return tfidf_vect

def main():
    
    messages = pd.read_csv("output/processed_msgs.csv")    
    messages2 = load_messages2()
    sms_data = pd.concat([messages, messages2]) # concatenate messages1 and messages2 to form our new dataset
    
    tfidf_vect = Tfidf_Vectorization(sms_data)
    
    #Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(tfidf_vect, sms_data['label'], test_size=0.3, random_state=101)
    
    #SVM
    svm = SVC(C = 100, gamma = 0.01)
    svm.fit(X_train, y_train)
    pred = svm.predict(X_test)
    
    print('\n################# SVM #################\n')
    print(classification_report(y_test, pred))
    print('\n')
    print(accuracy_score(y_test, pred))
    
    #Multinomial Naive Bayes
    mnb = MultinomialNB(alpha = 0.25)
    mnb.fit(X_train, y_train)
    pred = mnb.predict(X_test)
    
    print('\n################# Multinomial NB #################\n')
    print(classification_report(y_test, pred))
    print('\n')
    print(accuracy_score(y_test, pred))
    
if __name__ == "__main__":
    main()