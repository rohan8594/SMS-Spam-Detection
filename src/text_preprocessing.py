# File:        text_preprocess.py
#
# Author:      Rohan Patel
#
# Date:        05/09/2018
#
# Description: This script loads the sms spam data, organizes the data into a pandas dataframe, adds a new 
#              feature (length of message) to the data, applies some basic text pre-processing techniques 
#              like stopword removal and punctuation removal, and converts the messages into a list of processed 
#              tokens. Finally, the processed dataframe is copied into a new csv file processed_msgs.csv

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import string
nltk.download('stopwords')

def loadData():
    
    messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names = ['label', 'message'])
    messages['length'] = messages['message'].apply(len)
    
    return messages

def text_process(data):
    '''
    1. remove punc
    2. do stemming of words
    3. remove stop words
    4. return list of clean text words
    '''
    nopunc = [c for c in data if c not in string.punctuation] #remove punctuations
    nopunc = ''.join(nopunc)
    
    stemmed = ''
    nopunc = nopunc.split()
    for i in nopunc:
        stemmer = SnowballStemmer('english')
        stemmed += (stemmer.stem(i)) + ' ' # stemming of words
        
    clean_msgs = [word for word in stemmed.split() if word.lower() not in stopwords.words('english')] # remove stopwords
    
    return clean_msgs

def main():
    
    messages = loadData()
    #print(messages)
    messages['processed_msg'] = messages['message'].apply(text_process)
    
    print('\n################################################## Processed Messages ##################################################\n')
    with pd.option_context('expand_frame_repr', False):
    	print (messages)
    #print(messages)

    messages.to_csv('output/processed_msgs.csv', encoding='utf-8', index=False) #copy processed messages dataframe to a new csv file

if __name__ == "__main__":
    main()