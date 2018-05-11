import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

def loadData():
    
    messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names = ['label', 'message'])
    messages['length'] = messages['message'].apply(len)
    
    return messages

def text_process(data):
    '''
    1. remove punc
    2. remove stop words
    3. return list of clean text words
    '''
    
    nopunc = [c for c in data if c not in string.punctuation] #remove punctuations
    nopunc = ''.join(nopunc)
    
    #nltk.download('stopwords')
    clean_msgs = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')] # remove stopwords
    
    return clean_msgs

def main():
    
    messages = loadData()
    #print(messages)
    messages['message'] = messages['message'].apply(text_process)
    
    print('\n########## Clean Messages ##########\n')
    print(messages)

if __name__ == "__main__":
    main()