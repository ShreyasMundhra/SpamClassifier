import pandas as pd
import numpy as np
import os
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer as vec

# read txt file
def readtxt(file):
    with open(file,'r+', encoding='utf-8', errors='ignore') as file_obj:
        return file_obj.read()

# separate email into sections e.g. sender, receiver, subject, body etc.
def get_sections(email):
    tokenizer = RegexpTokenizer(r'\w+')
    lines = email.split("\n")
    subject = []
    body = []

    for line in lines:
        words = tokenizer.tokenize(line)
        for i in range(0,len(words)):
            if(words[i].lower() == 'subject'):
                subject = words[i+1:]
                break
            if (i == len(words)-1):
                body = body + words

    print(subject)
    print(body)

    return subject, body

def writeDictToCsv(dict):
    with open('words.csv','wb') as file_object:
        for key in dict:
            file_object.write(key + ',' + str(dict[key]) + '\n')

# create dictionary with the word as key and count as value for entire dataset to find important features for the model
def get_word_dict():
    word_count = {}
    data = []
    for directory in os.listdir('train_data'):
        if(os.path.isdir('train_data/' + directory)):
            for filename in os.listdir('train_data/' + directory):
                email = readtxt('train_data/' + directory + '/' + filename)
                print(filename)
                subject, body = get_sections(email)

                data.append(' '.join(body))
                for word in body:
                    if(word in word_count):
                        word_count[word] = word_count[word] + 1
                    else:
                        word_count[word] = 0

    tf = vec(input='content', analyzer='word', min_df=0, stop_words='english', sublinear_tf=True, decode_error='ignore', max_features=100)
    tfidf_matrix = tf.fit_transform(data)
    print("Features: " + str(tf.get_feature_names()))

    return word_count,tfidf_matrix

# readtxt('Summary.txt')
word_dict,tfidf_matrix = get_word_dict()
pd.DataFrame(tfidf_matrix.toarray()).to_csv("features.csv")