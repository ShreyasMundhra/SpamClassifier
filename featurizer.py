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

def writeDictToCsv(di):
    with open('words.csv','wb') as file_object:
        for key in di:
            file_object.write(key + ',' + str(di[key]) + '\n')

def writeTfIdfToCsv(data):
    tf = vec(input='content', analyzer='word', min_df=0, stop_words='english', sublinear_tf=True, decode_error='ignore',
             max_features=100)
    tfidf_matrix = tf.fit_transform(data)
    print("Features: " + str(tf.get_feature_names()))

    df = pd.DataFrame(data=tfidf_matrix.toarray(),columns=tf.get_feature_names())
    df.to_csv('tfidf.csv',index=False)

# create dictionary with the word as key and count as value for entire dataset to find important features for the model
def get_email_bodies():
    # word_count = {}
    data = []
    classes = []
    for directory in os.listdir('train_data'):
        if(os.path.isdir('train_data/' + directory)):
            for filename in os.listdir('train_data/' + directory):
                if directory == 'ham':
                    classes.append('HAM')
                else:
                    classes.append('SPAM')
                email = readtxt('train_data/' + directory + '/' + filename)
                print(filename)
                subject, body = get_sections(email)

                data.append(' '.join(body))

    return data, classes

def evaluate_on_test_set(test, classifier, tf):
    test_input = tf.transform(test['body'])
    test_target = test['class'].values
    return classifier.score(test_input, test_target)
