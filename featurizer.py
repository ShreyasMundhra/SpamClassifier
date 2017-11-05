import pandas as pd
import numpy as np
import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag, ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer as vec


# read txt file
def readtxt(file):
    with open(file,'r+', encoding='latin-1', errors='ignore') as file_obj:
        return file_obj.read()

def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words

# separate email into sections e.g. sender, receiver, subject, body etc.
def get_sections(email):
    # tokenizer = RegexpTokenizer(r'\w+')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    lines = email.split("\n")
    subject = []
    body = []

    for line in lines:
        words = tokenizer.tokenize(line)
        # pos = nltk.pos_tag(words)
        # print(pos)
        words_no_num = []
        for i in range(0,len(words)):
            if(words[i].isdigit()):
                continue

            words_no_num.append(words[i])
            if(words[i].lower() == 'subject'):
                subject = words[i+1:]
                break
            if (i == len(words)-1):
                body = body + words_no_num

    # print(subject)
    # print(body)

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
                    classes.append('0')
                else:
                    classes.append('1')
                email = readtxt('train_data/' + directory + '/' + filename)
                # print(filename)
                subject, body = get_sections(email)
                data.append(' '.join(body))

    return data, classes

def get_email_bodies_test():
    data = []
    for i in range(1, 801):
        email = readtxt('test_data/test_email_' + str(i) + '.txt')
        # print(filename)
        subject, body = get_sections(email)

        data.append(' '.join(body))

    return data

def create_results_csv(preds):
    df = pd.DataFrame()
    df['email_id'] = [i for i in range(1,len(preds) + 1)]
    df['labels'] = preds

    df.to_csv('preds.csv',index=False)


def evaluate_on_test_set(test, classifier, tf):
    test_input = tf.transform(test['body'])
    # test_target = test['class'].values

    preds = classifier.predict(test_input.toarray())
    create_results_csv(preds)
    # return classifier.score(test_input, test_target)
