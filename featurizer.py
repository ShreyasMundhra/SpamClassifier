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
                subject, body = get_sections(email)
                data.append(' '.join(body))

    return data, classes

def get_email_bodies_test():
    data = []
    for i in range(1, 801):
        email = readtxt('test_data/test_email_' + str(i) + '.txt')
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

def get_word_dict(stage):
    data = list()
    if(stage == 'train'):
        temp, classes = get_email_bodies()
        for t in temp:
            data.append(t)
    elif(stage == 'test'):
        temp = get_email_bodies_test()
        for t in temp:
            data.append(t)
    elif(stage == 'train_test'):
        temp1, classes = get_email_bodies()
        temp2 = get_email_bodies_test()
        for t in temp1+temp2:
            data.append(t)

    word_count = {}
    for row in data:
        words = row.split()
        for word in words:
            if (word in word_count.keys()):
                word_count[word] = word_count[word] + 1
            else:
                word_count[word] = 0

    return word_count

def writeDictToCsv(words_dict,filename):
    with open(filename + '.csv','w') as file_object:
        for key in words_dict.keys():
            try:
                file_object.write(key + ',' + str(words_dict[key]) + '\n')
            except UnicodeEncodeError:
                continue

if __name__ == "__main__":
    train_count = get_word_dict('train')
    test_count = get_word_dict('test')
    total_count = get_word_dict('train_test')

    writeDictToCsv(train_count,'train_count')
    writeDictToCsv(test_count,'test_count')
    writeDictToCsv(total_count,'train_test_count')
