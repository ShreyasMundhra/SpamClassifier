import pandas as pd
import os
from nltk.tokenize import RegexpTokenizer

# read txt file
def readtxt(file):
    with open(file,'r+') as file_obj:
        return file_obj.read()

# main function to generate features
def featurize():
    return

# separate email into sections e.g. sender, receiver, subject, body etc.
def get_sections(email):
    tokenizer = RegexpTokenizer(r'\w+')
    lines = email.split("\n")
    subject = []
    body = []

    for line in lines:
        # words = line.split()
        words = tokenizer.tokenize(line)
        for i in range(0,len(words)):
            if(words[i].lower() == 'subject'):
                subject = words[i+1:]
                break
            if (i == len(words)-1):
                body = body + words

    print subject
    print body

    return subject, body

def writeToCsv(dict):
    with open('words.csv','wb') as file_object:
        for key in dict:
            file_object.write(key + ',' + str(dict[key]) + '\n')


# create dictionary with the word as key and count as value for entire dataset to find important features for the model
def get_word_dict():
    word_count = {}
    for directory in os.listdir('train_data'):
        for filename in os.listdir('train_data/' + directory):
            email = readtxt('train_data/' + directory + '/' + filename)
            subject, body = get_sections(email)

            for word in body:
                if(word in word_count):
                    word_count[word] = word_count[word] + 1
                else:
                    word_count[word] = 0
    return word_count

# readtxt('Summary.txt')
word_dict = get_word_dict()
writeToCsv(word_dict)