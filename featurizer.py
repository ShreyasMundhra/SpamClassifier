import pandas as pd
import os

# read txt file
def readtxt(file):
    with open(file,'r+') as file_obj:
        return file_obj.read()

# main function to generate features
def featurize():
    return

# separate email into sections e.g. sender, receiver, subject, body etc.
def get_sections(email):
    lines = email.split("\n")
    subject = ""
    body = ""

    for line in lines:
        words = line.split()
        for i in range(0,len(words)):
            if(words[i].lower() == 'subject:'):
                subject = ' '.join(words[i+1:])
                break
            if (i == len(words)-1):
                body = body + line

    print subject
    print body

    return subject, body

# create dictionary with the word as key and count as value for entire dataset to find important features for the model
def get_word_dict():
    word_count = {}
    # print os.listdir('.')
    # for filename in os.listdir('train_data'):
    #     email = readtxt(filename)
    #     body = get_sections(email)
    email = readtxt('train_data/ham/0007.1999-12-14.farmer.ham.txt')
    get_sections(email)
    # print body

# readtxt('Summary.txt')
get_word_dict()