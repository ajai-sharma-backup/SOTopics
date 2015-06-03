import numpy as np
import pandas as pd
import os
import csv
import string
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

""" This program will create a statistical model which predicts the topic of a 
stack overflow post.

For example, if someone writes a post on stack overflow asking for help on a 
particular topic, then using the model outputted by this program, one could 
potentially predict the topic of the post based on the text of the topic alone.

This program takes in training data and creates a feature matrix of word 
frequencies and power features based on each stack overflow post. A Random 
Forests algorithm is then run on the matrix to create a model which can 
predict the topic of a post from new data. 

Note that this is a mulitlabel problem as opposed to a multiclass problem since
each post can potentially have more than one topic assigned to it.
"""



# Some functions to help in HTML Processing.
def StripInline(s, w1 = '<code', w2 = '</code>'):
    """ This function takes a string s as input, and removes all patterns that
    start with w1, and end with w2, but stays within a line (No /n in the middle)
    """
    return re.sub(w1 + '((?!' + w1 + ').)*?' + w2, '', s, 
                flags = re.MULTILINE | re.S)
                
def StripBlock(s, w1 = '<pre', w2 = '</pre>'):
    """ This function takes a string s as input, and removes all patterns that
    start with w1 and end with w2, and spans multiple lines
    """
    return re.sub('^' + w1 + '((?!' + w1 + ').)*' + w2 + '$', '', s, 
                flags = re.MULTILINE | re.S)


# Start extracting data from HTML
iden = [] # ID number of post.
title = [] # Title of the post
body = [] # Body of the post.
tags = [] # Topics of the post (for training the model).

# open and extract data.
with open("train.csv",'rb') as f:
    reader = csv.reader(f)
    reader.next()
    for line in reader:
        iden.append(line[0])
        title.append(line[1])
        body.append(line[2])
        tags.append(line[3])

# import a list of common words to filter out of our data.
commonWords = []
with open("common-english-words.txt", 'rb') as f:
    reader = csv.reader(f)
    for word in reader:
        commonWords = word
commonWords = commonWords + ['im','youre','youll','st','nd','rd','th']
    
# remove symbols and html tags ("clean the data").
bodySplit = []
for post in body:
    temp = StripInline(post,w1='\'',w2='\'')
    temp = StripInline(temp,w1='<a',w2='</a>')
    temp = StripInline(temp,w1='\$\$',w2='\$\$')
    temp = StripBlock(temp,w1='<code>',w2='</code>')
    temp = StripInline(temp,w1='<code>',w2='</code>')
    temp = StripInline(temp,w1='<pre>',w2='</pre>')
    temp = temp.replace('<p>',' ').replace('</p>',' ').replace('\n',' ').replace('<br>',' ').replace('</br>',' ').replace('<strong>',' ').replace('</strong>', ' ').replace('(', ' ').replace(')', ' ')
    temp = re.sub(r"[0-9!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]",' ',temp)
    temp = temp.lower().split()
    tempSub = []
    for word in temp:
        if word not in commonWords:
            tempSub.append(word)
    bodySplit.append(tempSub)
    
# This splits and processes the words for the titles
titleSplit = []
for post in title:
    temp = StripInline(StripInline(post,w1='\$',w2='\$'),w1='\'',w2='\'')
    temp = re.sub(r"[0-9!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]",' ',temp)
    temp = temp.lower().split()
    tempSub = []
    for word in temp:
        if word not in commonWords:
            tempSub.append(word)
    titleSplit.append(tempSub)

# Creates a list of dictionaries of word counts for each body.
allBodyWordCounts = []
for wordList in bodySplit:
    wordCounts = {}
    wordSet = set(wordList)
    for w in wordSet:
        wordCounts[w] = 0;
        for v in wordList:
            if w == v:
                wordCounts[w] = wordCounts[w] + 1
    allBodyWordCounts.append(wordCounts)

# Creates a list of dictionaries of word counts for each title.
allTitleWordCounts = []
for wordList in titleSplit:
    wordCounts = {}
    wordSet = set(wordList)
    for w in wordSet:
        wordCounts[w] = 0;
        for v in wordList:
            if w == v:
                wordCounts[w] = wordCounts[w] + 1
    allTitleWordCounts.append(wordCounts)

# Merge the title word counts with the body word counts.
mergedWordCounts = []
for i in range(len(allTitleWordCounts)):
    a = allTitleWordCounts[i]
    b = allBodyWordCounts[i]
    c = b.copy()
    for elem in a:
        c[elem] = b.get(elem,0) + a[elem]
    mergedWordCounts.append(c)

# Creates a set of unique words.
uniqueWords = set()
for wc in mergedWordCounts:
    for k in wc.keys():
        uniqueWords.add(k)    

# labels each post 1 if it is about R, 0 else.
labels = []
for tag in tags:
    temp = tag.split()
    if 'r' in temp:
        labels.append(1)
    else:
        labels.append(0)

# total number of times each word appears in all the posts
totalWordCounts = {}
for i in range(len(mergedWordCounts)):
    a = mergedWordCounts[i]
    for word in a:
        totalWordCounts[word] = a[word] + totalWordCounts.get(word, 0)

# keep only the words that appear in more than 10 posts.
keep = []
for word in totalWordCounts:
    if totalWordCounts[word] != 1:
        keep.append(word)
        
np.save("keep.npy",keep)

# create the feature matrix.
X = np.zeros(shape = [len(mergedWordCounts),len(keep)], dtype = np.float16)
for i_word in range(len(keep)):
    for i_post in range(len(mergedWordCounts)):
        X[i_post][i_word] = mergedWordCounts[i_post].get(keep[i_word], 0)
        
xRow = np.sum(X, axis = 1)

X = X / xRow[:, np.newaxis]

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier()

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(X,labels)
joblib.dump(forest, "RFModel.pkl")
