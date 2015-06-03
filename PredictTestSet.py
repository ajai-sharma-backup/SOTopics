import numpy as np
import pandas as pd
import os
import csv
import string
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

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

keep = np.load("keep.npy")


iden = []
title = []
body = []

with open("XtestKaggle2.csv",'rb') as f:
    reader = csv.reader(f)
    reader.next()
    for line in reader:
        iden.append(line[0])
        title.append(line[1])
        body.append(line[2])

commonWords = []
with open("common-english-words.txt", 'rb') as f:
    reader = csv.reader(f)
    for word in reader:
        commonWords = word
commonWords = commonWords + ['im','youre','youll','st','nd','rd','th']
    
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


# 4.1 Number of words in a title (Note these are raw titles, unprocessed)
titleLengths = []
for t in title:
    titleLengths.append(len(t.split()))
        
# 4.2 Count the number of <code> blocks in a post.
codeCounts = []
for b in body:
    codeCounts.append(b.count("<code>"))

# creates the test matrix
Xtest = np.zeros(shape = [len(mergedWordCounts),len(keep) + 2], dtype = np.float16)
for i_word in range(len(keep)):
    for i_post in range(len(mergedWordCounts)):
        Xtest[i_post][i_word] = mergedWordCounts[i_post].get(keep[i_word], 0)
        
xTestRow = np.sum(Xtest, axis = 1)

Xtest = Xtest / xTestRow[:, np.newaxis]


# Add the power features
for i in xrange(len(titleLengths)):
    Xtest[i][-2] = titleLengths[i]
    Xtest[i][-1] = codeCounts[i]    

# Create the random forest object which will include all the parameters
# for the fit
forestTest = joblib.load("MultilabelModel.pkl")

# Fit the training data to the Survived labels and create the decision trees
Yhat = forestTest.predict(Xtest)

newCols = np.array([1,2,0,4,3])
YhatSubmit = Yhat[:,newCols]
ids = np.array(range(1,len(codeCounts)+1))[...,None]
YhatSubmit = np.append(ids, YhatSubmit, 1)
np.save("YhatMatrix", Yhat) # save matrix
np.savetxt("Yhatmulti.csv", YhatSubmit, delimiter = ",", fmt = "%d")



################################################################
Yactual = []
with open("Ytest1.csv",'rb') as f:
    reader = csv.reader(f)
    reader.next()
    for line in reader:
        Yactual.append(int(line[1]))

(500 - sum((Yhat - Yactual)**2))/500.0


Yhat = pd.DataFrame(data = zip(range(1, len(Xtest) + 1), Yhat), columns=['id', 'tag'])
Yhat.to_csv('Yhat1.csv',index=False,header=True)

np.savetxt("Xprocessed1.csv", Xtest, delimiter = ",")
