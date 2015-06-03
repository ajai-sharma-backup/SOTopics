import numpy as np
import csv
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import random

""" This script will create a statistical model which predicts the topic(s) of a 
stack overflow post.

For example, if someone writes a post on stack overflow asking for help on a 
particular topic, then using the model outputted by this program, one could 
potentially predict the topic of the post based on the text of the topic alone.

This program takes in a training set and creates a feature matrix of word 
frequencies and power features based on each stack overflow post. A Random 
Forests algorithm is then run on the matrix to create a model which can 
predict the topic of a post from new data. 

Note that this is a mulitlabel problem as opposed to a multiclass problem since
each post can potentially have more than one topic assigned to it.

Todo: Get rid of the for-loops. Replace with list comprehensions where possible.
"""


# A couple of utility functions to help with data cleaning.
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



"""
Part 1: Process the data and create feature matrix.
"""

iden = [] # ID numbers of posts
title = [] # Titles of posts
body = [] # html Bodies of posts
tags = [] # labels o
with open("train.csv",'rb') as f:
    reader = csv.reader(f)
    reader.next()
    for line in reader:
        iden.append(line[0])
        title.append(line[1])
        body.append(line[2])
        tags.append(line[3])

# Read in a list of common words to filter out.
commonWords = []
with open("common-english-words.txt", 'rb') as f:
    reader = csv.reader(f)
    for word in reader:
        commonWords = word
commonWords = commonWords + ['im','youre','youll','st','nd','rd','th']

# Process the html bodies of each post (removing html tags, etc.)    
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
    
# Process the titles of each post.
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

# Count the words in each post.
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

# Count the words in each title.
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

# Creates a set of words in all the posts.
uniqueWords = set()
for wc in mergedWordCounts:
    for k in wc.keys():
        uniqueWords.add(k)    

# Returns 1 if label s is in a set of tags x
def getTag(s,x):
    if s in x.split():
        return 1
    else:
        return 0

# labels each post 1 if it is about R, 0 else.
rLabels = map(lambda x: getTag("r", x), tags)
statLabels = map(lambda x: getTag("statistics", x), tags)
mlLabels = map(lambda x: getTag("machine-learning", x), tags)
mathLabels = map(lambda x: getTag("math", x), tags)
numpyLabels = map(lambda x: getTag("numpy", x), tags)

# Make this into a numpy array
multilabels = np.asarray((rLabels, statLabels, mlLabels, mathLabels, numpyLabels))
multilabels = np.transpose(multilabels)

# total number of times each word appears in all the posts
totalWordCounts = {}
for i in range(len(mergedWordCounts)):
    a = mergedWordCounts[i]
    for word in a:
        totalWordCounts[word] = a[word] + totalWordCounts.get(word, 0)

# keep only the words that appear in more than 10 posts.
keep = []
for word in totalWordCounts:
    if ((totalWordCounts[word] != 1) and (totalWordCounts[word] <= 10)):
        keep.append(word)
        
np.save("keep.npy",keep)

""" Power feature extraction: 
We extract meta features, e.g. how long the title is, how many non-alphabetical characters there are,
how many words in the body of a post, etc.
"""

pf1titleLengths = map(lambda x: len(x.split()), title)
pf2codeCounts = map(lambda x: x.count("<code>"), body)
pf3wordsPerPost = map(lambda x: len(x), bodySplit)
pf4numLines = map(lambda x: x.count("\n"), body)
pf5numEqns = map(lambda x: x.count("$"), body)
pf6numParen = map(lambda x: x.count("("), body)
pf7numBracket = map(lambda x: x.count("["), body)
pf8numColon = map(lambda x: x.count(":"), body)
pf9numAst = map(lambda x: x.count("*"), body)
pf10numPlus = map(lambda x: x.count("+"), body)

pfMatrix = np.array((pf1titleLengths, pf2codeCounts, pf3wordsPerPost,
pf4numLines, pf5numEqns, pf6numParen, pf7numBracket, pf8numColon, pf9numAst,
pf10numPlus))
pfMatrix = pfMatrix.transpose()

# Construct and fill the matrix.
X = np.zeros(shape = [len(mergedWordCounts),len(keep)], dtype = np.float16)
for i_word in range(len(keep)):
    for i_post in range(len(mergedWordCounts)):
        X[i_post][i_word] = mergedWordCounts[i_post].get(keep[i_word], 0)
                
xRow = np.sum(X, axis = 1)

X = X / xRow[:, np.newaxis]
X2 = np.nan_to_num(X)
X2cv = X2[0:1000,:]
    
# Fit the model. (Only used 50 trees when doing CV, otherwise, it took too long).
forest = RandomForestClassifier(n_estimators = 50)
# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(X,multilabels)
joblib.dump(forest, "MultilabelModel.pkl")    

""" The remaining data is for diagnostic purposes/model evaluation:
We perform CV, we look at histograms of word counts, create ROC curves,
find the AUC, confusion matrices, etc.

10-fold CV:
 Returns two lists. 
 List 1: The number of trees used to split.
 List 2: The 10-fold CV error corresponding to the number of trees used.
 Basically Random Forests says that sqrt(numberOfFeatures) is the ideal value
 We are testing this assumption.

 Code works like this:

 For each trees
    for each fold 1,..k..,10
        fit a model on all but the kth fold
        predict on kth fold
        find the error 
        take the average error on k folds
"""

def findCVError(Xmatrix, Ylabels):
    numberOfFeatures = []
    meanErrs = []
    folds = KFold(len(Ylabels), n_folds = 10)
    for nFeatures in [10, 120, 5000, 14483]:
        numberOfFeatures.append(nFeatures)
        errRates = []
        for train, test in folds:
            forest = RandomForestClassifier(max_features = nFeatures)
            forest = forest.fit(Xmatrix[train],Ylabels[train])
            Ytest = forest.predict(Xmatrix[test])
            numErrs = sum((Ytest - Ylabels[test])**2) + 0.0
            errRate = numErrs/len(Ylabels[test])
            errRates.append(errRate)
        meanErrs.append(sum(errRates)/len(errRates))
        print("Done with ")
        print(nFeatures)
    return((numberOfFeatures,meanErrs))
                        
############################################################
# Question 3: Make a histogram of the number of times word features 
# appear in a text
############################################################

postsPerWord = {}
for w in uniqueWords:
    for wl in mergedWordCounts:
        if w in wl:
            postsPerWord[w] = postsPerWord.get(w,0) + 1

# export the numbers to a csv and make a histogram in R

postCounts = []
for c in postsPerWord:
    postCounts.append(postsPerWord[c])

counts = open("counts.csv", 'wb')
countsWriter = csv.writer(counts)
for i in xrange(len(postCounts)):
    countsWriter.writerow([postCounts[i]])
                                    
# Question 6: Do a CV on max_features, number of features to 
# consider when looking for the best split.

# 6.1 CV on only the word features (See R file for plot)
rLabelsNP = np.array(rLabels[0:1000])
fullMatrixCV = findCVError(X2cv,rLabelsNP)
fullMatrixCV
# Here is the output:
# [(10, 120, 5000, 14483)]
# [(0.49899999999999994, 0.49499999999999994, 0.49899999999999994, 0.502)]


# 6.2a ROC Curve on Word Feature
rLabelsNP = np.array(rLabels)
wordModel = RandomForestClassifier(n_estimators = 6)
wordModel = wordModel.fit(X2, rLabelsNP)
yhatProb = wordModel.predict_proba(X2)
yhatprob1 = yhatProb[:,1]
fpr, tpr, thresholds = roc_curve(rLabelsNP, yhatprob1)
roc_auc = auc(fpr, tpr)
plt.title('ROC Curve Word Features')
plt.plot(fpr, tpr, 'b',
label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1], 'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()

# The accuracy of the word features is 
yhatwm = wordModel.predict(X2)
1 - (sum((yhatwm - rLabelsNP)**2 + 0.0)/len(rLabels))
# 0.78545123062898814

# The confusion matrix from only word features:
#array([[ 8633,  5524],
#       [  360, 12908]])


# The NPV and PPV of word features
confusion_matrix(rLabelsNP, yhatwm)
# PPV = 0.6098043370770644
# NPV = 0.9728670485378353


# 6.2b ROC Curve on Power Feature
rLabelsNP = np.array(rLabels)
pfModel = RandomForestClassifier(n_estimators = 6)
pfModel = pfModel.fit(pfMatrix, rLabelsNP)
yhatProb = pfModel.predict_proba(pfMatrix)
yhatprob1 = yhatProb[:,1]
fpr, tpr, thresholds = roc_curve(rLabelsNP, yhatprob1)
roc_auc = auc(fpr, tpr)
plt.title('ROC Curve Power Features')
plt.plot(fpr, tpr, 'b',
label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1], 'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()

# The accuracy of the power features is 
yhatpf = pfModel.predict(pfMatrix)
1 - (sum((yhatpf - rLabelsNP)**2 + 0.0)/len(rLabels))
# accuracy = 0.96521422060164086

# The confusion matrix from power features
#array([[13936,   221],
#       [  733, 12535]])

# The NPV and PPV of power features
confusion_matrix(rLabelsNP, yhatpf)
# PPV = 0.9843893480257117
# NPV = 0.9447542960506482

# 6.2c ROC Curve on Combined Features
pfWordMatrix = np.append(X2, pfMatrix, axis = 1)
pfWordModel = RandomForestClassifier(n_estimators = 6)
pfWordModel = pfWordModel.fit(pfWordMatrix, rLabelsNP)
print("Model fit")
yhatProb = pfWordModel.predict_proba(pfWordMatrix)
print("probs predicted")

yhatprob1 = yhatProb[:,1]
fpr, tpr, thresholds = roc_curve(rLabelsNP, yhatprob1)
roc_auc = auc(fpr, tpr)
plt.title('ROC Curve Word and Power Features')
plt.plot(fpr, tpr, 'b',
label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1], 'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()
# The accuracy of the word and power features is 
yhatwpf = pfWordModel.predict(pfWordMatrix)
1 - (sum((yhatwpf - rLabelsNP)**2 + 0.0)/len(rLabels))
# 0.97082953509571557
# The NPV and PPV of power features
confusion_matrix(rLabelsNP, yhatwpf)
#array([[14007,   150],
#       [  650, 12618]])
# PPV = 0.9894045348590803
# NPV = 0.9510099487488695

# 8. Multilabel classification using word features only.

# 8.1 Train model
wordMulti = RandomForestClassifier(n_estimators = 6)
wordMulti = wordMulti.fit(X2, multilabels)
yhatwmulti = wordMulti.predict(X2)

# 8.2 Find accuracy 
1 - (sum( (yhatwmulti - multilabels)**2) / (5*len(body)))
# The accuracy is 0.88062716499544214

# Class by class accuracy
1 - sum((yhatwmulti - multilabels)**2,axis = 0) / len(body)
# r: 0.72630811,  statistics: 0.90862352, machine-learning:  0.96984503,  
# math: 0.85969006, numpy:  0.9386691

# Cross Validated Accuracy (Note: Training set is first 25000 entries, rest are test set)
wordMulticv = RandomForestClassifier(n_estimators = 1)
wordMulticv = wordMulti.fit(X2[0:25000,:], multilabels[0:25000,:])
yhatwmulticv = wordMulti.predict(X2[25000:,:])
1 - (sum( (yhatwmulticv - multilabels[25000:,:])**2) / (5*len(body[25000:])))
# 0.82193814432989687

# Class by class CV accuracy
1 - sum((yhatwmulticv - multilabels[25000:,:])**2,axis = 0) / len(body[25000:])
# r: 0.59917526, statistics: 0.86103093, machine-learning: 0.94268041,  
# math: 0.79092784, numpy: 0.91587629

# 9.1 Multilabel classification using power features only.

# 9.1.1 Train model (via CV)
pfMulticv = RandomForestClassifier(n_estimators = 6)
pfMulticv = pfMulticv.fit(pfMatrix[0:25000,:], multilabels[0:25000,:])
yhatwmultipcv = pfMulticv.predict(pfMatrix[25000:,:])

# 9.1.2 Find CV accuracy
1 - (sum( (yhatwmultipcv - multilabels[25000:,:])**2) / (5*len(body[25000:])))
# 0.86705154639175253

# Class by class CV accuracy
1 - sum((yhatwmultipcv - multilabels[25000:,:])**2,axis = 0) / len(body[25000:])
# r: 0.76453608, statistics: 0.9014433 , machine-learning: 0.9414433 ,  
# math: 0.80824742, numpy:  0.91958763

# 9.2 Multilabel Classification using both word and power features
# 9.2.1 Train model
wpfMulticv = RandomForestClassifier(n_estimators = 6)
wpfMulticv = pfMulticv.fit(pfWordMatrix[0:25000,:], multilabels[0:25000,:])
yhatwmultipwcv = wpfMulticv.predict(pfWordMatrix[25000:,:])

# 9.2.2 Find CV Accuracy
1 - (sum( (yhatwmultipwcv - multilabels[25000:,:])**2) / (5*len(body[25000:])))
# 0.86861855670103094

# Class by class CV accuracy
1 - sum((yhatwmultipwcv - multilabels[25000:,:])**2,axis = 0) / len(body[25000:])
# r: 0.75340206, statistics: 0.90597938, machine-learning  0.94185567,  
# math: 0.81113402, numpy: 0.93072165



