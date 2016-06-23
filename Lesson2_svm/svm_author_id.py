#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



#########################################################
### your code goes here ###

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

from sklearn import svm
#clf = svm.SVC(kernel='linear')

# rbf parameters
# http://qiita.com/sz_dr/items/f3d6630137b184156a67
clf = svm.SVC(kernel='rbf', C=10000)

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "predition time:", round(time()-t1, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)

print "accuracy :", accuracy

print "#10 :", pred[10]
print "#26 :", pred[26]
print "#50 :", pred[50]

counter = 0
for p in pred:
    if p == 1:
        counter += 1

print "Chris :", counter


#########################################################

# linear
# full data set:
# training time: 297.914 s
# predition time: 27.919 s
# 0.984072810011

# linear
# 1/100 data set
#training time: 0.14 s
#predition time: 1.271 s
#0.884527872582

# rbf C=10000, full data set
# raining time: 844.575 s
# predition time: 14.259 s
# 0.990898748578

# Finel
# training time: 366.889 s
# predition time: 14.458 s
# accuracy : 0.990898748578
# #10 : 1
# #26 : 0
# #50 : 1
# Chris : 877




