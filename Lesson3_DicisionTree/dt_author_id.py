#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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

print "the number of feature :", len(features_train[0])

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)

t0 = time()
clf = clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "predition time:", round(time()-t1, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)

print "accuracy :", accuracy

#########################################################

# training time: 89.443 s
# predition time: 0.039 s
# accuracy : 0.977246871445

# change ../tools/email_preprocess.py
# selector = SelectPercentile(f_classif, percentile=10)
# -> selector = SelectPercentile(f_classif, percentile=1)
#
# training time: 5.55 s
# predition time: 0.003 s
# accuracy : 0.966439135381
#
# A large value for percentile lead to a more complex.



