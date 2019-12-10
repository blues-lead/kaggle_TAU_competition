# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 01:01:15 2019

@author: Anton
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import os
from sklearn.metrics import accuracy_score

def main():
    dir1 = 'D:\\docs\\kaggle_sgn_competition\\vehicle\\train\\train'
    dir2 = 'C:\\Users\\Anton\\Google Drive\\vehicle\\train\\train'
    os.chdir(dir2)
    X = np.load('feature_matrix.npy')
    y = np.load('gt_vector.npy')
    y = y.ravel()
    #X = np.array(X[:1000,:])
    #y = np.array(y[:1000])
    #model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    #model.fit(X_train, y_train)
    #pred = model.predict(X_test)
    #print(accuracy_score(y_test,pred))
#==============================================================================
    clfs = {}
    clfs['kneighbor'] = {'alg':KNeighborsClassifier() ,'score':0, 'stdev':0}
    clfs['svm'] = {'alg':SVC(), 'score':0, 'stdev':0}
    clfs['logreg'] = {'alg':LogisticRegression(), 'score':0, 'stdev':0}
    clfs['lda'] = {'alg':LinearDiscriminantAnalysis(),'score':0, 'stdev':0}
    clfs['random_forest'] = {'alg':RandomForestClassifier(),'score':0,'stdev':0}
    clfs['extra_forest'] = {'alg':ExtraTreesClassifier(),'score':0,'stdev':0}
    clfs['ada_boost'] = {'alg':AdaBoostClassifier(),'score':0,'stdev':0}
    clfs['gradient_boost'] = {'alg':GradientBoostingClassifier(),'score':0, 'stdev':0}
#==============================================================================   
    for key in clfs:
        print(key,"is processing")
        clfs[key]['alg'].fit(X_train,y_train)
        #scores_vector = cross_val_score(clfs[key]['alg'],F,y)
        pred = clfs[key]['alg'].predict(X_test)
        #clfs[key]['score'] = scores_vector.mean()
        clfs[key]['score'] = accuracy_score(y_test,pred)
        clfs[key]['stdev'] = 0
    print()
    print("{:^20s}{:^20s}{:^20}".format("Algorithm","Scores","+/- deviation"))    
    print("="*60)
    for key in clfs:
        print("{:^20s}{:^20.2f}{:^20.2f}".format(key,clfs[key]['score'],clfs[key]['stdev']))
    print("="*60)    
    
    
main()