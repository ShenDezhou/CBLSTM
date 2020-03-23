#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2019年5月17日

@author: Administrator
'''
from sklearn.feature_extraction.text import CountVectorizer
import os
import codecs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pkuseg

class Sentiment(object):
    vectorizer=None
    log_model=None
    acc_score=None
        
    def __init__(self):
        pass
    
    @classmethod
    def load_model(cls_obj):
        
        data = []
        data_labels = []
        for filename in os.listdir(u"./hotelcomment/正面"):
            if filename.endswith(".txt"):
                with codecs.open("./hotelcomment/正面/"+filename, 'r', encoding='utf-8') as f:
                    text = f.read()
                    data.append(text)
                    data_labels.append('pos')
                continue
            else:
                continue
        
        for filename in os.listdir(u"./hotelcomment/负面"):
            if filename.endswith(".txt"):
                with codecs.open(u"./hotelcomment/负面/"+filename, 'r', encoding='utf-8') as f:
                    text = f.read()
                    data.append(text)
                    data_labels.append('neg')
                continue
            else:
                continue
        
        print(len(data), len(data_labels))
        seg = pkuseg.pkuseg(model_name='web')   
        
        cls_obj.vectorizer = CountVectorizer(
            analyzer = lambda text: seg.cut(text),
            lowercase = False,
        )
        features = cls_obj.vectorizer.fit_transform(
            data
        )
        features_nd = features.toarray()
        
        X_train, X_test, y_train, y_test  = train_test_split(
                features_nd, 
                data_labels,
                train_size=0.80, 
                random_state=1234)
        
        cls_obj.log_model = LogisticRegression()
        cls_obj.log_model = cls_obj.log_model.fit(X=X_train, y=y_train)
        y_pred = cls_obj.log_model.predict(X_test)
        cls_obj.acc_score=accuracy_score(y_test, y_pred)
        return cls_obj