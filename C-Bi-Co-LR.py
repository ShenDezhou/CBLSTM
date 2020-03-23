from sklearn.feature_extraction.text import CountVectorizer
import os
import codecs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pkuseg
import numpy
import pickle
from sklearn.model_selection import GroupShuffleSplit
from sklearn_crfsuite import metrics
from sklearn.multiclass import OneVsRestClassifier

# #              precision    recall  f1-score   support
#
#            0     0.8137    0.8137    0.8137      1197
#            1     0.8137    0.8137    0.8137      1197
#
#     accuracy                         0.8137      2394
#    macro avg     0.8137    0.8137    0.8137      2394
# weighted avg     0.8137    0.8137    0.8137      2394

rydict = dict(zip(list("PN"), range(len("PN"))))

def getFeatures(cvector, line):
    return cvector.transform([line]).toarray().flatten()


#808 976
maxlen=976
maxdim=5000


MODE = 1

if MODE==1:
    STATES = list("PN")
    seg = pkuseg.pkuseg(model_name='web')
    with codecs.open('plain/ctrip_hotel_pos.utf8', 'r', encoding='utf8') as fp:
        with codecs.open('plain/ctrip_hotel_neg.utf8', 'r', encoding='utf8') as fn:
            with codecs.open('plain/ctrip_hotel_pos_group.utf8', 'r', encoding='utf8') as fpg:
                with codecs.open('plain/ctrip_hotel_neg_group.utf8', 'r', encoding='utf8') as fng:
                    with codecs.open('model/ctrip_hotel_CRF-Bi-Co-LR_split.utf8', 'w', encoding='utf8') as fs:
                        pxlines = fp.readlines()
                        nxlines = fn.readlines()
                        pylines = ['P']*len(pxlines)
                        nylines = ['N']*len(nxlines)

                        pxlines.extend(nxlines)
                        pylines.extend(nylines)

                        pglines = fpg.readlines()[0].strip().split('\t')
                        nglines = fng.readlines()[0].strip().split('\t')
                        pglines.extend(nglines)

                        gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
                        for train, test in gss.split(pxlines, pylines, groups=pglines):
                            print(train,test)
                            for t in train:
                                fs.write(str(t)+'\t')
                            fs.write('\n')
                            for t in test:
                                fs.write(str(t)+'\t')
                            fs.write('\n')
                            pxtrain = numpy.array(pxlines)[train]
                            pytrain = numpy.array(pylines)[train]
                            pxtest = numpy.array(pxlines)[test]
                            pytest = numpy.array(pylines)[test]

                            cvector = CountVectorizer(
                                analyzer='char',
                                lowercase=False,
                                ngram_range=(2,2),
                                max_features =None,
                            )

                            cvector.fit(pxtrain)
                            X = []
                            y = []

                            print('process X list.')
                            counter = 0

                            for line in pxtrain:
                                line = line.replace(" ", "").strip()
                                line = '\n' * (maxlen - len(line)) + line
                                assert len(line) == maxlen
                                X.append(getFeatures(cvector, line))
                                # X.append([rxdict.get(e, 0) for e in list(line)])
                                # break
                                counter += 1
                                if counter % 1000 == 0 and counter != 0:
                                    print('.')

                            X = numpy.array(X)
                            print(len(X), X.shape)


                            print('process y list.')
                            for line in pytrain:
                                sline = numpy.zeros((len("PN")), dtype=int)
                                sline[rydict[line]] = 1
                                y.append(sline)
                            print(len(y))
                            y = numpy.array(y)
                            print(len(y), y.shape)


                            print('validate size.')
                            assert len(X) == len(y)
                            # for i in range(len(X)):
                            #     assert len(X[i]) == len(y[i])

                            model = OneVsRestClassifier(LogisticRegression())
                            model.fit(X,y)

                            X = []
                            y = []

                            print('process X list.')
                            counter = 0
                            for line in pxtest:
                                line = line.replace(" ", "").strip()
                                line = '\n' * (maxlen - len(line)) + line
                                assert len(line) == maxlen
                                X.append(getFeatures(cvector, line))
                                # X.append([rxdict.get(e, 0) for e in list(line)])
                                # break
                                counter += 1
                                if counter % 1000 == 0 and counter != 0:
                                    print('.')

                            X = numpy.array(X)
                            print(len(X), X.shape)

                            print('process y list.')
                            for line in pytest:
                                sline = numpy.zeros((len("PN")), dtype=int)
                                sline[rydict[line]] = 1
                                y.append(sline)
                            print(len(y))
                            y = numpy.array(y)
                            print(len(y), y.shape)

                            yp = model.predict(X)
                            print(len(yp), yp.shape)


                            # for i in range(len(y)):
                            assert len(yp) == len(y)
                            m = metrics.flat_classification_report(
                                y, yp, digits=4
                            )
                            print(m)
                            fs.write(m+"\n")

