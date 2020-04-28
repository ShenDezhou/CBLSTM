import xgboost as xgb


import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Normalizer
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

#binary:hinge
#               precision    recall  f1-score   support
#
#            0     0.6157    0.6157    0.6157      1197
#            1     0.6157    0.6157    0.6157      1197
#
#     accuracy                         0.6157      2394
#    macro avg     0.6157    0.6157    0.6157      2394
# weighted avg     0.6157    0.6157    0.6157      2394

rydict = dict(zip(list("PN"), range(len("PN"))))

def getFeatures(cvector, line):
    return cvector.transform([line]).toarray().flatten()


#808 976
maxlen=976
maxdim=5000
modelfile = os.path.basename(__file__).split(".")[0]

MODE = 1

if MODE==1:
    STATES = list("PN")
    seg = pkuseg.pkuseg(model_name='web')
    with codecs.open('plain/ctrip_hotel_pos.utf8', 'r', encoding='utf8') as fp:
        with codecs.open('plain/ctrip_hotel_neg.utf8', 'r', encoding='utf8') as fn:
            with codecs.open('plain/ctrip_hotel_pos_group.utf8', 'r', encoding='utf8') as fpg:
                with codecs.open('plain/ctrip_hotel_neg_group.utf8', 'r', encoding='utf8') as fng:
                    with codecs.open('model/ctrip_hotel_%s_split.utf8'%modelfile, 'w', encoding='utf8') as fs:
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
                        index =0
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

                            cvector = TfidfVectorizer(
                                analyzer= 'char',
                                lowercase=False,
                                ngram_range=(1,2),
                                max_features =None,
                            )

                            cvector.fit(pxtrain)
                            joblib.dump(cvector, "lr/ctrip_tfidfvec%d.pkl.gz" % index, compress=('gzip', 3))

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
                            X = Normalizer(norm='l2').fit_transform(X)
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

                            trainM = xgb.DMatrix(data=X,label=y)

                            device='cpu'
                            if device=='gpu':
                                param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:hinge', 'gpu_id':0, 'tree_method':'gpu_hist'}
                            else:
                                param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:hinge', }
                            epochs = 2
                            model = xgb.train(param, dtrain=trainM, num_boost_round=epochs)
                            # xgb.plot_importance(model)
                            # xgb.plot_tree(model, num_trees=2)
                            # model = OneVsRestClassifier(LogisticRegression())
                            # model.fit(X,y)
                            joblib.dump(model,"gbdt/ctrip_comment%d.pkl.gz"%index, compress=('gzip',3))
                            index+=1
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
                            X = Normalizer(norm="l2").fit_transform(X)
                            print(len(X), X.shape)

                            print('process y list.')
                            for line in pytest:
                                sline = numpy.zeros((len("PN")), dtype=int)
                                sline[rydict[line]] = 1
                                y.append(sline)
                            print(len(y))
                            y = numpy.array(y)
                            print(len(y), y.shape)

                            testM = xgb.DMatrix(data=X)
                            yp = model.predict(testM)
                            print(len(yp), yp.shape)

                            ypl = []
                            for line in yp:
                                sline = numpy.zeros((len("PN")), dtype=int)
                                if line < 0.5:
                                    sline[0] = 1
                                else:
                                    sline[1] = 1
                                ypl.append(sline)
                            print(len(ypl))
                            yp = numpy.array(ypl)
                            print(len(yp), yp.shape)

                            # for i in range(len(y)):
                            assert len(yp) == len(y)
                            m = metrics.flat_classification_report(
                                y, yp, digits=4
                            )
                            print(m)
                            fs.write(m+"\n")

