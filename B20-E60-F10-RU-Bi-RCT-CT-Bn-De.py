# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
import codecs
import os
import re
import string
import pickle

import numpy
from keras import regularizers
from keras.layers import Dense, Embedding, RepeatVector, add, Add, SpatialDropout1D, LSTM, Input, Bidirectional, Flatten,CuDNNLSTM, Lambda, Dropout, BatchNormalization, Maximum, concatenate,Reshape, Concatenate
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adagrad
from sklearn_crfsuite import metrics
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras.initializers import Constant
from keras.backend import repeat_elements


#               precision    recall  f1-score   support
#
#            P     0.7629    0.6393    0.6957       463
#            N     0.7936    0.8747    0.8321       734
#
#     accuracy                         0.7836      1197
#    macro avg     0.7782    0.7570    0.7639      1197
# weighted avg     0.7817    0.7836    0.7793      1197

dicts = []
unidicts = []
predicts = []
sufdicts = []
longdicts = []
puncdicts = []
digitsdicts = []
chidigitsdicts = []
letterdicts = []
otherdicts = []

Thresholds = 0.95


def getTopN(dictlist):
    adict = {}
    for w in dictlist:
        adict[w] = adict.get(w, 0) + 1
    topN = max(adict.values())
    alist = [k for k, v in adict.items() if v >= topN * Thresholds]
    return alist


with codecs.open('ctrip_dic/pku_training_words.utf8', 'r', encoding='utf8') as fa:
    with codecs.open('ctrip_dic/pku_test_words.utf8', 'r', encoding='utf8') as fb:
        with codecs.open('ctrip_dic/contract_words.utf8', 'r', encoding='utf8') as fc:
            lines = fa.readlines()
            lines.extend(fb.readlines())
            lines.extend(fc.readlines())
            lines = [line.strip() for line in lines]
            dicts.extend(lines)
            # uni, pre, suf, long 这四个判断应该依赖外部词典，置信区间为95%，目前没有外部词典，就先用训练集词典来替代
            unidicts.extend([line for line in lines if len(line) == 1 and re.search(u'[\u4e00-\u9fff]', line)])
            predicts.extend([line[0] for line in lines if len(line) > 1 and re.search(u'[\u4e00-\u9fff]', line)])
            predicts = getTopN(predicts)
            sufdicts.extend([line[-1] for line in lines if len(line) > 1 and re.search(u'[\u4e00-\u9fff]', line)])
            sufdicts = getTopN(sufdicts)
            longdicts.extend([line for line in lines if len(line) > 3 and re.search(u'[\u4e00-\u9fff]', line)])
            puncdicts.extend(string.punctuation)
            puncdicts.extend(list("！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰–‘’‛“”„‟…‧﹏"))
            digitsdicts.extend(string.digits)
            chidigitsdicts.extend(list("零一二三四五六七八九十百千万亿兆〇零壹贰叁肆伍陆柒捌玖拾佰仟萬億兆"))
            letterdicts.extend(string.ascii_letters)

            somedicts = []
            somedicts.extend(unidicts)
            somedicts.extend(predicts)
            somedicts.extend(sufdicts)
            somedicts.extend(longdicts)
            somedicts.extend(puncdicts)
            somedicts.extend(digitsdicts)
            somedicts.extend(chidigitsdicts)
            somedicts.extend(letterdicts)
            otherdicts.extend(set(dicts) - set(somedicts))

chars = []
with codecs.open('ctrip_dic/ctrip_dict.utf8', 'r', encoding='utf8') as f:
    lines = f.readlines()
    for line in lines:
        for w in line:
            if w == '\n':
                continue
            else:
                chars.append(w)
print(len(chars))

bigrams = []
with codecs.open('ctrip_dic/ctrip_bigram.utf8', 'r', encoding='utf8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if len(line) > 0:
            bigrams.append(line)
print(len(bigrams))

rxdict = dict(zip(chars, range(1, 1 + len(chars))))
rxdict['\n'] = 0

rbxdict = dict(zip(bigrams, range(1, 1+len(bigrams))))
rbxdict['\n'] = 0

rydict = dict(zip(list("PN"), range(len("PN"))))


def getNgram(sentence, i):
    ngrams = []
    ch = sentence[i]
    ngrams.append(rxdict[ch])
    return ngrams


def getFeaturesDict(sentence, i):
    features = []
    features.extend(getNgram(sentence, i))
    assert len(features) == 1
    # featuresdic = dict([(str(j), features[j]) for j in range(len(features))])
    # return featuresdic
    return features

def getCharType(ch):
    dictofdicts = [puncdicts, digitsdicts, chidigitsdicts, letterdicts, unidicts, predicts, sufdicts]
    extradicts = [longdicts, otherdicts]
    if ch == '\n':
        return str(len(dictofdicts) + len(extradicts) - 1)

    types = []

    for i in range(len(dictofdicts)):
        if ch in dictofdicts[i]:
            types.append(i)
            break

    for i in range(len(extradicts)):
        for word in extradicts[i]:
            if ch in word:
                types.append(i + len(dictofdicts))
                break
        if len(types) > 0:
            break

    if len(types) == 0:
        return str(len(dictofdicts) + len(extradicts) - 1)

    assert len(types) == 1 or len(types) == 2, "{} {} {}".format(ch, len(types), types)

    return str(types[0])



def safea(sentence, i):
    if i < 0:
        return '\n'
    if i >= len(sentence):
        return '\n'
    return sentence[i]


def getBigram(sentence, i):
    #5 + 4*2 + 2*3=19
    ngrams = []
    for offset in [0, 1, 2]:
        ngrams.append(safea(sentence, i + offset))
    return ngrams


def getBigramVector(sentence, i):
    ngrams = getBigram(sentence, i)
    ngramv = []
    for ngram in ngrams:
        for ch in ngram:
            ngramv.append(rxdict.get(ch,0))
    return ngramv

def getUBgram(sentence, i):
    #3 + 2 = 5
    ngrams = []
    for offset in [0, 1, 2]:
        ngrams.append(safea(sentence, i + offset))

    for offset in [0, 1]:
        ngrams.append(safea(sentence, i + offset) + safea(sentence, i + offset + 1))

    # for offset in [0]:
    #     ngrams.append(safea(sentence, i + offset) + safea(sentence, i + offset + 1) + safea(sentence, i + offset + 2))
    return ngrams

def getUBgramVector(sentence, i):
    ngrams = getUBgram(sentence, i)
    ngramv = []
    for ngram in ngrams:
        if len(ngram)==1:
            ngramv.append(rxdict.get(ngram,0))
        if len(ngram)==2:
            if '\n' in ngram:
                ngramv.append(0)
            else:
                ngramv.append(rbxdict.get(ngram, 0))
    return ngramv


def getNgram(sentence, i):
    #5 + 4*2 + 2*3=19
    ngrams = []
    for offset in [-2, -1, 0, 1, 2]:
        ngrams.append(safea(sentence, i + offset))

    for offset in [-2, -1, 0, 1]:
        ngrams.append(safea(sentence, i + offset) + safea(sentence, i + offset + 1))

    for offset in [-1, 0]:
        ngrams.append(safea(sentence, i + offset) + safea(sentence, i + offset + 1) + safea(sentence, i + offset + 2))
    return ngrams


def getNgramVector(sentence, i):
    ngrams = getNgram(sentence, i)
    ngramv = []
    for ngram in ngrams:
        for word in ngram:
            for ch in word:
                ngramv.append(rxdict.get(ch,0))
    return ngramv


def getReduplication(sentence, i):
    reduplication = []
    for offset in [-2, -1]:
        if safea(sentence, i) == safea(sentence, i + offset):
            reduplication.append('1')
        else:
            reduplication.append('0')
    return reduplication

def getReduplicationVector(sentence, i):
    reduplicationv =[int(e) for e in getReduplication(sentence,i)]
    return reduplicationv

def getType(sentence, i):
    types = []
    for offset in [-1, 0, 1]:
        types.append(getCharType(safea(sentence, i + offset)))
    # types.append(getCharType(safea(sentence, i + offset - 1)) + getCharType(safea(sentence, i + offset)) + getCharType(
    #         safea(sentence, i + offset + 1)))
    return types

def getTypeVector(sentence, i):
    types = getType(sentence,i)
    types = [int(t) for t in types]
    return types

def getFeatures(sentence, i):
    #3+2+2+3
    features = []
    features.extend(getUBgramVector(sentence, i))
    features.extend(getReduplicationVector(sentence, i))
    features.extend(getTypeVector(sentence, i))
    assert len(features) == 10, (len(features),features)
    return features


def getFeaturesDict(sentence, i):
    features = []
    features.extend(getNgramVector(sentence, i))
    features.extend(getReduplicationVector(sentence, i))
    features.extend(getType(sentence, i))
    assert len(features) == 24
    featuresdic = dict([(str(j), features[j]) for j in range(len(features))])
    return featuresdic


batch_size = 20
maxlen = 976
nFeatures = 3+2+2+3
word_size = 100
redup_size = 2
type_size = 9
Hidden = 150
Regularization = 1e-4
Dropoutrate = 0.2
learningrate = 0.2
Marginlossdiscount = 0.2
nState = 2
EPOCHS = 60
modelfile = os.path.basename(__file__).split(".")[0]



MODE = 3

if MODE == 1:
    with codecs.open('plain/ctrip_training.utf8', 'r', encoding='utf8') as ft:
        with codecs.open('plain/ctrip_training_states.utf8', 'r', encoding='utf8') as fs:
            with codecs.open('model/f10/ctrip_train_crffeatures.pkl', 'wb') as fx:
                with codecs.open('model/f10/ctrip_train_crfstates.pkl', 'wb') as fy:
                    xlines = ft.readlines()
                    ylines = fs.readlines()
                    X = []
                    y = []

                    print('process X list.')
                    counter = 0
                    for line in xlines:
                        line = line.replace(" ", "").strip()
                        line = '\n' * (maxlen - len(line)) + line
                        assert len(line) == maxlen
                        X.append([getFeatures(line, i) for i in range(len(line))])
                        # X.append([rxdict.get(e, 0) for e in list(line)])
                        # break
                        counter += 1
                        if counter % 1000 == 0 and counter != 0:
                            print('.')

                    X = numpy.array(X)
                    print(len(X), X.shape)
                    # X = pad_sequences(X, maxlen=maxlen, padding='pre', value=[0]*nFeatures)
                    # print(len(X), X.shape)

                    print('process y list.')
                    assert len(ylines) == 1
                    for line in ylines:
                        line = line.strip()
                        # line = 'P' * (maxlen - len(line)) + line
                        line = [rydict[s] for s in line]
                        sline = numpy.zeros((len(line), len("PN")), dtype=int)
                        for g in range(len(line)):
                            sline[g, line[g]] = 1
                        y = sline
                        # break
                    #print(len(y))
                    # y = pad_sequences(y, maxlen=maxlen, padding='pre', value=3)
                    #y = numpy.array(y)
                    print(len(y), y.shape)

                    print('validate size.')
                    # for i in range(len(X)):
                    assert len(X) == len(y)

                    print('output to file.')
                    sX = pickle.dumps(X)
                    fx.write(sX)
                    sy = pickle.dumps(y)
                    fy.write(sy)

if MODE==2:
    loss = "squared_hinge"
    optimizer = "nadam"#Adagrad(lr=0.2) # "adagrad"
    metric= "accuracy"
    sequence = Input(shape=(maxlen,nFeatures,))
    seqsa, seqsb, seqsc, seqsd, seqse = Lambda(lambda x: [x[:,:,0],x[:,:,1],x[:,:,2],x[:,:,3],x[:,:,4]])(sequence)
    denseinputs = Lambda(lambda x: [x[:,:,5],x[:,:,6],x[:,:,7],x[:,:,8],x[:,:,9]])(sequence)
    assert len(denseinputs) == 5

    # zhwiki_emb = numpy.load("ctrip_dic/zhwiki_embedding.npy")
    embeddeda = Embedding(len(chars) + 1, word_size, input_length=maxlen, mask_zero=False)(seqsa)
    embeddedb = Embedding(len(chars) + 1, word_size, input_length=maxlen, mask_zero=False)(seqsb)
    embeddedc = Embedding(len(chars) + 1, word_size, input_length=maxlen, mask_zero=False)(seqsc)

    # zhwiki_biemb = numpy.load("ctrip/zhwiki_biembedding.npy")
    embeddedd = Embedding(len(bigrams) + 1, word_size, input_length=maxlen,
                          mask_zero=False)(seqsd)
    embeddede = Embedding(len(bigrams) + 1, word_size, input_length=maxlen,
                          mask_zero=False)(seqse)

    embeddeds = [Embedding(redup_size, word_size, embeddings_regularizer=regularizers.l2(Regularization),  input_length=maxlen, mask_zero=False)(i) for i in denseinputs[0:2]]
    embeddedt = [Embedding(type_size, word_size, embeddings_regularizer=regularizers.l2(Regularization), input_length=maxlen, mask_zero=False)(i) for i in denseinputs[2:]]

    maximuma = Maximum()([embeddeda, embeddedb])
    maximumb = Maximum()([embeddedc, embeddedb])

    sumbigram = concatenate([embeddeda, maximuma, maximumb, embeddedd, embeddede])
    # bnBigram = BatchNormalization()(sumbigram)
    sumduplication = concatenate(embeddeds)
    # bnType = BatchNormalization()(sumtypes)
    sumtype = concatenate(embeddedt)

    concat = concatenate([sumbigram, sumduplication, sumtype])


    dropout = SpatialDropout1D(rate=Dropoutrate)(concat)
    blstm = Bidirectional(CuDNNLSTM(Hidden, batch_input_shape=(maxlen, nFeatures), return_sequences=False), merge_mode='sum')(dropout)
    # dropout = Dropout(rate=Dropoutrate)(blstm)
    batchNorm = BatchNormalization()(blstm)
    dense = Dense(nState, activation='softmax', kernel_regularizer=regularizers.l2(Regularization))(batchNorm)

    model = Model(input=sequence, output=dense)
    # model.compile(loss='categorical_crossentropy', optimizer=adagrad, metrics=["accuracy"])
    # optimizer = Adagrad(lr=learningrate)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    model.summary()

    # model_json = model.to_json()
    # with open("keras/pretrained-extradim-dropout-bilstm-bn-arch.json", "w") as json_file:
    #     json_file.write(model_json)
    # model.save_weights("keras/pretrained-extradim-dropout-bilstm-bn-weights.h5")
    # #model.save("keras/pretrained-extradim-dropout-bilstm-bn.h5")
    # print("INI")

    with codecs.open('model/f10/ctrip_train_crffeatures.pkl', 'rb') as fx:
        with codecs.open('model/f10/ctrip_train_crfstates.pkl', 'rb') as fy:
            with codecs.open('model/ctrip_train_%s_model.pkl'%modelfile, 'wb') as fm:
                bx = fx.read()
                by = fy.read()
                X = pickle.loads(bx)
                y = pickle.loads(by)
                print(X[-1])
                print(y[-1])
                # for i in range(len(X)):
                assert len(X) == len(y)
                print('training')

                history = model.fit(X, y, batch_size=batch_size, nb_epoch=EPOCHS, verbose=1)

                print('trained')
                sm = pickle.dumps(model)
                fm.write(sm)

                # yp = model.predict(X)
                # print(yp)
                # m = metrics.flat_classification_report(
                #     y, yp, labels=list("BMES"), digits=4
                # )
                # print(m)
                # model_json = model.to_json()
                # with open("keras/B20-E60-F10-RU-Bi-RCT-CT-Bn-De.json", "w") as json_file:
                #     json_file.write(model_json)
                # model.save_weights("keras/B20-E60-F10-PU-Bi-RCT-CT-Bn-De-weights.h5")
                model.save("keras/%s.h5"%modelfile)
                print('FIN')

if MODE == 3:
    STATES = list("PN")
    with codecs.open('plain/ctrip_test.utf8', 'r', encoding='utf8') as ft:
        with codecs.open('baseline/ctrip_test_%s_states.txt'%modelfile, 'w', encoding='utf8') as fl:
            with codecs.open('model/ctrip_train_%s_model.pkl'%modelfile, 'rb') as fm:
                bm = fm.read()
                model = pickle.loads(bm)
                # model = load_model("keras/pretrained-extradim-dropout-bilstm-bn.h5")
                # model.summary()

                xlines = ft.readlines()
                X = []
                print('process X list.')
                counter = 0
                for line in xlines:
                    line = line.replace(" ", "").strip()
                    line = '\n' * (maxlen - len(line)) + line
                    X.append([getFeatures(line, i) for i in range(len(line))])
                    # X.append([rxdict.get(e, 0) for e in list(line)])
                    counter += 1
                    if counter % 1000 == 0 and counter != 0:
                        print('.')
                X = numpy.array(X)
                print(len(X), X.shape)

                yp = model.predict(X)
                print(yp.shape)
                for i in range(yp.shape[0]):
                    sl = yp[i]
                    # lens = len(xlines[i].strip())
                    # for s in sl[-lens:]:
                    i = numpy.argmax(sl)
                    fl.write(STATES[i])
                    # fl.write('\n')
                print('FIN')

