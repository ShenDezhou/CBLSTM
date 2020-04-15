import logging
import re
import string
import codecs

import joblib
import numpy
import sys, getopt

PKUTRDIC = '../pku_dic/pku_training_words.utf8'
PKUTEDIC = '../pku_dic/pku_test_words.utf8'
CASEDIC = '../pku_dic/contract_words.utf8'
MODELARCH = '../crf/contract4.pkl.gz'

try:
    opts, args = getopt.getopt(sys.argv, "hr:e:c:m:", ["pkutrdic=", "pkutedic=", "casedic=", "model="])
except getopt.GetoptError:
    print(opts)
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('pub_bilstm_bn.py -r <pkutrdic> -e <pkutedic> -c <casedic> -m <model>')
        sys.exit()
    elif opt in ("-r", "--pkutrdic"):
        PKUTRDIC = arg
    elif opt in ("-e", "--pkutedic"):
        PKUTEDIC = arg
    elif opt in ("-c", "--casedic"):
        CASEDIC = arg
    elif opt in ("-m", "--model"):
        MODELARCH = arg



class UBTRT_CRF:
    model = None
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

    def __init__(self):
        with codecs.open(PKUTRDIC, 'r', encoding='utf8') as fa:
            with codecs.open(PKUTEDIC, 'r', encoding='utf8') as fb:
                with codecs.open(CASEDIC, 'r', encoding='utf8') as fc:
                    lines = fa.readlines()
                    lines.extend(fb.readlines())
                    lines.extend(fc.readlines())
                    lines = [line.strip() for line in lines]
                    self.dicts.extend(lines)
                    # uni, pre, suf, long 这四个判断应该依赖外部词典，置信区间为95%，目前没有外部词典，就先用训练集词典来替代
                    self.unidicts.extend([line for line in lines if len(line) == 1 and re.search(u'[\u4e00-\u9fff]', line)])
                    self.predicts.extend(
                        [line[0] for line in lines if len(line) > 1 and re.search(u'[\u4e00-\u9fff]', line)])
                    self.predicts = self.getTopN(self.predicts)
                    self.sufdicts.extend(
                        [line[-1] for line in lines if len(line) > 1 and re.search(u'[\u4e00-\u9fff]', line)])
                    self.sufdicts = self.getTopN(self.sufdicts)
                    self.longdicts.extend([line for line in lines if len(line) > 3 and re.search(u'[\u4e00-\u9fff]', line)])
                    self.puncdicts.extend(string.punctuation)
                    self.puncdicts.extend(list("！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰–‘’‛“”„‟…‧﹏"))
                    self.digitsdicts.extend(string.digits)
                    self.chidigitsdicts.extend(list("零一二三四五六七八九十百千万亿兆〇零壹贰叁肆伍陆柒捌玖拾佰仟萬億兆"))
                    self.letterdicts.extend(string.ascii_letters)

                    somedicts = []
                    somedicts.extend(self.unidicts)
                    somedicts.extend(self.predicts)
                    somedicts.extend(self.sufdicts)
                    somedicts.extend(self.longdicts)
                    somedicts.extend(self.puncdicts)
                    somedicts.extend(self.digitsdicts)
                    somedicts.extend(self.chidigitsdicts)
                    somedicts.extend(self.letterdicts)
                    self.otherdicts.extend(set(self.dicts) - set(somedicts))

    def getTopN(self, dictlist):
        adict = {}
        for w in dictlist:
            adict[w] = adict.get(w, 0) + 1
        topN = max(adict.values())
        alist = [k for k, v in adict.items() if v >= topN * self.Thresholds]
        return alist



    def getCharType(self, ch):
        types = []

        dictofdicts = [self.puncdicts, self.digitsdicts, self.chidigitsdicts, self.letterdicts, self.unidicts, self.predicts, self.sufdicts]
        for i in range(len(dictofdicts)):
            if ch in dictofdicts[i]:
                types.append(i)
                break

        extradicts = [self.longdicts, self.otherdicts]
        for i in range(len(extradicts)):
            for word in extradicts[i]:
                if ch in word:
                    types.append(i + len(dictofdicts))
                    break
            if len(types) > 0:
                break


        assert len(types) == 1 or len(types) == 2, "{} {} {}".format(ch, len(types), types)
        return str(types[0])

    def safea(self, sentence, i):
        if i < 0:
            return ''
        if i >= len(sentence):
            return ''
        return sentence[i]

    def getNgram(self, sentence, i):
        ngrams = []
        for offset in [-2, -1, 0, 1, 2]:
            ngrams.append(self.safea(sentence, i + offset))

        for offset in [-2, -1, 0, 1]:
            ngrams.append(self.safea(sentence, i + offset) + self.safea(sentence, i + offset + 1))

        for offset in [-1, 0]:
            ngrams.append(
                self.safea(sentence, i + offset) + self.safea(sentence, i + offset + 1) + self.safea(sentence, i + offset + 2))
        return ngrams

    def getReduplication(self, sentence, i):
        reduplication = []
        for offset in [-2, -1]:
            if self.safea(sentence, i) == self.safea(sentence, i + offset):
                reduplication.append('1')
            else:
                reduplication.append('0')
        return reduplication

    def getType(self, sentence, i):
        types = []
        for offset in [-1, 0, 1]:
            types.append(self.getCharType(self.safea(sentence, i + offset)))
        types.append(
            self.getCharType(self.safea(sentence, i + offset - 1)) + self.getCharType(self.safea(sentence, i + offset)) + self.getCharType(
                self.safea(sentence, i + offset + 1)))
        return types

    def getFeatures(self, sentence, i):
        features = []
        features.extend(self.getNgram(sentence, i))
        features.extend(self.getReduplication(sentence, i))
        features.extend(self.getType(sentence, i))
        return features

    def getFeaturesDict(self, sentence, i):
        features = []
        features.extend(self.getNgram(sentence, i))
        features.extend(self.getReduplication(sentence, i))
        features.extend(self.getType(sentence, i))
        assert len(features) == 17
        featuresdic = dict([(str(j), features[j]) for j in range(len(features))])
        return featuresdic

    def loadCrf(self):
        self.model = joblib.load(MODELARCH)

    def predict(self, sentences):
        X = []
        print('process X list.')
        counter = 0
        for line in sentences:
            line = line.replace(" ", "").strip()
            X.append([self.getFeaturesDict(line, i) for i in range(len(line))])
            counter += 1
            if counter % 1000 == 0 and counter != 0:
                print('.')
        X = numpy.array(X)
        print(len(X), X.shape)

        yp = self.model.predict(X)
        print(yp)
        return yp


    def cut(self, sentences):
        statelines = self.predict(sentences)
        segments = []
        for linenumber in range(len(statelines)):
            stateline = statelines[linenumber]
            sentence = sentences[linenumber].strip()
            assert len(stateline) == len(sentence)
            if len(stateline) != len(sentence):
                print(linenumber)
            word = ""
            for i in range(len(stateline)):
                word+=sentence[i]
                if stateline[i] == 'E' or stateline[i] == 'S':
                    word+=" "
            segments.append(word.strip())
        return segments


logging.basicConfig(level=logging.INFO, format='%(asctime)-18s %(message)s')
l = logging.getLogger()

if __name__ == "__main__":
    l.info("create pub-bilstm-bn:")
    bilstm = UBTRT_CRF()
    l.info("load crf model:")
    bilstm.loadCrf()
    l.info("inference:")
    segs = bilstm.cut(["成都市武侯区人民法院", "（2015）武侯民初字第7043号"])
    l.info("inference done.")
    print(segs)

    l.info("inference:")

    PKUTEST = '../plain/contract_train_nospace.utf8'
    l.info("inference pkutest:")
    with codecs.open(PKUTEST, 'r', encoding='utf8') as fr:
        segs = bilstm.cut(fr.readlines())
    l.info("inference pkutest done.")
    print(segs)
