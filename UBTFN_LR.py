import logging

from keras.models import load_model
import codecs
import numpy
import sys, getopt

UNIGRAM = 'ctrip_dic/ctrip_dict.utf8'
BIGRAM = 'ctrip_dic/ctrip_bigram.utf8'
VECTOR = 'lr/ctrip_tfidfvec2.pkl.gz'
MODEL = 'lr/ctrip_comment2.pkl.gz'

try:
    opts, args = getopt.getopt(sys.argv, "hu:b:w:a:", ["unigram=", "bigram=", "word=", "arch="])
except getopt.GetoptError:
    print(opts)
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('pub_bilstm_bn.py -u <unigramfile> -b <bigramfile>  -w <wordfile> -a <modelfile>')
        sys.exit()
    elif opt in ("-u", "--unigram"):
        UNIGRAM = arg
    elif opt in ("-b", "--bigram"):
        BIGRAM = arg
    elif opt in ("-a", "--archandweight"):
        MODELARCH = arg
    elif opt in ("-w", "--word"):
        MODELWEIGHT = arg


class UB_TFN_LR:
    chars = []
    bigrams = []
    words = []
    rxdict = {}
    rbxdict = {}
    rwxdict = {}
    model = None
    maxlen = 976
    STATES = list("PN")

    def __init__(self):
        with codecs.open(UNIGRAM, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                for w in line:
                    if w == '\n':
                        continue
                    else:
                        self.chars.append(w)
        print(len(self.chars))

        with codecs.open(BIGRAM, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if len(line) > 0:
                    self.bigrams.append(line)
        print(len(self.bigrams))

        with codecs.open(WORD, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if len(line) > 0:
                    self.words.append(line)
        print(len(self.words))

        self.rxdict = dict(zip(self.chars, range(1, 1 + len(self.chars))))
        self.rxdict['\n'] = 0

        self.rbxdict = dict(zip(self.bigrams, range(1, 1 + len(self.bigrams))))
        self.rbxdict['\n'] = 0

        self.rwxdict = dict(zip(self.words, range(1, 1 + len(self.words))))
        self.rwxdict['\n'] = 0

    def safea(self, sentence, i):
        if i < 0:
            return '\n'
        if i >= len(sentence):
            return '\n'
        return sentence[i]

    def getUBgram(self, sentence, i):
        # 3 + 2 = 5
        ngrams = []
        for offset in [0, 1, 2]:
            ngrams.append(self.safea(sentence, i + offset))

        for offset in [0, 1]:
            ngrams.append(self.safea(sentence, i + offset) + self.safea(sentence, i + offset + 1))

        return ngrams

    def getUBgramVector(self, sentence, i):
        ngrams = self.getUBgram(sentence, i)
        ngramv = []
        for ngram in ngrams:
            if len(ngram) == 1:
                ngramv.append(self.rxdict.get(ngram, 0))
            if len(ngram) == 2:
                if '\n' in ngram:
                    ngramv.append(0)
                else:
                    ngramv.append(self.rbxdict.get(ngram, 0))
        return ngramv

    def getWord(self, sentence, i):
        # 1
        ngrams = []
        for offset in [0]:
            ngrams.append(
                self.safea(sentence, i + offset) + self.safea(sentence, i + offset + 1) + self.safea(sentence, i + offset + 2))
        return ngrams

    def getWordVector(self, sentence, i):
        ngrams = self.getWord(sentence, i)
        ngramv = []
        for ngram in ngrams:
            if len(ngram) == 3:
                if '\n' in ngram:
                    ngramv.append(0)
                else:
                    ngramv.append(self.rwxdict.get(ngram, 0))
        return ngramv

    def getFeatures(self, sentence, i):
        features = []
        features.extend(self.getUBgramVector(sentence, i))
        features.extend(self.getWordVector(sentence, i))
        assert len(features) == 6, (len(features), features)
        return features

    def loadKeras(self):
        self.model = load_model(MODEL)
        self.model.summary()

    def predict(self, sentences):
        X = []
        # print('process X list.')
        counter = 0
        for line in sentences:
            line = line.replace(" ", "").strip()
            line = '\n' * (self.maxlen - len(line)) + line
            X.append([self.getFeatures(line, i) for i in range(len(line))])
            counter += 1
            if counter % 1000 == 0 and counter != 0:
                print('.')
        X = numpy.array(X)
        # print(len(X), X.shape)

        yp = self.model.predict(X)
        print(yp.shape)

        ypi = numpy.argmax(yp, axis=1)
        print(ypi.shape)
        return ypi

    def opinion(self, sentences):
        states = self.predict(sentences)
        opinions = [self.STATES[s] for s in states]
        return opinions


logging.basicConfig(level=logging.INFO, format='%(asctime)-18s %(message)s')
l = logging.getLogger()

if __name__ == "__main__":
    l.info("create pubw-bilstm-bn:")
    bilstm = PUBW_BiLSTM_BN()
    l.info("load keras model:")
    bilstm.loadKeras()
    l.info("inference:")
    opis = bilstm.opinion(["哈哈哈。", "那酒店也就呵呵了。","到半夜竟然没暖气,怎么住啊????!!!!!!!!!!"])
    l.info("inference done.")
    print(opis)

    # l.info("inference:")
    # PKUTEST = 'plain/pku_test.utf8'
    # l.info("inference pkutest:")
    # with codecs.open(PKUTEST, 'r', encoding='utf8') as fr:
    #     opis = bilstm.opinion(fr.readlines())
    # l.info("inference pkutest done.")
    # print(opis)