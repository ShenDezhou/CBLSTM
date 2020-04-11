import logging

from keras.models import model_from_json, load_model
import codecs
import numpy
import sys, getopt

UNIGRAM = '../pku_dic/pku_dict.utf8'
BIGRAM = '../pku_dic/pku_bigram.utf8'
MODELARCH = 'G:\LSTM\keras\B20-E60-F5-PU-Bi-Bn-De.json'
MODELWEIGHT = "G:\LSTM\keras\B20-E60-F5-PU-Bi-Bn-De-weights.h5"
try:
    opts, args = getopt.getopt(sys.argv, "hu:b:a:w:", ["unigram=", "bigram=", "arch=", "weight="])
except getopt.GetoptError:
    print(opts)
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('pub_bilstm_bn.py -u <unigramfile> -b <bigramfile> -a <archfile> -w <weightfile>')
        sys.exit()
    elif opt in ("-u", "--unigram"):
        UNIGRAM = arg
    elif opt in ("-b", "--bigram"):
        BIGRAM = arg
    elif opt in ("-a", "--arch"):
        MODELARCH = arg
    elif opt in ("-w", "--weight"):
        MODELWEIGHT = arg


class PUB_BiLSTM_BN:
    chars = []
    bigrams = []
    rxdict = {}
    rbxdict = {}
    model = None
    maxlen = 1019
    STATES = list("BMES")

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

        self.rxdict = dict(zip(self.chars, range(1, 1 + len(self.chars))))
        self.rxdict['\n'] = 0

        self.rbxdict = dict(zip(self.bigrams, range(1, 1 + len(self.bigrams))))
        self.rbxdict['\n'] = 0

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

    def getFeatures(self, sentence, i):
        features = []
        features.extend(self.getUBgramVector(sentence, i))
        assert len(features) == 5, (len(features), features)
        return features

    def loadKeras(self):
        if MODELARCH:
            json_file = open(MODELARCH, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
            self.model.load_weights(MODELWEIGHT)
        else:
            self.model = load_model(MODELWEIGHT)
        self.model.summary()

    def predict(self, sentences):
        X = []
        print('process X list.')
        counter = 0
        for line in sentences:
            line = line.replace(" ", "").strip()
            line = '\n' * (self.maxlen - len(line)) + line
            X.append([self.getFeatures(line, i) for i in range(len(line))])
            counter += 1
            if counter % 1000 == 0 and counter != 0:
                print('.')
        X = numpy.array(X)
        print(len(X), X.shape)

        yp = self.model.predict(X)
        print(yp.shape)

        ys = []
        for i in range(yp.shape[0]):
            sl = yp[i]
            lens = len(sentences[i].strip())
            yss = []
            for s in sl[-lens:]:
                i = numpy.argmax(s)
                yss.append(self.STATES[i])
            ys.append("".join(yss))
        return ys

    def cut(self, sentences):
        statelines = self.predict(sentences)
        segments = []
        for linenumber in range(len(statelines)):
            stateline = statelines[linenumber].strip()
            sentence = sentences[linenumber].strip()
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
    bilstm = PUB_BiLSTM_BN()
    l.info("load keras model:")
    bilstm.loadKeras()
    l.info("inference:")
    segs = bilstm.cut(["我昨天去清华大学。", "他明天去北京大学，再后天去麻省理工大学。"])
    l.info("inference done.")
    print(segs)

    l.info("inference:")

    PKUTEST = 'plain/pku_test.utf8'
    l.info("inference pkutest:")
    with codecs.open(PKUTEST, 'r', encoding='utf8') as fr:
        segs = bilstm.cut(fr.readlines())
    l.info("inference pkutest done.")
    print(segs)