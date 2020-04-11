import codecs
import numpy

from PUB_BiLSTM_BN import PUB_BiLSTM_BN

bilstm = PUB_BiLSTM_BN()
bilstm.loadKeras()
print('model loaded')

with codecs.open('ctrip_hotel_pos.utf8', 'r', encoding='utf8') as fp:
    with codecs.open('ctrip_hotel_neg.utf8', 'r', encoding='utf8') as fn:
        with codecs.open('ctrip_traintest.utf8', 'r', encoding='utf8') as fi:
            with codecs.open('../ctrip_dic/ctrip_word.utf8', 'w', encoding='utf8') as fb:
                xplines = fp.readlines()
                xnlines = fn.readlines()
                yplines = ['P'] * len(xplines)
                ynlines = ['N'] * len(xnlines)
                yplines.extend(ynlines)
                xplines.extend(xnlines)
                train = fi.readlines()[0].strip().split("\t")
                train = [int(i) for i in train]
                xlines = numpy.array(xplines)[train]
                ylines = numpy.array(yplines)[train]

                xlines = [line.strip() for line in xlines]
                words = bilstm.cut(xlines)
                w_dic = []
                for line in words:
                    w_dic.extend(line.split(' '))
                words = list(set(w_dic))
                words = [word for word in words if len(word)>0]
                words.sort()
                for word in words:
                    fb.write(word+'\n')
print('FIN')