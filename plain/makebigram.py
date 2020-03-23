import codecs
import numpy

with codecs.open('ctrip_hotel_pos.utf8', 'r', encoding='utf8') as fp:
    with codecs.open('ctrip_hotel_neg.utf8', 'r', encoding='utf8') as fn:
        with codecs.open('ctrip_traintest.utf8', 'r', encoding='utf8') as fi:
            with codecs.open('../ctrip_dic/ctrip_bigram.utf8', 'w', encoding='utf8') as fb:
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

                bigrams = []
                for line in xlines:
                    line = line.replace(" ", "").strip()
                    chars = list(line)
                    if len(chars) < 2:
                        continue
                    for i in range(len(chars) - 1):
                        bigrams.append(chars[i] + chars[i + 1] + "\n")
                bigrams = list(set(bigrams))
                bigrams.sort()
                fb.writelines(bigrams)
print('FIN')