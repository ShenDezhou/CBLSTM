import codecs
import numpy

with codecs.open('ctrip_hotel_pos.utf8', 'r', encoding='utf8') as fp:
    with codecs.open('ctrip_hotel_neg.utf8', 'r', encoding='utf8') as fn:
        with codecs.open('ctrip_traintest.utf8', 'r', encoding='utf8') as fi:
            with codecs.open('ctrip_training.utf8', 'w', encoding='utf8') as ft:
                with codecs.open('ctrip_training_states.utf8', 'w', encoding='utf8') as fts:
                    with codecs.open('ctrip_test.utf8', 'w', encoding='utf8') as fte:
                        with codecs.open('ctrip_test_states.utf8', 'w', encoding='utf8') as ftes:
                            xplines = fp.readlines()
                            xnlines = fn.readlines()
                            yplines = ['P'] * len(xplines)
                            ynlines = ['N'] * len(xnlines)
                            yplines.extend(ynlines)
                            xplines.extend(xnlines)
                            filines = fi.readlines()
                            train = filines[0].strip().split("\t")
                            train = [int(i) for i in train]
                            xlines = numpy.array(xplines)[train]
                            ylines = numpy.array(yplines)[train]

                            xtrain = [line for line in xlines]
                            ytrain = [line for line in ylines]

                            test = filines[1].strip().split("\t")
                            test = [int(i) for i in test]
                            xlines = numpy.array(xplines)[test]
                            ylines = numpy.array(yplines)[test]

                            xtest = [line for line in xlines]
                            ytest = [line for line in ylines]
                            ft.writelines(xtrain)
                            fts.write("".join(ytrain))
                            fte.writelines(xtest)
                            ftes.write("".join(ytest))


print('FIN')